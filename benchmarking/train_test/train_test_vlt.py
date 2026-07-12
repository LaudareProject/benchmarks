"""
Training and prediction script for VLT for OCR/OMR.
"""

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
)

from ..utils import (
    get_adaptive_batch_size,
    get_adaptive_num_workers,
    get_augment_policy,
    load_image_stems_from_json,
    save_text_predictions,
)
from .train_test_trocr import (
    CustomVisionEncoderDecoderModel,
    TrOCRDataCollator,
    _HuttnerOneCycleTrainer,
)

MAX_TARGET_LENGTH = 128
MAX_NEW_TOKENS = 128
MIN_VAL_SAMPLES = 4
TTA_ROUNDS = 5


class VLTDataset(Dataset):
    """Dataset for VLT OCR/OMR line recognition."""

    def __init__(
        self,
        json_file,
        data_dir,
        processor,
        debug=False,
        augment_transform=None,
        tta_rounds: int = 1,
    ):
        self.json_file = Path(json_file) if json_file else None
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.debug = debug
        self.augment_transform = augment_transform
        self.tta_rounds = max(1, int(tta_rounds))

        if self.json_file is not None:
            with open(self.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            annotations = data.get("annotations", [])
            image_map = {image["id"]: image for image in data.get("images", [])}
        else:
            annotations = []
            image_map = {}

        if self.debug:
            annotations = annotations[:5]

        self.base_samples = []
        for sample_id, ann in enumerate(annotations):
            image_info = image_map.get(ann.get("image_id"))
            if image_info is None:
                continue

            self.base_samples.append(
                {
                    "sample_id": sample_id,
                    "ann": ann,
                    "image_info": image_info,
                    "image_stem": Path(image_info["file_name"]).stem,
                    "text": (ann.get("description") or ann.get("text") or "").strip(),
                }
            )

        self.samples = []
        for _ in range(self.tta_rounds):
            self.samples.extend(self.base_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ann = sample["ann"]
        image_info = sample["image_info"]
        image_path = self.data_dir / image_info["file_name"]

        full_image = cv2.imread(str(image_path))
        if full_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        x, y, w, h = [int(value) for value in ann["bbox"]]
        margin = 8
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(full_image.shape[1] - x, w + 2 * margin)
        h = min(full_image.shape[0] - y, h + 2 * margin)

        line_image = full_image[y : y + h, x : x + w]
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(line_image)

        if self.augment_transform is not None:
            pil_image = self.augment_transform(pil_image)

        encoding = self.processor(
            images=pil_image,
            text=sample["text"],
            padding="max_length",
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"].squeeze(0),
            "sample_id": sample["sample_id"],
            "image_stem": sample["image_stem"],
        }


class VLTPredictCollator:
    def __call__(self, features):
        return {
            "pixel_values": torch.stack([feature["pixel_values"] for feature in features]).float(),
            "sample_ids": [feature["sample_id"] for feature in features],
            "image_stems": [feature["image_stem"] for feature in features],
        }


def _freeze_decoder(model):
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False


def _get_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _resolve_batch_size(args) -> int:
    if args.debug or not torch.cuda.is_available():
        return 1
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 2 if total_gb >= 24 else 1


def _consensus_prediction(candidates):
    if not candidates:
        return ""

    candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
    if not candidates:
        return ""

    counter = Counter(candidates)
    candidate, count = counter.most_common(1)[0]
    if count > 1:
        return candidate

    best_candidate = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        score = sum(
            SequenceMatcher(None, candidate, other).ratio() for other in candidates
        ) / len(candidates)
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate


def load_model(model_identifier, load_model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if load_model_path and Path(load_model_path).exists():
        model_source = str(load_model_path)
        processor_source = str(load_model_path)
        print(f"   Fine-tuning from checkpoint: {model_source}")
    else:
        model_source = model_identifier
        processor_source = model_identifier
        print(f"   Loading base model: {model_source}")

    processor = TrOCRProcessor.from_pretrained(processor_source)
    model = CustomVisionEncoderDecoderModel.from_pretrained(model_source)
    model.to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if load_model_path and Path(load_model_path).exists():
        _freeze_decoder(model)
        print("   🔒 Decoder frozen for checkpoint adaptation")

    return model, processor, device


def make_datasets(args, train_json, val_json, processor):
    augment_transform = None
    if args.augment:
        print("   💪 Augmentation enabled.")
        augment_transform = get_augment_policy()

    train_dataset = VLTDataset(
        train_json,
        args.data_dir or args.train_dir,
        processor,
        debug=args.debug,
        augment_transform=augment_transform,
    )
    val_dataset = VLTDataset(
        val_json,
        args.data_dir or args.train_dir,
        processor,
        debug=args.debug,
    )
    return train_dataset, val_dataset


def train(args, model, processor, train_dataset, val_dataset):
    from ..evaluation import calculate_wer_cer

    artifacts_path = Path(tempfile.mkdtemp(prefix="vlt_artifacts_"))
    train_json = Path(train_dataset.json_file)
    image_root = args.data_dir or args.train_dir

    base_batch_size = 8
    base_lr = 3.5e-6

    adaptive_batch = get_adaptive_batch_size(
        train_json, image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, image_root)
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.2e} (base: {base_lr:.2e})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    has_sufficient_val_data = len(val_dataset) >= MIN_VAL_SAMPLES
    if not has_sufficient_val_data:
        print(
            f"⚠️  Validation set too small ({len(val_dataset)} samples), disabling validation"
        )

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        wer, cer = calculate_wer_cer(pred_str, labels_str)
        return {"wer": wer, "cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(artifacts_path),
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        per_device_train_batch_size=adaptive_batch,
        per_device_eval_batch_size=adaptive_batch,
        gradient_accumulation_steps=1,
        dataloader_num_workers=adaptive_workers,
        fp16=False,
        num_train_epochs=1 if args.debug else 40,
        save_strategy="epoch",
        eval_strategy="epoch" if has_sufficient_val_data else "no",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=scaled_lr,
        warmup_ratio=0.0,
        weight_decay=0.0,
        label_smoothing_factor=0.0,
        load_best_model_at_end=has_sufficient_val_data,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=True,
    )

    trainer_kwargs = dict(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        data_collator=TrOCRDataCollator(processor),
    )

    if has_sufficient_val_data:
        trainer_kwargs["eval_dataset"] = val_dataset
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=1 if args.debug else 15)
        ]

    trainer = _HuttnerOneCycleTrainer(
        **trainer_kwargs,
        onecycle_max_lr=2.5e-5,
        onecycle_pct_start=0.1,
        onecycle_base_momentum=0.85,
        onecycle_max_momentum=0.95,
        onecycle_initial_lr=1e-9,
        onecycle_final_div_factor=2.2e4,
    )

    trainer.train()
    return trainer, artifacts_path


def save_model(trainer, processor, save_model_path):
    if not save_model_path:
        return

    try:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_model_path))
        processor.save_pretrained(str(save_model_path))
        print(f"   💾 Saved model to standardized path: {save_model_path}")
    except Exception as e:
        print(f"   ⚠️  Failed to save model to standardized path: {e}")


def predict(args, model, processor, output_dir, test_json):
    tta_rounds = 1 if args.debug else TTA_ROUNDS
    augment_transform = get_augment_policy() if tta_rounds > 1 else None

    dataset = VLTDataset(
        test_json,
        args.data_dir or args.test_dir,
        processor,
        debug=args.debug,
        augment_transform=augment_transform,
        tta_rounds=tta_rounds,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=_resolve_batch_size(args),
        shuffle=False,
        num_workers=0,
        collate_fn=VLTPredictCollator(),
    )

    expected_stems = load_image_stems_from_json(Path(test_json))
    line_votes = defaultdict(list)

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            sample_ids = batch.pop("sample_ids")
            image_stems = batch.pop("image_stems")
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            outputs = model.generate(
                pixel_values=batch["pixel_values"],
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
            )
            pred_str = processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for sample_id, image_stem, prediction in zip(
                sample_ids, image_stems, pred_str
            ):
                line_votes[sample_id].append(prediction.strip())
                if args.debug:
                    print(f"   [batch {batch_idx}] {image_stem}: {prediction[:80]}")

    line_predictions = {}
    for sample in dataset.base_samples:
        line_predictions[sample["sample_id"]] = _consensus_prediction(
            line_votes.get(sample["sample_id"], [])
        )

    page_predictions = defaultdict(list)
    for sample in dataset.base_samples:
        page_predictions[sample["image_stem"]].append(
            line_predictions.get(sample["sample_id"], "")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_stem in expected_stems:
        text = " ".join(page_predictions.get(image_stem, []))
        save_text_predictions(image_stem, text, output_dir)

    print(f"✅ Predictions saved to {output_dir}")


def train_test_vlt(
    args,
    is_train_test_mode,
    is_sequential,
    output_dir,
    train_json,
    val_json,
    test_json,
    save_model_path,
    load_model_path,
    model_identifier,
):
    if args.task not in {"ocr", "omr"}:
        raise ValueError("vlt supports only task=ocr|omr")

    print(f"📦 Loading VLT model ({model_identifier})...")
    model, processor, device = load_model(model_identifier, load_model_path)

    train_dataset, val_dataset = make_datasets(args, train_json, val_json, processor)

    print(f"🚀 Training on {device}...")
    trainer, artifacts_path = train(args, model, processor, train_dataset, val_dataset)

    save_model(trainer, processor, save_model_path)

    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(args, model, processor, output_dir, test_json)

    if artifacts_path and artifacts_path.exists():
        shutil.rmtree(artifacts_path)

    print(f"✅ Predictions saved to {output_dir}")
