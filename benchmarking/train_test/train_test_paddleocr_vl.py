"""
Training and prediction script for PaddleOCR-VL for OCR/OMR.
"""

import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    PaddleOCRVLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from ..utils import (
    get_adaptive_num_workers,
    load_image_stems_from_json,
    save_text_predictions,
    get_augment_policy,
)

PROMPTS = {
    "ocr": "OCR:",
    "omr": "OMR:",
}
PAPER_GLOBAL_BATCH_SIZE = 128
PAPER_MAX_LR = 5e-6
PAPER_MIN_LR = 5e-7
PAPER_SFT_EPOCHS = 2
MAX_PIXELS = 2048 * 28 * 28
MAX_NEW_TOKENS = 256
UPSCALE_THRESHOLD = 1500
MIN_VAL_SAMPLES = 4


def _get_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_annotations(json_path: Path, debug: bool):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"][:5] if debug else data["annotations"]
    image_map = {image["id"]: image for image in data["images"]}
    return annotations, image_map


def _target_text(ann: dict) -> str:
    return (ann.get("description") or ann.get("text") or "").strip()


def _crop_line_image(data_dir: Path, image_info: dict, ann: dict) -> Image.Image:
    image_path = data_dir / image_info["file_name"]
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
    image = Image.fromarray(line_image)

    if image.width < UPSCALE_THRESHOLD and image.height < UPSCALE_THRESHOLD:
        image = image.resize(
            (image.width * 2, image.height * 2), Image.Resampling.LANCZOS
        )

    return image


def _ensure_tensors(inputs) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        output[key] = value
    return output


def _prepare_inputs(processor, image: Image.Image, prompt: str, target_text: Optional[str] = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=target_text is None,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": MAX_PIXELS,
            }
        },
    )
    return _ensure_tensors(inputs)


class PaddleOCRVLTrainDataset(Dataset):
    def __init__(
        self,
        json_file,
        data_dir,
        processor,
        task: str,
        debug: bool = False,
        augment_transform=None,
    ):
        self.json_file = Path(json_file)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.task = task
        self.prompt = PROMPTS[task]
        self.augment_transform = augment_transform

        annotations, image_map = _load_annotations(self.json_file, debug)
        self.samples = []
        for ann in annotations:
            text = _target_text(ann)
            if not text:
                continue
            image_info = image_map.get(ann["image_id"])
            if image_info is None:
                continue
            self.samples.append(
                {
                    "ann": ann,
                    "image_info": image_info,
                    "target_text": text,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = _crop_line_image(self.data_dir, sample["image_info"], sample["ann"])
        if self.augment_transform:
            image = self.augment_transform(image)

        prompt_inputs = _prepare_inputs(self.processor, image, self.prompt)
        full_inputs = _prepare_inputs(
            self.processor,
            image,
            self.prompt,
            target_text=sample["target_text"],
        )

        prompt_length = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].squeeze(0).clone().long()
        labels[:prompt_length] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return {
            "input_ids": full_inputs["input_ids"].squeeze(0).long(),
            "attention_mask": full_inputs["attention_mask"].squeeze(0).long(),
            "pixel_values": full_inputs["pixel_values"].float(),
            "image_grid_thw": full_inputs["image_grid_thw"].reshape(-1, 3).long(),
            "labels": labels,
        }


class PaddleOCRVLPredictDataset(Dataset):
    def __init__(self, json_file, data_dir, processor, task: str, debug: bool = False):
        self.json_file = Path(json_file)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.task = task
        self.prompt = PROMPTS[task]

        annotations, image_map = _load_annotations(self.json_file, debug)
        self.samples = []
        for ann in annotations:
            image_info = image_map.get(ann["image_id"])
            if image_info is None:
                continue
            self.samples.append(
                {
                    "ann": ann,
                    "image_info": image_info,
                    "image_stem": Path(image_info["file_name"]).stem,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = _crop_line_image(self.data_dir, sample["image_info"], sample["ann"])
        inputs = _prepare_inputs(self.processor, image, self.prompt)

        return {
            "input_ids": inputs["input_ids"].squeeze(0).long(),
            "attention_mask": inputs["attention_mask"].squeeze(0).long(),
            "pixel_values": inputs["pixel_values"].float(),
            "image_grid_thw": inputs["image_grid_thw"].reshape(-1, 3).long(),
            "image_stem": sample["image_stem"],
        }


class PaddleOCRVLCollator:
    def __init__(self, pad_token_id: int, with_labels: bool, pad_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.with_labels = with_labels
        self.pad_side = pad_side

    def __call__(self, features):
        max_len = max(feature["input_ids"].shape[0] for feature in features)

        input_ids = []
        attention_masks = []
        labels = []
        pixel_values = []
        image_grid_thw = []
        image_stems = []

        for feature in features:
            seq_len = feature["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_id_pad = torch.full(
                (pad_len,), self.pad_token_id, dtype=feature["input_ids"].dtype
            )
            attention_pad = torch.zeros(
                (pad_len,), dtype=feature["attention_mask"].dtype
            )

            if self.pad_side == "left":
                input_ids.append(torch.cat([input_id_pad, feature["input_ids"]], dim=0))
                attention_masks.append(
                    torch.cat([attention_pad, feature["attention_mask"]], dim=0)
                )
                if self.with_labels:
                    label_pad = torch.full((pad_len,), -100, dtype=feature["labels"].dtype)
                    labels.append(torch.cat([label_pad, feature["labels"]], dim=0))
            else:
                input_ids.append(torch.cat([feature["input_ids"], input_id_pad], dim=0))
                attention_masks.append(
                    torch.cat([feature["attention_mask"], attention_pad], dim=0)
                )
                if self.with_labels:
                    label_pad = torch.full((pad_len,), -100, dtype=feature["labels"].dtype)
                    labels.append(torch.cat([feature["labels"], label_pad], dim=0))

            pixel_values.append(feature["pixel_values"])
            image_grid_thw.append(feature["image_grid_thw"])
            if "image_stem" in feature:
                image_stems.append(feature["image_stem"])

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        }

        if self.with_labels:
            batch["labels"] = torch.stack(labels, dim=0)
        if image_stems:
            batch["image_stems"] = image_stems
        return batch


class _PaddleOCRVLTrainer(Trainer):
    def __init__(self, *args, min_learning_rate: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_learning_rate = float(min_learning_rate)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        if optimizer is None:
            raise RuntimeError("Optimizer must be created before scheduler")

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps),
            eta_min=self._min_learning_rate,
        )
        return self.lr_scheduler


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _resolve_batch_size(args, train_json: Path, image_root: Path) -> int:
    if not torch.cuda.is_available() or args.debug:
        return 1

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 24:
        return 2
    return 1


def load_model(model_identifier, load_model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype()

    if load_model_path and Path(load_model_path).exists():
        model_source = str(load_model_path)
        print(f"   Fine-tuning from checkpoint: {model_source}")
    else:
        model_source = model_identifier
        print(f"   Loading base model: {model_source}")

    processor = AutoProcessor.from_pretrained(model_identifier, use_fast=False)

    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        model_source,
        dtype=dtype,
    ).to(device)
    return model, processor, device


def make_datasets(args, train_json, val_json, processor):
    augment_transform = None
    if args.augment:
        print("   💪 Augmentation enabled.")
        augment_transform = get_augment_policy()

    train_dataset = PaddleOCRVLTrainDataset(
        train_json,
        args.data_dir or args.train_dir,
        processor,
        task=args.task,
        debug=args.debug,
        augment_transform=augment_transform,
    )
    val_dataset = PaddleOCRVLTrainDataset(
        val_json,
        args.data_dir or args.train_dir,
        processor,
        task=args.task,
        debug=args.debug,
    )
    return train_dataset, val_dataset


def train(args, model, processor, train_dataset, val_dataset):
    artifacts_path = Path(tempfile.mkdtemp(prefix="paddleocr_vl_artifacts_"))
    image_root = args.data_dir or args.train_dir
    train_json = Path(train_dataset.json_file)

    per_device_batch_size = _resolve_batch_size(args, train_json, image_root)
    gradient_accumulation_steps = max(
        1, math.ceil(PAPER_GLOBAL_BATCH_SIZE / per_device_batch_size)
    )
    adaptive_workers = 0 if args.debug else min(4, get_adaptive_num_workers(train_json, image_root))

    print(f"   📊 Per-device batch size: {per_device_batch_size}")
    print(f"   🔁 Gradient accumulation: {gradient_accumulation_steps}")
    print(
        f"   📈 LR schedule: cosine {PAPER_MAX_LR:.1e} → {PAPER_MIN_LR:.1e}"
    )
    print(f"   👷 Data workers: {adaptive_workers}")

    has_sufficient_val_data = len(val_dataset) >= MIN_VAL_SAMPLES
    if not has_sufficient_val_data:
        print(
            f"⚠️  Validation set too small ({len(val_dataset)} samples), disabling validation"
        )

    dtype = _get_dtype()
    training_args = TrainingArguments(
        output_dir=str(artifacts_path),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=adaptive_workers,
        num_train_epochs=1 if args.debug else PAPER_SFT_EPOCHS,
        learning_rate=PAPER_MAX_LR,
        save_strategy="epoch",
        eval_strategy="epoch" if has_sufficient_val_data else "no",
        logging_strategy="steps",
        logging_steps=5 if args.debug else 25,
        load_best_model_at_end=has_sufficient_val_data,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=torch.cuda.is_available(),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )

    trainer = _PaddleOCRVLTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_sufficient_val_data else None,
        data_collator=PaddleOCRVLCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            with_labels=True,
            pad_side="right",
        ),
        min_learning_rate=PAPER_MIN_LR,
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


def predict(args, model, processor, device, output_dir, test_json):
    dataset = PaddleOCRVLPredictDataset(
        test_json,
        args.data_dir or args.test_dir,
        processor,
        task=args.task,
        debug=args.debug,
    )
    collator = PaddleOCRVLCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        with_labels=False,
        pad_side="left",
    )
    batch_size = 1
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_gb >= 24 and not args.debug:
            batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    expected_stems = load_image_stems_from_json(Path(test_json))
    image_predictions = defaultdict(list)

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            image_stems = batch.pop("image_stems")
            batch = _move_batch_to_device(batch, device)
            context_length = batch["input_ids"].shape[1]
            outputs = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS)

            for image_stem, output_ids in zip(image_stems, outputs):
                generated = output_ids[context_length:]
                prediction = processor.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                image_predictions[image_stem].append(prediction)
                if args.debug:
                    print(f"   [batch {batch_idx}] {image_stem}: {prediction[:80]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_stem in expected_stems:
        text = " ".join(image_predictions.get(image_stem, []))
        save_text_predictions(image_stem, text, output_dir)

    print(f"✅ Predictions saved to {output_dir}")


def train_test_paddleocr_vl(
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
        raise ValueError("paddleocr_vl supports only task=ocr|omr")

    print(f"📦 Loading PaddleOCR-VL model ({model_identifier})...")
    model, processor, device = load_model(model_identifier, load_model_path)

    train_dataset, val_dataset = make_datasets(args, train_json, val_json, processor)

    print(f"🚀 Training on {device}...")
    trainer, artifacts_path = train(args, model, processor, train_dataset, val_dataset)
    save_model(trainer, processor, save_model_path)

    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(args, trainer.model, processor, device, output_dir, test_json)
    else:
        print("⏭️  No test split provided; skipping prediction.")

    if artifacts_path.exists():
        shutil.rmtree(artifacts_path)
