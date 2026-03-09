"""
Training and prediction script for TrOCR for OCR/OMR.
"""

import tempfile
import json
from pathlib import Path
import shutil

import cv2
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from ..utils import (
    get_augment_policy,
    get_adaptive_batch_size,
    get_adaptive_num_workers,
)


class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def forward(self, pixel_values=None, labels=None, **kwargs):
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")
        return super().forward(pixel_values=pixel_values, labels=labels, **kwargs)


class TrOCRDataCollator:
    """Custom data collator that creates decoder_input_ids for teacher forcing."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features]).float()
        labels = torch.stack([f["labels"] for f in features]).long()

        # Create decoder_input_ids by shifting labels right
        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = self.processor.tokenizer.cls_token_id

        # Replace padding token id with -100 (ignored in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }


class _HuttnerOneCycleTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with AdamW + OneCycleLR (Huttner hyperparameters)."""

    def __init__(
        self,
        *args,
        onecycle_max_lr: float,
        onecycle_pct_start: float,
        onecycle_base_momentum: float,
        onecycle_max_momentum: float,
        onecycle_initial_lr: float,
        onecycle_final_div_factor: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._onecycle_max_lr = float(onecycle_max_lr)
        self._onecycle_pct_start = float(onecycle_pct_start)
        self._onecycle_base_momentum = float(onecycle_base_momentum)
        self._onecycle_max_momentum = float(onecycle_max_momentum)
        self._onecycle_initial_lr = float(onecycle_initial_lr)
        self._onecycle_final_div_factor = float(onecycle_final_div_factor)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        wd = 1e-4
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self._onecycle_max_lr,
            weight_decay=wd,
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        if optimizer is None:
            raise RuntimeError("Optimizer must be created before scheduler")

        if self._onecycle_initial_lr <= 0:
            raise ValueError("onecycle_initial_lr must be > 0")
        div_factor = self._onecycle_max_lr / self._onecycle_initial_lr

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self._onecycle_max_lr,
            total_steps=num_training_steps,
            pct_start=self._onecycle_pct_start,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=self._onecycle_base_momentum,
            max_momentum=self._onecycle_max_momentum,
            div_factor=div_factor,
            final_div_factor=self._onecycle_final_div_factor,
        )
        return self.lr_scheduler


class TrOCRDataset(Dataset):
    """Dataset for TrOCR, loading line images from COCO-style annotations."""

    def __init__(
        self,
        json_file,
        data_dir,
        processor,
        max_target_length=128,
        debug=False,
        annotations_to_use=None,
        image_id_map=None,
        augment_transform=None,
    ):
        self.json_file = Path(json_file) if json_file else None
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.debug = debug
        self.augment_transform = augment_transform

        if annotations_to_use is not None:
            self.annotations = annotations_to_use
            self.image_id_to_info = image_id_map
        elif self.json_file is not None:
            with open(self.json_file, "r") as f:
                data = json.load(f)
            self.annotations = data["annotations"]
            self.image_id_to_info = {img["id"]: img for img in data["images"]}
        else:
            self.annotations = []
            self.image_id_to_info = {}

        if self.debug:
            self.annotations = self.annotations[:5]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        text = ann.get("description") or ann.get("text") or ""
        image_id = ann["image_id"]
        if self.image_id_to_info:
            image_info = self.image_id_to_info[image_id]
            original_image_path = self.data_dir / image_info["file_name"]
        else:
            raise ValueError("No image info available")

        # Load the full image
        full_image = cv2.imread(str(original_image_path))
        if full_image is None:
            raise FileNotFoundError(f"Could not read image: {original_image_path}")

        # Crop the line image using bbox [x, y, width, height] with margin
        x, y, w, h = [int(c) for c in ann["bbox"]]
        margin = 8
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(full_image.shape[1] - x, w + 2 * margin)
        h = min(full_image.shape[0] - y, h + 2 * margin)

        line_image = full_image[y : y + h, x : x + w]
        # Convert BGR to RGB, as TrOCRProcessor expects RGB
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(line_image)

        if self.augment_transform:
            pil_image = self.augment_transform(pil_image)

        # Process image and text together (like ablation study)
        encoding = self.processor(
            images=pil_image,
            text=text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension and convert to proper format
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}


def train(args, model, processor, train_dataset, val_dataset):
    from ..evaluation import calculate_wer_cer

    artifacts_path = Path(tempfile.mkdtemp(prefix="trocr_artifacts_"))

    # OLD: Calculate adaptive parameters (Huttner et al. 2025: batch_size=8, max_lr=5.5e-6)
    # OLD: Scale LR linearly with batch size: lr = base_lr * (batch_size / base_batch_size)
    train_json = Path(train_dataset.json_file)
    image_root = args.data_dir or args.train_dir

    base_batch_size = 8  # Huttner et al. 2025
    base_lr = 5.5e-6  # Huttner et al. 2025 (One-Cycle max LR)

    adaptive_batch = get_adaptive_batch_size(
        train_json, image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, image_root)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    # Tune formulas to match Huttner hyperparameters for the reference run:
    # ./run.sh --framework trocr --task ocr --model-name "small" --edition diplomatic --data-dir ./data/I-Ct_91
    # Reference output: adaptive batch size 5 -> target 8, adaptive workers 4 -> target 2.
    batch_size_scale = 1.6
    worker_scale = 0.5
    adaptive_batch = max(1, int(adaptive_batch * batch_size_scale))
    adaptive_workers = max(0, int(adaptive_workers * worker_scale))
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.2e} (base: {base_lr:.2e})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    # NEW (disabled): Match train_eval_trocr.py - use simple fixed batch size based on CUDA availability
    # and fixed learning rate without scaling
    # adaptive_batch = 8 if torch.cuda.is_available() else 4
    # adaptive_workers = 2 if torch.cuda.is_available() else 0

    min_val_samples = 4  # Minimum samples needed for reliable eval_loss
    val_dataset_size = len(val_dataset)
    has_sufficient_val_data = val_dataset_size >= min_val_samples

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids[pred_ids < 0] = processor.tokenizer.pad_token_id
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        wer, cer = calculate_wer_cer(pred_str, labels_str)
        return {"wer": wer, "cer": cer}

    if not has_sufficient_val_data:
        print(
            f"⚠️  Validation set too small ({val_dataset_size} samples), disabling validation"
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(artifacts_path),
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,
            per_device_train_batch_size=adaptive_batch,
            per_device_eval_batch_size=adaptive_batch,
            gradient_accumulation_steps=1,
            dataloader_num_workers=adaptive_workers,
            fp16=False,
            num_train_epochs=1 if args.debug else 50,
            save_strategy="epoch",
            eval_strategy="no",
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=scaled_lr,
            warmup_ratio=0.0,
            weight_decay=0.0,
            label_smoothing_factor=0.0,
            load_best_model_at_end=False,
            greater_is_better=False,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=True,
        )
        eval_dataset = None
    else:
        print(f"✅ Validation set has {len(val_dataset)} samples, enabling validation")
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(artifacts_path),
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,
            per_device_train_batch_size=adaptive_batch,
            per_device_eval_batch_size=adaptive_batch,
            gradient_accumulation_steps=1,
            dataloader_num_workers=adaptive_workers,
            fp16=False,
            num_train_epochs=1 if args.debug else 50,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=scaled_lr,
            warmup_ratio=0.0,
            weight_decay=0.0,
            label_smoothing_factor=0.0,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=True,
        )
        eval_dataset = val_dataset

    onecycle_max_lr = 3e-5
    onecycle_pct_start = 0.1
    onecycle_base_momentum = 0.85
    onecycle_max_momentum = 0.95
    onecycle_initial_lr = 1e-9
    onecycle_final_div_factor = 2.2e4

    trainer_kwargs = dict(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        data_collator=TrOCRDataCollator(processor),
    )

    # Only add eval_dataset if we have one
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=1 if args.debug else 15)
        ]

    trainer = _HuttnerOneCycleTrainer(
        **trainer_kwargs,
        onecycle_max_lr=onecycle_max_lr,
        onecycle_pct_start=onecycle_pct_start,
        onecycle_base_momentum=onecycle_base_momentum,
        onecycle_max_momentum=onecycle_max_momentum,
        onecycle_initial_lr=onecycle_initial_lr,
        onecycle_final_div_factor=onecycle_final_div_factor,
    )

    trainer.train()
    return trainer, artifacts_path


def predict(args, model, processor, output_dir, test_json, trainer):
    test_dataset = TrOCRDataset(
        test_json, args.data_dir or args.test_dir, processor, debug=args.debug
    )
    test_results = trainer.predict(test_dataset)

    pred_ids = test_results.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Group predictions by image
    annotations = (
        test_dataset.annotations[:100] if args.debug else test_dataset.annotations
    )
    image_predictions = {}
    for i, ann in enumerate(annotations):
        image_id = ann["image_id"]
        if test_dataset.image_id_to_info:
            image_info = test_dataset.image_id_to_info[image_id]
            image_name = Path(image_info["file_name"]).stem
        else:
            image_name = f"image_{i}"

        if image_name not in image_predictions:
            image_predictions[image_name] = []
        image_predictions[image_name].append(pred_str[i])

    # Write one prediction file per image
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_name, lines in image_predictions.items():
        output_file = output_dir / f"{image_name}.pred.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(lines))


def save_model(trainer, save_model_path):
    if save_model_path:
        try:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_model_path))
            print(f"   💾 Saved model to standardized path: {save_model_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy model to standardized path: {e}")


def load_model(model_identifier, load_model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if load_model_path and load_model_path.exists():
        model_to_load = load_model_path
        print(f"   Fine-tuning from sequential model: {model_to_load}")
    else:
        model_to_load = model_identifier
        print(f"   Loading base model: {model_to_load}")

    processor = TrOCRProcessor.from_pretrained(model_identifier)
    model = CustomVisionEncoderDecoderModel.from_pretrained(model_to_load)
    model.to(device)

    # Configure model for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return model, processor


def make_datasets(args, train_json, val_json, test_json, processor):
    augment_transform = None
    if args.augment:
        print("   💪 Augmentation enabled.")
        augment_transform = get_augment_policy()

    train_dataset = TrOCRDataset(
        train_json,
        args.data_dir or args.train_dir,
        processor,
        debug=args.debug,
        augment_transform=augment_transform,
    )
    val_dataset = TrOCRDataset(
        val_json, args.data_dir or args.train_dir, processor, debug=args.debug
    )

    return train_dataset, val_dataset


def train_test_trocr(
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
    task = "omr" if "omr" in str(train_json) else "ocr"

    # --- 1. Load processor, model, and datasets ---
    print(f"📦 Loading model ({model_identifier})...")
    model, processor = load_model(model_identifier, load_model_path)

    train_dataset, val_dataset = make_datasets(
        args, train_json, val_json, test_json, processor
    )

    # --- 2. Train the model ---
    print(
        f"🚀 Training on {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}..."
    )
    trainer, artifacts_path = train(args, model, processor, train_dataset, val_dataset)

    save_model(trainer, save_model_path)

    # --- 3. Generate predictions ---
    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(args, model, processor, output_dir, test_json, trainer)

    # Clean up training artifacts after predictions are complete
    if artifacts_path and artifacts_path.exists():
        shutil.rmtree(artifacts_path)

    print(f"✅ Predictions saved to {output_dir}")
