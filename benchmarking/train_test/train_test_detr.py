"""
Training and prediction script for DETR for layout detection.
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import BoundingBoxFormat
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..annotations.ann_handler import create_new_pagexml_file
from ..utils import (
    get_augment_policy,
    get_adaptive_batch_size,
    get_adaptive_num_workers,
)


class DetrCocoDataset(Dataset):
    def __init__(
        self,
        json_file,
        data_dir,
        image_processor,
        cat_id_map,
        debug=False,
        data_to_use=None,
        transforms=None,
    ):
        self.json_file = Path(json_file) if json_file else None
        self.data_dir = Path(data_dir)
        self.image_processor = image_processor
        self.cat_id_map = cat_id_map
        self.debug = debug
        self.transforms = transforms

        if data_to_use:
            self.data = data_to_use
        elif self.json_file:
            with open(self.json_file, "r") as f:
                self.data = json.load(f)
        else:
            self.data = {"images": [], "annotations": []}

        if debug:
            self.image_info = {img["id"]: img for img in self.data["images"][:10]}
        else:
            self.image_info = {img["id"]: img for img in self.data["images"]}
        self.image_annotations = {img["id"]: [] for img in self.data["images"]}
        for ann in self.data["annotations"]:
            if ann["category_id"] in self.cat_id_map:
                self.image_annotations[ann["image_id"]].append(ann)

        self.image_ids = list(self.image_info.keys())

        # Filter out images that have no annotations left after filtering
        self.image_ids = [
            img_id for img_id in self.image_ids if self.image_annotations[img_id]
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        img_path = self.data_dir / image_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        annotations = self.image_annotations[image_id]

        if self.transforms and annotations:
            boxes = [ann["bbox"] for ann in annotations]
            boxes = F.convert_bounding_box_format(
                torch.tensor(boxes),
                old_format=BoundingBoxFormat.XYWH,
                new_format=BoundingBoxFormat.XYXY,
            )
            labels = [self.cat_id_map[ann["category_id"]] for ann in annotations]

            tv_target = {
                "boxes": boxes,
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

            image, tv_target = self.transforms(image, tv_target)

            aug_boxes = F.convert_bounding_box_format(
                tv_target["boxes"],
                old_format=BoundingBoxFormat.XYXY,
                new_format=BoundingBoxFormat.XYWH,
            )
            for i, ann in enumerate(annotations):
                ann["bbox"] = aug_boxes[i].tolist()

        # Format annotations for DetrImageProcessor
        target = {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": self.cat_id_map[ann["category_id"]],
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0,
                    "id": ann.get("id", idx),
                }
                for ann in annotations
            ],
        }

        return image, target


class DataCollator:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        batch = self.image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
        )
        return batch


def train(
    args,
    model,
    train_dataset,
    val_dataset,
    image_processor,
    artifacts_dir,
    train_json,
):
    # Calculate adaptive parameters with LR scaling
    train_json = Path(train_json)
    image_root = args.data_dir or args.train_dir

    base_batch_size = 2  # DETR base batch size
    base_lr = 2.5e-5  # DETR base learning rate

    adaptive_batch = get_adaptive_batch_size(
        train_json, image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, image_root)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.2e} (base: {base_lr:.2e})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    # Check if validation dataset has enough samples for eval_loss computation
    min_val_samples = 8  # Minimum samples needed for reliable eval_loss
    val_dataset_size = len(val_dataset)
    has_sufficient_val_data = val_dataset_size >= min_val_samples

    print(
        f"🔍 DEBUG: Validation dataset size: {val_dataset_size}, min required: {min_val_samples}"
    )
    print(f"🔍 DEBUG: Has sufficient validation data: {has_sufficient_val_data}")

    if not has_sufficient_val_data:
        print(
            f"⚠️  Validation set too small ({val_dataset_size} samples), disabling validation"
        )
        # Completely disable evaluation when validation set is too small
        training_args = TrainingArguments(
            output_dir=artifacts_dir,
            per_device_train_batch_size=adaptive_batch,
            dataloader_num_workers=adaptive_workers,
            dataloader_pin_memory=True,
            num_train_epochs=2 if args.debug else 100,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch",
            eval_strategy="no",
            logging_strategy="epoch",
            learning_rate=scaled_lr,
            weight_decay=1e-4,
            load_best_model_at_end=False,  # No model loading when no validation
            # Do NOT set metric_for_best_model when eval_strategy is "no"
            greater_is_better=False,
            max_grad_norm=10.0,  # Add gradient clipping
            dataloader_drop_last=True,  # Drop incomplete batches
        )
        eval_dataset = None
    else:
        print(f"✅ Validation set has {len(val_dataset)} samples, enabling validation")
        # Normal evaluation when sufficient validation data
        # Set up proper metrics for early stopping callback
        training_args = TrainingArguments(
            output_dir=artifacts_dir,
            per_device_train_batch_size=adaptive_batch,
            dataloader_num_workers=adaptive_workers,
            dataloader_pin_memory=True,
            num_train_epochs=2 if args.debug else 100,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=scaled_lr,
            weight_decay=1e-4,
            load_best_model_at_end=True,  # Enable loading best model for early stopping
            metric_for_best_model="eval_loss",  # Use evaluation loss for early stopping
            greater_is_better=False,  # Lower loss is better
            max_grad_norm=10.0,  # Add gradient clipping
            dataloader_drop_last=True,  # Drop incomplete batches
        )
        eval_dataset = val_dataset

    data_collator = DataCollator(image_processor=image_processor)

    # Create trainer with conditional eval_dataset
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train_dataset,
    }

    # Only add eval_dataset if we have one
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
        # Add early stopping callback when we have validation data
        if args.debug:
            trainer_kwargs["callbacks"] = [
                EarlyStoppingCallback(early_stopping_patience=1)
            ]
        else:
            trainer_kwargs["callbacks"] = [
                EarlyStoppingCallback(early_stopping_patience=5)
            ]

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    return trainer


def predict(
    args, model, image_processor, label_to_cat_id, categories, output_dir, test_json
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(test_json, "r") as f:
        test_data = json.load(f)
    test_images = test_data["images"]
    if args.debug:
        test_data["images"] = test_data["images"][:5]

    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for i, image_info in enumerate(test_images):
        img_path = (args.data_dir or args.test_dir) / image_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        detection_threshold = 0.1 if args.debug else 0.5
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=detection_threshold
        )[0]

        pred_annotations = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            # The model predicts class indices, need to map back to original category_id
            original_cat_id = label_to_cat_id[label.item()]
            pred_annotations.append(
                {
                    "bbox": [
                        box[0].item(),
                        box[1].item(),
                        (box[2] - box[0]).item(),
                        (box[3] - box[1]).item(),
                    ],
                    "category_id": original_cat_id,
                }
            )

        output_xml_path = output_dir / f"{Path(image_info['file_name']).stem}.xml"
        create_new_pagexml_file(
            output_xml_path,
            image_info["file_name"],
            image_info["width"],
            image_info["height"],
            pred_annotations,
            "layout",
            categories_list=categories,
        )

        if args.debug and i >= 4:
            break

    print(f"✅ Predictions saved to {output_dir}")


def save_model(trainer, save_model_path):
    if save_model_path:
        try:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_model_path))
            print(f"   💾 Saved model to standardized path: {save_model_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy model to standardized path: {e}")


def load_model(model_identifier, categories, load_model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    id2label = {i: cat["name"] for i, cat in enumerate(categories)}
    label2id = {v: k for k, v in id2label.items()}
    num_classes = len(categories)

    if load_model_path and load_model_path.exists():
        model_to_load = load_model_path
        print(f"   Fine-tuning from sequential model: {model_to_load}")
    else:
        model_to_load = model_identifier
        print(f"   Loading base model: {model_to_load}")

    model = DetrForObjectDetection.from_pretrained(
        model_to_load,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    return model


def make_datasets(args, train_json, val_json, test_json, image_processor):
    with open(train_json, "r") as f:
        train_data = json.load(f)
    categories = train_data["categories"]
    cat_id_to_label = {cat["id"]: i for i, cat in enumerate(categories)}
    label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}

    train_transforms = None
    if args.augment:
        print("   💪 Augmentation enabled.")
        train_transforms = get_augment_policy()

    train_dataset = DetrCocoDataset(
        train_json,
        args.data_dir or args.train_dir,
        image_processor,
        cat_id_map=cat_id_to_label,
        debug=args.debug,
        transforms=train_transforms,
    )

    val_dataset = DetrCocoDataset(
        val_json,
        args.data_dir or args.train_dir,
        image_processor,
        cat_id_map=cat_id_to_label,
        debug=args.debug,
    )

    return categories, label_to_cat_id, train_dataset, val_dataset


def train_test_detr(
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
    artifacts_dir = Path(tempfile.mkdtemp(prefix="detr_")) / "detr_artifacts"
    image_processor = DetrImageProcessor.from_pretrained(model_identifier)

    # --- 1. Data and Model Preparation ---
    print("📦 Preparing data and model...")
    categories, label_to_cat_id, train_dataset, val_dataset = make_datasets(
        args, train_json, val_json, test_json, image_processor
    )

    model = load_model(model_identifier, categories, load_model_path)

    # --- 2. Training ---
    print(
        f"🚀 Training on {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}..."
    )
    trainer = train(
        args,
        model,
        train_dataset,
        val_dataset,
        image_processor,
        artifacts_dir=artifacts_dir,
        train_json=train_json,
    )

    save_model(trainer, save_model_path)

    # --- 3. Prediction ---
    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(
            args,
            model,
            image_processor,
            label_to_cat_id,
            categories,
            output_dir,
            test_json,
        )

    shutil.rmtree(artifacts_dir)
