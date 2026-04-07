"""
Training and prediction script for Faster R-CNN for layout detection.
"""

import tempfile
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..annotations.ann_handler import create_new_pagexml_file
from ..utils import get_transforms, get_adaptive_batch_size, get_adaptive_num_workers


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # In debug mode, only run a few batches
        if getattr(data_loader.dataset, "debug", False) and i >= 4:
            break


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.train()  # Set model to training mode to get loss
    total_loss = 0
    count = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        count += 1
        # In debug mode, only run a few batches
        if getattr(data_loader.dataset, "debug", False) and i >= 4:
            break
    return total_loss / count if count > 0 else 0


class CocoLayoutDataset(Dataset):
    """COCO-style dataset for layout detection."""

    def __init__(
        self,
        json_file,
        data_dir,
        cat_id_map,
        debug=False,
        data_to_use=None,
        transforms=None,
    ):
        self.json_file = Path(json_file) if json_file else None
        self.data_dir = Path(data_dir)
        self.cat_id_map = cat_id_map
        self.debug = debug
        self.transforms = transforms

        if data_to_use:
            data = data_to_use
        else:
            with open(self.json_file, "r") as f:
                data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.image_annotations = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            if ann["category_id"] in self.cat_id_map:
                self.image_annotations[ann["image_id"]].append(ann)

        # Filter out images with no valid annotations
        self.images = [img for img in self.images if self.image_annotations[img["id"]]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path = self.data_dir / image_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        annotations = self.image_annotations[image_info["id"]]
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_map[ann["category_id"]])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def get_model(model_size, num_classes):
    """Get a Faster R-CNN model with a new head."""
    if model_size == "resnet50":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif model_size == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported model size for FastRCNNPredictor: {model_size}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def predict(
    args,
    best_model_path,
    id_cat_map,
    categories,
    device,
    model,
    output_dir,
    test_dataset,
):
    print("✍️  Generating predictions...")
    if not args.debug and best_model_path.exists():
        print(f"   -> Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

    output_dir.mkdir(exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, _ = test_dataset[i]
            image_info = test_dataset.images[i]
            prediction = model([img.to(device)])[0]

            # Convert predictions to COCO annotation format
            pred_annotations = []
            detection_threshold = 0.1 if args.debug else 0.5
            for j, box in enumerate(prediction["boxes"]):
                if prediction["scores"][j] > detection_threshold:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    label_id = prediction["labels"][j].item()
                    pred_annotations.append(
                        {
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "category_id": id_cat_map[label_id],
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
            if args.debug and i == 4:
                break

    print(f"✅ Predictions saved to {output_dir}")


def train(
    args,
    best_model_path,
    device,
    model,
    train_loader,
    val_loader,
    scaled_lr,
):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=scaled_lr, momentum=0.9, weight_decay=0.0005)

    num_epochs = 1 if args.debug else 100
    patience = 15
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(
        f"🚀 Training for up to {num_epochs} epochs on {device} with patience {patience}..."
    )

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        print(f"--- Epoch {epoch + 1} summary: Validation Loss: {val_loss:.4f} ---")

        if not args.debug:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"   -> New best validation loss. Model saved to {best_model_path}"
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(
                    f"   -> Early stopping triggered after {patience} epochs with no improvement."
                )
                break


def make_datasets(args, train_json, val_json, test_json):
    # Load categories from training JSON
    with open(train_json, "r") as f:
        train_data = json.load(f)
    # Map category IDs to a consecutive range starting from 1 (0 is background)
    cat_id_map = {cat["id"]: i + 1 for i, cat in enumerate(train_data["categories"])}
    id_cat_map = {v: k for k, v in cat_id_map.items()}  # inverted mapping
    num_classes = len(train_data["categories"]) + 1

    # Get transformations
    train_transforms, eval_transforms = get_transforms(augment=args.augment)

    # Calculate adaptive parameters with LR scaling
    train_json_path = Path(train_json)
    image_root = args.data_dir or args.train_dir

    base_batch_size = 2  # Faster R-CNN base batch size
    base_lr = 0.001  # Faster R-CNN base learning rate (SGD)

    adaptive_batch = get_adaptive_batch_size(
        train_json_path, image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json_path, image_root)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.4f} (base: {base_lr:.4f})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    val_dataset = CocoLayoutDataset(
        val_json,
        args.data_dir or args.train_dir,
        cat_id_map,
        debug=args.debug,
        transforms=eval_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=adaptive_batch,
        collate_fn=collate_fn,
        num_workers=adaptive_workers,
        pin_memory=True,
    )

    train_dataset = CocoLayoutDataset(
        train_json,
        args.data_dir or args.train_dir,
        cat_id_map,
        debug=args.debug,
        transforms=train_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=adaptive_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=adaptive_workers,
        pin_memory=True,
        persistent_workers=True if adaptive_workers > 0 else False,
    )

    if test_json is not None:
        test_dataset = CocoLayoutDataset(
            test_json,
            args.data_dir or args.test_dir,
            cat_id_map,
            debug=args.debug,
            transforms=eval_transforms,
        )
    else:
        test_dataset = None

    return (
        train_data["categories"],
        id_cat_map,
        num_classes,
        train_loader,
        val_loader,
        test_dataset,
        scaled_lr,
    )


def save_model(best_model_path, model, save_model_path):
    # Copy model to standardized path
    if save_model_path:
        try:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            if best_model_path.exists():
                shutil.copy2(best_model_path, save_model_path)
            else:
                torch.save(model.state_dict(), save_model_path)
            print(f"   💾 Saved model to standardized path: {save_model_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy model to standardized path: {e}")


def load_model(device, load_model_path, num_classes, model_identifier):
    model = get_model(model_identifier, num_classes)
    if load_model_path and load_model_path.exists():
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    model.to(device)
    return model


def train_test_faster_rcnn(
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
    # Create a consistent output directory name
    artifacts_dir = Path(tempfile.mkdtemp(prefix="faster_rcnn_"))

    best_model_path = artifacts_dir / "best_model.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (
        categories,
        id_cat_map,
        num_classes,
        train_loader,
        val_loader,
        test_dataset,
        scaled_lr,
    ) = make_datasets(args, train_json, val_json, test_json)

    model = load_model(device, load_model_path, num_classes, model_identifier)

    train(
        args,
        best_model_path,
        device,
        model,
        train_loader,
        val_loader,
        scaled_lr,
    )

    save_model(best_model_path, model, save_model_path)

    # --- Prediction Phase ---
    if test_dataset is not None:
        predict(
            args,
            best_model_path,
            id_cat_map,
            categories,
            device,
            model,
            output_dir,
            test_dataset,
        )

    shutil.rmtree(artifacts_dir)
