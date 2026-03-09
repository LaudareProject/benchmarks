"""
Training and prediction script for YOLO for layout detection.
"""

import tempfile
import json
import os
import shutil
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO

from ..annotations.ann_handler import create_new_pagexml_file
from ..utils import (
    get_augment_policy,
    get_adaptive_batch_size,
    get_adaptive_num_workers,
)
from ..ultralytics_monkey_patch import apply_ultralytics_monkey_patch

apply_ultralytics_monkey_patch()


def prepare_yolo_data(data_json, data_dir, yolo_data_dir, split, cat_id_map, debug):
    """Prepares a split of data in YOLO format."""
    (yolo_data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (yolo_data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    with open(data_json, "r") as f:
        data = json.load(f)

    if debug:
        images = data["images"][:5]
    else:
        images = data["images"]
    image_info = {img["id"]: img for img in images}

    label_lines_written = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_info:
            continue

        img = image_info[img_id]
        img_w, img_h = img["width"], img["height"]
        file_name = img["file_name"]

        # Create symlink to image
        src_img_path = data_dir / file_name
        dst_img_path = yolo_data_dir / "images" / split / Path(file_name).name
        if not dst_img_path.exists():
            os.symlink(src_img_path.absolute(), dst_img_path.absolute())

        # Create label file
        if cat_id_map is not None:
            label_file = (
                yolo_data_dir / "labels" / split / f"{Path(file_name).stem}.txt"
            )
            with open(label_file, "a") as f:
                cat_id = cat_id_map[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                f.write(f"{cat_id} {x_center} {y_center} {norm_w} {norm_h}\n")
                label_lines_written.setdefault(label_file, 0)
                label_lines_written[label_file] += 1


def train(args, yolo_data_dir, model, artifacts_dir, train_json, image_root):
    epochs = 3 if args.debug else 100
    patience = 15

    # Calculate adaptive parameters with LR scaling
    base_batch_size = 16  # YOLO default
    base_lr = 0.01  # YOLO default for object detection

    adaptive_batch = get_adaptive_batch_size(
        train_json, image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, image_root)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.4f} (base: {base_lr:.4f})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    auto_augment = None
    if args.augment:
        print("   💪 Augmentation enabled (TrivialAugmentWide).")
        auto_augment = get_augment_policy("str")

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_yaml_path = yolo_data_dir / "data.yaml"
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        patience=patience,
        project=str(artifacts_dir),
        device=device_str,
        exist_ok=True,
        name="train",
        augment=args.augment,
        auto_augment=auto_augment,
        batch=adaptive_batch,
        workers=adaptive_workers,
        lr0=scaled_lr,  # Initial learning rate (scaled)
    )

    # Find the best model path
    best_model_path = artifacts_dir / "train" / "weights" / "best.pt"
    if not best_model_path.exists():
        last_model_path = artifacts_dir / "train" / "weights" / "last.pt"
        if not last_model_path.exists():
            print("❌ No trained model found!")
            return None
        best_model_path = last_model_path
    return best_model_path


def predict(
    args, best_model_path, yolo_data_dir, id_cat_map, categories, output_dir, test_json
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(best_model_path)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    test_images_dir = yolo_data_dir / "images" / "test"
    prepare_yolo_data(
        test_json,
        args.data_dir or args.test_dir,
        yolo_data_dir,
        "test",
        None,
        args.debug,
    )

    detection_threshold = 0.1 if args.debug else 0.25
    results = model.predict(
        source=str(test_images_dir), device=device_str, conf=detection_threshold
    )

    # Map image file names back to their original info from COCO json
    test_images = json.load(open(test_json, "r"))["images"]
    test_file_map = {Path(img["file_name"]).name: img for img in test_images}

    for res in results:
        img_path = Path(res.path)
        image_info = test_file_map.get(img_path.name)
        if not image_info:
            continue

        pred_annotations = []
        if res.boxes:
            for box in res.boxes:
                # Bbox in xyxy format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_idx = int(box.cls[0].item())
                original_cat_id = id_cat_map[class_idx]
                pred_annotations.append(
                    {
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
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

    print(f"✅ Predictions saved to {output_dir}")


def save_model(best_model_path, save_model_path):
    if save_model_path:
        old_save_model_path = None
        if save_model_path.suffix != ".pt":
            old_save_model_path = save_model_path
            save_model_path = save_model_path.with_suffix(".pt")
        try:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model_path, save_model_path)
            if old_save_model_path:
                shutil.copy2(best_model_path, old_save_model_path)
            print(f"   💾 Saved model to standardized path: {save_model_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy model to standardized path: {e}")


def make_datasets(args, train_json, val_json, test_json, yolo_data_dir):
    with open(train_json, "r") as f:
        train_data = json.load(f)
    categories = train_data["categories"]
    cat_id_map = {cat["id"]: i for i, cat in enumerate(categories)}
    id_cat_map = {i: cat["id"] for i, cat in enumerate(categories)}
    cat_names = [cat["name"] for cat in categories]

    prepare_yolo_data(
        train_json,
        args.data_dir or args.train_dir,
        yolo_data_dir,
        "train",
        cat_id_map,
        args.debug,
    )
    prepare_yolo_data(
        val_json,
        args.data_dir or args.train_dir,
        yolo_data_dir,
        "val",
        cat_id_map,
        args.debug,
    )

    data_yaml = {
        "path": str(yolo_data_dir.resolve()),
        "train": "images/train",
        "val": "images/val" if val_json is not None else "images/train",
        "names": {i: name for i, name in enumerate(cat_names)},
    }
    data_yaml_path = yolo_data_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"   ✅ YOLO data ready in {yolo_data_dir}")

    return categories, id_cat_map


def train_test_yolo(
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
    # Create temporary directory for YOLO data
    artifacts_dir = Path(tempfile.mkdtemp(prefix="yolo_artifacts_"))
    yolo_data_dir = artifacts_dir / "yolo_data"

    if yolo_data_dir.exists():
        shutil.rmtree(yolo_data_dir)

    # --- 1. Prepare data ---
    print("📦 Preparing data in YOLO format...")
    categories, id_cat_map = make_datasets(
        args, train_json, val_json, test_json, yolo_data_dir
    )

    if load_model_path:
        if Path(load_model_path).suffix != ".pt":
            load_model_path = Path(load_model_path).with_suffix(".pt")
    model = YOLO(load_model_path if load_model_path else model_identifier)
    # --- 2. Training ---
    print("🚀 Training YOLO model...")
    image_root = args.data_dir or args.train_dir
    best_model_path = train(
        args, yolo_data_dir, model, artifacts_dir, train_json, image_root
    )

    if best_model_path is None:
        return

    save_model(best_model_path, save_model_path)

    # --- 3. Prediction ---
    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(
            args,
            best_model_path,
            yolo_data_dir,
            id_cat_map,
            categories,
            output_dir,
            test_json,
        )

    shutil.rmtree(artifacts_dir)
