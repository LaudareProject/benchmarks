"""
Training and prediction script for DocLayout-YOLO for layout detection.
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .train_test_yolo import make_datasets, predict, save_model, train


def _resolve_model_identifier(model_identifier: str) -> str:
    if "::" not in model_identifier:
        return model_identifier

    repo_id, filename = model_identifier.split("::", 1)
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"   📥 Downloaded weights to: {local_path}")
    return local_path


def train_test_doclayout_yolo(
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
    artifacts_dir = Path(tempfile.mkdtemp(prefix="doclayout_yolo_artifacts_"))
    yolo_data_dir = artifacts_dir / "yolo_data"

    if yolo_data_dir.exists():
        shutil.rmtree(yolo_data_dir)

    print("📦 Preparing data in YOLO format...")
    categories, id_cat_map = make_datasets(
        args, train_json, val_json, test_json, yolo_data_dir
    )

    if load_model_path:
        model_source = Path(load_model_path)
        if model_source.suffix != ".pt":
            model_source = model_source.with_suffix(".pt")
    else:
        model_source = _resolve_model_identifier(model_identifier)

    print(f"   Loading weights from: {model_source}")
    model = YOLO(model_source)

    print("🚀 Training DocLayout-YOLO model...")
    image_root = args.data_dir or args.train_dir
    best_model_path = train(
        args, yolo_data_dir, model, artifacts_dir, train_json, image_root
    )

    if best_model_path is None:
        shutil.rmtree(artifacts_dir)
        return

    save_model(best_model_path, save_model_path)

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
