import json
import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from ..utils import get_adaptive_batch_size, get_adaptive_num_workers

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALAMARI_TRAIN_EXEC = PROJECT_ROOT / ".venv-calamari" / "bin" / "calamari-train"
CALAMARI_PREDICT_EXEC = PROJECT_ROOT / ".venv-calamari" / "bin" / "calamari-predict"


def _create_calamari_gt_files(
    annotations: List, image_id_to_path: Dict, data_dir: Path
) -> tuple[List[Path], set[str]]:
    """
    Helper function to crop line images and create .gt.txt files.
    """
    line_image_files = []
    failed_image_names = set()
    for ann in annotations:
        text = ann.get("description") or ann.get("text") or ""
        if not text:
            continue

        image_id = ann["image_id"]
        original_image_path = image_id_to_path.get(image_id)

        if not original_image_path or not original_image_path.exists():
            continue

        full_image = cv2.imread(str(original_image_path))
        if full_image is None:
            continue

        x, y, w, h = [int(c) for c in ann["bbox"]]
        line_image = full_image[y : y + h, x : x + w]

        line_base_name = f"{original_image_path.stem}_line_{ann['id']}"
        line_image_path = data_dir / f"{line_base_name}.png"
        # scale horizontally, otherwise calamari has issues
        # heuristic: 50 px per char, 300 px per word minimum, with height 350
        n_words = len(text.split())
        target_width = max(int(len(text) * 50 / 350 * h), n_words * 300 / 350 * h)
        target_width = max(target_width, line_image.shape[1])
        if line_image.size == 0 or h <= 0 or round(target_width) <= 0:
            failed_image_names.add(original_image_path.stem)
            continue

        try:
            line_image = cv2.resize(line_image, (round(target_width), h))
        except cv2.error:
            failed_image_names.add(original_image_path.stem)
            continue
        cv2.imwrite(str(line_image_path), line_image)
        line_image_files.append(line_image_path)

        gt_path = data_dir / f"{line_base_name}.gt.txt"
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(text)
    return line_image_files, failed_image_names


def _load_json_data(json_file: Path, image_root: Path) -> tuple[List, Dict]:
    with open(json_file, "r") as f:
        data = json.load(f)

    image_id_to_path = {
        img["id"]: image_root / img["file_name"] for img in data.get("images", [])
    }
    return data.get("annotations", []), image_id_to_path


def prepare_calamari_data(
    json_file: Path, data_dir: Path, image_root: Path, debug: bool = False
) -> tuple[List[Path], set[str]]:
    """
    Converts COCO-style JSON annotations into a flat directory of line images and
    corresponding .gt.txt files that Calamari can consume.
    """
    if not json_file.exists():
        return [], set()

    data_dir.mkdir(parents=True, exist_ok=True)
    annotations, image_id_to_path = _load_json_data(json_file, image_root)
    if debug:
        annotations = annotations[:40]
        print(f"   🐛 Using {len(annotations)} annotations (debug mode)")
    return _create_calamari_gt_files(annotations, image_id_to_path, data_dir)


def run_calamari_command(command: list, log_file: Path):
    """Runs a Calamari shell command and logs its output."""
    try:
        with open(log_file, "a") as lf:
            lf.write(f"Running command: {' '.join(str(c) for c in command)}\n")
            result = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            lf.write(f"Stdout: {result.stdout}\n")
            if result.stderr:
                lf.write(f"Stderr: {result.stderr}\n")
    except Exception as e:
        print(f"   ❌ Error running command: {e}")


def _aggregate_line_predictions_to_image_level(output_dir: Path):
    """
    Aggregate line-level predictions back to image-level for evaluation.

    Calamari produces predictions like:
    - c063_line_1.pred.txt
    - c063_line_2.pred.txt

    We need to combine them into:
    - c063.pred.txt
    """
    pred_files = list(output_dir.glob("*_line_*.pred.txt"))

    if not pred_files:
        print("   ⚠️  No line prediction files found to aggregate")
        return

    # Group predictions by original image name
    image_predictions = {}

    for pred_file in pred_files:
        # Extract original image name from line prediction filename
        # Format: {image_stem}_line_{line_id}.pred.txt
        filename = pred_file.stem  # removes .pred.txt
        if filename.endswith(".pred"):
            filename = filename[:-5]  # remove .pred suffix

        # Find the last occurrence of "_line_" to split correctly
        line_marker = "_line_"
        line_idx = filename.rfind(line_marker)

        if line_idx == -1:
            continue  # Skip files that don't match expected pattern

        original_image_name = filename[:line_idx]
        line_id = filename[line_idx + len(line_marker) :]

        # Read the prediction text
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                pred_text = f.read().strip()
        except Exception as e:
            print(f"   ⚠️  Error reading {pred_file}: {e}")
            continue

        # Group by original image
        if original_image_name not in image_predictions:
            image_predictions[original_image_name] = []

        image_predictions[original_image_name].append((int(line_id), pred_text))

    # Write aggregated predictions
    for image_name, line_preds in image_predictions.items():
        # Sort by line ID to maintain order
        line_preds.sort(key=lambda x: x[0])

        # Combine all line texts with spaces
        combined_text = " ".join(pred[1] for pred in line_preds if pred[1])

        # Write to image-level prediction file
        output_file = output_dir / f"{image_name}.pred.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(combined_text)
        except Exception as e:
            print(f"   ⚠️  Error writing {output_file}: {e}")

    # remove line-level prediction files
    for line_pred_file in output_dir.glob("*_line_*.pred.txt"):
        line_pred_file.unlink()

    print(f"   ✅ Aggregated predictions for {len(image_predictions)} images")


def _train_eval_calamari_impl(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    artifacts_dir: Path,
    output_dir: Path,
    task: str,
    log_file: Path,
    train_image_root: Path,
    test_image_root: Path,
    debug: bool = False,
    cuda: bool = True,
    pretrained_model: Optional[Path | str] = None,
    epochs: int = 100,
    learning_rate: Optional[float] = None,
    early_stopping_patience: int = 15,
    calamari_preload: bool = True,
    augment: bool = True,
    save_model_path: Optional[Path] = None,
):
    """Trains and evaluates a Calamari model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate adaptive parameters with LR scaling
    base_batch_size = 4  # Calamari default
    base_lr = 0.001  # Calamari default (1e-3)

    adaptive_batch_size = get_adaptive_batch_size(
        train_json, train_image_root, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, train_image_root)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch_size / base_batch_size)

    # Override learning_rate parameter if not explicitly set
    if learning_rate is None:
        learning_rate = scaled_lr

    print(f"   📊 Adaptive batch size: {adaptive_batch_size}")
    print(f"   📈 Scaled learning rate: {learning_rate:.2e} (base: {base_lr:.2e})")
    print(f"   👷 Adaptive workers: {adaptive_workers}")

    # Calamari works with image files and gt.txt files in the same directory.
    print("   📦 Preparing data for Calamari...")

    train_dir = artifacts_dir / "data" / "train"
    val_dir = artifacts_dir / "data" / "val"
    test_dir = artifacts_dir / "data" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    prepare_calamari_data(val_json, val_dir, train_image_root, debug)
    prepare_calamari_data(train_json, train_dir, train_image_root, debug)
    failed_test_images = set()
    if test_json:
        _, failed_test_images = prepare_calamari_data(
            test_json, test_dir, test_image_root, debug
        )

    print("   ⚙️  Training Calamari model...")
    model_output_dir = artifacts_dir / "model"

    # Pass wildcard patterns as literal args to avoid huge argv lists.
    train_images_pattern = str(train_dir / "*.png")
    val_images_pattern = str(val_dir / "*.png")

    train_cmd = [
        str(CALAMARI_TRAIN_EXEC),
        "--train.images",
        train_images_pattern,
        "--trainer.output_dir",
        str(model_output_dir),
        "--early_stopping.n_to_go",
        str(early_stopping_patience),
        "--train.preload",
        str(calamari_preload),
        "--train.num_processes",
        str(adaptive_workers),
    ]

    if any(val_dir.glob("*.png")):
        train_cmd.extend(
            [
                "--val.images",
                val_images_pattern,
                "--val.preload",
                str(calamari_preload),
                "--val.num_processes",
                str(adaptive_workers),
            ]
        )
    else:
        train_cmd.extend(["--trainer.gen", "TrainOnly"])

    if augment:
        train_cmd.extend(
            [
                "--n_augmentations",
                "1",
                "--trainer.data_aug_retrain_on_original",
                "false",
            ]
        )
        print("   💪 Augmentation enabled, without retraining on original data.")

    epochs_to_run = 2 if debug else epochs
    train_cmd.extend(["--trainer.epochs", str(epochs_to_run)])

    if learning_rate is not None:
        train_cmd.extend(["--learning_rate.lr", str(learning_rate)])

    if cuda:
        train_cmd.extend(
            [
                "--device.gpus",
                "0",
                "--train.batch_size",
                str(adaptive_batch_size),
                "--train.batch_size",
                str(adaptive_batch_size),
            ]
        )

    if pretrained_model:
        train_cmd.extend(["--warmstart.model", pretrained_model])

    run_calamari_command(train_cmd, log_file)

    # The best model is saved in the 'best.ckpt' subdirectory
    best_model_path = model_output_dir / "best.ckpt"
    if not best_model_path.exists():
        print("   ❌ Calamari training failed, no best model found.")
    else:
        print(f"   ✅ Found trained model: {best_model_path}")

    # Copy model to standardized sequential learning path if specified
    if save_model_path:
        old_save_model_path = None
        if save_model_path.suffix != ".ckpt":
            old_save_model_path = save_model_path
            save_model_path = save_model_path.with_suffix(".ckpt")
        try:
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(best_model_path, save_model_path, dirs_exist_ok=True)
            # saving the json file, otherwise calamari complains
            shutil.copy(
                best_model_path.with_suffix(".ckpt.json"),
                save_model_path.with_suffix(".ckpt.json"),
            )
            if old_save_model_path:
                # also save a copy without .ckpt suffix for making the framework work
                shutil.copytree(
                    best_model_path,
                    old_save_model_path,
                    dirs_exist_ok=True,
                )
            print(f"   💾 Saved model to standardized path: {save_model_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy model to standardized path: {e}")

    if test_json:
        print("   🔮 Running predictions...")
        test_images_pattern = str(test_dir / "*.png")
        predict_cmd = [
            str(CALAMARI_PREDICT_EXEC),
            "--checkpoint",
            str(best_model_path),
            "--data.images",
            test_images_pattern,
            "--output_dir",
            str(output_dir),
            "--extended_prediction_data_format",
            "pred",
        ]
        run_calamari_command(predict_cmd, log_file)

        # since images were split into lines (e.g. c063.png -> c063_line_1.png, c063_line_2.png),
        # we now need to aggregate the predictions back to the original image level
        # so that the framework evaluation can work as expected
        print("   🔗 Aggregating line predictions to image level...")
        _aggregate_line_predictions_to_image_level(output_dir)

        # failed test crops/resizes are counted as recognition errors
        for image_name in failed_test_images:
            failed_output_file = output_dir / f"{image_name}.pred.txt"
            with open(failed_output_file, "w", encoding="utf-8") as f:
                f.write("")


def train_test_calamari(
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
    """Entry point for Calamari training with expanded parameter signature."""

    # Create log file
    log_file = (
        PROJECT_ROOT
        / "logs"
        / f"calamari_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cuda_available = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))

    artifacts_dir = Path(tempfile.mkdtemp(prefix="calamari_artifacts_"))

    if load_model_path is None:
        # we use path as identifier
        load_model_path = model_identifier

    if Path(load_model_path).suffix != ".ckpt":
        load_model_path = Path(load_model_path).with_suffix(".ckpt")

    # Call the existing train_eval_calamari function
    _train_eval_calamari_impl(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        task=args.task,
        log_file=log_file,
        train_image_root=args.data_dir or args.train_dir,
        test_image_root=args.data_dir or args.test_dir,
        debug=args.debug,
        cuda=cuda_available,
        pretrained_model=load_model_path,
        augment=True,
        save_model_path=Path(save_model_path),
    )

    shutil.rmtree(artifacts_dir)
