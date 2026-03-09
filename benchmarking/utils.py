import json
import os
from pathlib import Path
from typing import Tuple, List, Optional

try:
    import torch
    import torchvision.transforms.v2 as T

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.absolute()

# Use experiment ID for isolation of results and trained models
# Falls back to "default" for backward compatibility
EXPERIMENT_ID = os.environ.get("LAUDARE_EXPERIMENT_ID", "default")
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_ID
MODELS_DIR = PROJECT_ROOT / "models" / EXPERIMENT_ID


def get_pretrained_model_path(
    framework: str, task: str, model_name: str, pretrain_dir: Optional[Path] = None
) -> Path:
    """Get the path where pretrained model should be stored.

    Note: Pretrained models are isolated per experiment to avoid conflicts
    during concurrent training, but synthetic pretrain data is shared.
    """
    pretrained_dir = MODELS_DIR / "pretrained"

    # Include directory basename in the filename if custom pretrain_dir is provided
    if pretrain_dir:
        dir_basename = pretrain_dir.name
        path = pretrained_dir / f"{framework}_{task}_{model_name}_{dir_basename}"
    else:
        path = pretrained_dir / f"{framework}_{task}_{model_name}"

    return path


def get_dataset_statistics(json_file: Path, image_root: Path) -> dict:
    """Get statistics about dataset for adaptive batch sizing."""
    with open(json_file, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        return {"avg_pixels": 0, "max_pixels": 0, "count": 0}

    # Sample a few images to estimate size
    sample_size = min(10, len(images))
    total_pixels = 0
    max_pixels = 0

    for img_info in images[:sample_size]:
        img_path = image_root / img_info["file_name"]
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    pixels = img.width * img.height
                    total_pixels += pixels
                    max_pixels = max(max_pixels, pixels)
            except:
                continue

    return {
        "avg_pixels": total_pixels / sample_size if sample_size > 0 else 0,
        "max_pixels": max_pixels,
        "count": len(images),
    }


def get_adaptive_batch_size(
    json_file: Path, image_root: Path, base_batch_size: int = 4
) -> int:
    """Calculate adaptive batch size based on image dimensions."""
    stats = get_dataset_statistics(json_file, image_root)
    avg_pixels = stats["avg_pixels"]

    # Reference: 4MP image = batch size 4
    reference_pixels = 4_000_000

    if avg_pixels == 0:
        return base_batch_size

    # Scale batch size inversely with image size
    adaptive_batch = max(1, int(base_batch_size * reference_pixels / avg_pixels))

    return adaptive_batch


def get_adaptive_num_workers(json_file: Path, image_root: Path) -> int:
    """Calculate adaptive number of workers based on dataset size."""
    stats = get_dataset_statistics(json_file, image_root)
    avg_pixels = stats["avg_pixels"]

    # Larger images = fewer workers to avoid memory issues
    if avg_pixels > 6_000_000:
        return 2
    elif avg_pixels > 4_000_000:
        return 4
    else:
        return 8


def path_json2pagexml(
    json_file: Path, task: str, edition: str, data_dir: Path
) -> List[Path]:
    """Given a COCO-style JSON file, return the corresponding PageXML files."""
    if not json_file.exists():
        return []

    with open(json_file, "r") as f:
        data = json.load(f)

    xml_files = []
    seen_files = set()

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in data.get("images", [])
    }

    original_pagexml_dir = (
        data_dir
        / f"annotations-{edition}"
        / "processed_splits"
        / f"pagexml_all_{edition}_{task}"
    )
    # We need to iterate through annotations to ensure we only include images that are actually used.
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id in image_id_to_filename:
            img_filename = image_id_to_filename[image_id]
            img_name = Path(img_filename).stem
            xml_file = original_pagexml_dir / f"{img_name}.xml"

            if xml_file.exists() and xml_file not in seen_files:
                xml_files.append(xml_file)
                seen_files.add(xml_file)
            elif not xml_file.exists():
                print(f"Warning: XML file does not exist: {xml_file}")

    return xml_files


def check_existing_pretrained_model(
    framework, task, model_name, pretrain_dir: Optional[Path] = None
):
    """Helper function to check for existing pretrained models - updated to use new system."""
    # Use new pretrained model system
    pretrained_model_path = get_pretrained_model_path(
        framework, task, model_name, pretrain_dir
    )
    if pretrained_model_path.exists():
        return pretrained_model_path
    else:
        return None


def get_json_paths(
    task, is_train_test_mode, train_dir, test_dir, is_sequential, fold_data_path, args
):
    """Get JSON paths based on mode."""
    if is_train_test_mode:
        if train_dir and test_dir:
            train_ann_dir = train_dir / f"annotations-{args.edition}"
            test_ann_dir = test_dir / f"annotations-{args.edition}"
            train_json = train_ann_dir / f"{task}_train.json"
            val_json = train_ann_dir / f"{task}_val.json"
            test_json = test_ann_dir / f"{task}_test.json"
            return train_json, val_json, test_json
        else:
            raise ValueError(
                "Train and test directories must be specified in train/test mode"
            )
    elif is_sequential:
        if fold_data_path:
            train_json = (
                fold_data_path
                / f"seq_{args.sequential_step:02d}"
                / f"{task}_train.json"
            )
            val_json = (
                fold_data_path / f"seq_{args.sequential_step:02d}" / f"{task}_val.json"
            )
            test_json = fold_data_path / "sequential_test" / f"{task}_test.json"
            return train_json, val_json, test_json
        else:
            raise ValueError("Fold data path must be specified in sequential mode")
    elif fold_data_path:
        train_json = fold_data_path / f"{task}_train.json"
        val_json = fold_data_path / f"{task}_val.json"
        test_json = fold_data_path / f"{task}_test.json"
        return train_json, val_json, test_json
    else:
        raise ValueError("Unable to determine data paths")


def get_augment_policy(type: str = "transform"):
    """
    Returns the TrivialAugmentWide transform policy for PIL images.
    This is suitable for tasks where the image is processed further
    by another library (e.g., Hugging Face processors) after augmentation.
    """
    if not TORCH_AVAILABLE:
        if type == "str":
            return "trivialaugment"
        raise ImportError("PyTorch not available for transform-based augmentation")

    if type == "transform":
        return T.TrivialAugmentWide()  # for usual torchvision pipelines
    elif type == "str":
        return "trivialaugment"  # for ultralytics (yolo)
    else:
        raise ValueError("type must be 'transform' or 'str'")


def get_transforms(augment=False):
    """
    Returns a pair of transforms for object detection models that require
    tensor conversion and handle bounding boxes.

    Args:
        augment (bool): Whether to include augmentation in the training transform.

    Returns:
        tuple: A pair of (train_transform, eval_transform).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available for transforms")

    if augment:
        train_transform = T.Compose(
            [
                T.TrivialAugmentWide(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
    else:
        train_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    eval_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    return train_transform, eval_transform


def model_name2identifier(
    framework: str, task: str, name: str, models_file: Path
) -> Tuple[str, str]:
    """Retrieves model name and value from the models JSON file based on an index."""
    with open(models_file, "r") as f:
        models = json.load(f)

    try:
        task_models = models[framework][task]
    except KeyError:
        raise ValueError(
            f"No models found for framework '{framework}' and task '{task}'"
        )

    return task_models[name]


def setup_with_args(args):
    """Modified setup that takes pre-parsed arguments and sets up all paths."""
    # Setup paths based on mode
    data_path = Path()
    if args.train_dir and args.test_dir:
        # Train/Test directory mode
        fold_name = f"{args.train_dir.name}_{args.test_dir.name}"
        current_results_dir = RESULTS_DIR / "train_test" / fold_name
    else:
        # Standard mode with processed splits
        data_root_processed = (
            args.data_dir / f"annotations-{args.edition}" / "processed_splits"
        )
        data_dir_name = args.data_dir.name

        # Determine paths based on mode (n-fold vs sequential)
        if args.sequential_step is not None:
            fold_name = f"seq_{args.sequential_step:02d}"
            data_path = data_root_processed / args.sequential_strategy
            current_results_dir = (
                RESULTS_DIR
                / data_dir_name
                / args.edition
                / "sequential"
                / args.sequential_strategy
                / f"{fold_name}"
            )
        else:
            fold_name = f"fold_{args.fold}"
            data_path = data_root_processed / f"train_test_{args.fold}"
            current_results_dir = RESULTS_DIR / data_dir_name / args.edition / fold_name

    output_dir = current_results_dir / f"{args.framework}_{args.task}_{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle pretrained models
    if args.enable_pretrain:
        existing_pretrained_model = check_existing_pretrained_model(
            args.framework,
            args.task,
            args.model_name,
            getattr(args, "pretrain_dir", None),
        )
        if existing_pretrained_model:
            args.pretrained_model = Path(existing_pretrained_model)
        else:
            args.pretrained_model = None
    else:
        args.pretrained_model = None

    # Get model information
    models_file = PROJECT_ROOT / "benchmarking" / "models.json"
    base_model = model_name2identifier(
        args.framework, args.task, args.model_name, models_file
    )

    # Determine which model to use: sequential path takes precedence over base model
    if base_model:
        model_identifier = base_model
        print(f"   Fine-tuning from base model: {base_model}")
    else:
        model_identifier = None
        print("   Training from scratch.")

    # Validation checks
    if args.train_dir is not None and args.test_dir is None:
        raise ValueError("Both train and test directories must be specified.")
    if args.train_dir is None and args.test_dir is not None:
        raise ValueError("Both train and test directories must be specified.")
    is_train_test_mode = args.train_dir is not None and args.test_dir is not None
    is_sequential = args.sequential_step is not None

    if is_sequential and is_train_test_mode:
        raise ValueError(
            "Cannot use both sequential mode and train/test mode simultaneously."
        )
    if not is_train_test_mode and data_path is None:
        raise ValueError(
            "Must specify either sequential mode, train/test mode, or single fold mode."
        )

    train_json, val_json, test_json = get_json_paths(
        args.task,
        is_train_test_mode,
        args.train_dir,
        args.test_dir,
        is_sequential,
        data_path,
        args,
    )
    save_model_path = output_dir / "best_model"

    load_model_path = args.pretrained_model
    if is_sequential and args.sequential_step > 0:
        prev_step = args.sequential_step - 1
        prev_model_path = (
            output_dir.parent.parent
            / f"seq_{prev_step:02d}"
            / f"{args.framework}_{args.task}_{args.model_name}"
            / "best_model"
        )
        if prev_model_path.exists():
            load_model_path = prev_model_path
        else:
            raise RuntimeError(
                f"❌ Previous step model not found for sequential learning: {prev_model_path}"
            )

    out = dict(
        args=args,
        is_train_test_mode=is_train_test_mode,
        is_sequential=is_sequential,
        output_dir=output_dir / "predictions",
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        save_model_path=save_model_path,
        load_model_path=load_model_path,
        model_identifier=model_identifier,
    )
    if args.debug:
        from pprint import pprint

        print("----------")
        print("Setup:")
        pprint(out)
        print("----------")
    return out
