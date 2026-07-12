import argparse
import importlib
import json
import os
from itertools import islice
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_FILE = PROJECT_ROOT / "benchmarking" / "models.json"
DEFAULT_OUTPUT_MODELS_FILE = PROJECT_ROOT / "benchmarking" / "models_catmus.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "CATMuS_medieval"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "catmus"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "catmus_pretrain"

SUPPORTED_FRAMEWORKS = ("kraken", "calamari", "trocr", "dtrocr", "paddleocr_vl", "vlt")
RECOGNITION_TASKS = ("ocr", "omr")
TASK_CATEGORY_IDS = {"ocr": 6, "omr": 5}
FRAMEWORK_SUFFIXES = {
    "kraken": ".mlmodel",
    "calamari": ".ckpt",
}
SPLIT_FILENAME_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain recognition frameworks on CATMuS/medieval and emit models_catmus.json."
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        required=True,
        help=f"Frameworks to pretrain. Supported: {', '.join(SUPPORTED_FRAMEWORKS)} or 'all'.",
    )
    parser.add_argument(
        "--dataset-id",
        default="CATMuS/medieval",
        help="Hugging Face dataset id to download.",
    )
    parser.add_argument(
        "--catmus-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Where the CATMuS dataset should be materialized in Laudare train/test format.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Where pretrained checkpoints should be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Where per-run artifacts should be stored.",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=DEFAULT_MODELS_FILE,
        help="Source models.json to read base model identifiers from.",
    )
    parser.add_argument(
        "--output-models-file",
        type=Path,
        default=DEFAULT_OUTPUT_MODELS_FILE,
        help="Output models_catmus.json path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Optional device selector, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable framework augmentation flags during CATMuS pretraining.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run each framework in debug mode.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        help="Optional cap per CATMuS split while materializing the dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Rebuild the local CATMuS dataset even if it already exists.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain checkpoints even if the target checkpoint path already exists.",
    )
    return parser.parse_args()


def set_device(device: str | None) -> None:
    if not device:
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        return
    raise ValueError("--device must be 'cpu' or of the form 'cuda:N'")


def normalize_frameworks(values: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for raw in values:
        for framework in raw.split(","):
            framework = framework.strip()
            if framework:
                selected.append(framework)

    if not selected:
        raise ValueError("No frameworks provided")

    if "all" in selected:
        invalid = [fw for fw in selected if fw != "all"]
        if invalid:
            raise ValueError("Use either 'all' or an explicit framework list, not both")
        return list(SUPPORTED_FRAMEWORKS)

    invalid = [fw for fw in selected if fw not in SUPPORTED_FRAMEWORKS]
    if invalid:
        raise ValueError(
            f"Unsupported CATMuS frameworks: {', '.join(sorted(set(invalid)))}. Supported: {', '.join(SUPPORTED_FRAMEWORKS)}"
        )

    deduped: list[str] = []
    seen = set()
    for framework in selected:
        if framework not in seen:
            deduped.append(framework)
            seen.add(framework)
    return deduped


def load_models(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def checkpoint_path_for(framework: str, task: str, model_name: str, checkpoint_root: Path) -> Path:
    checkpoint_path = checkpoint_root / framework / task / model_name
    suffix = FRAMEWORK_SUFFIXES.get(framework)
    if suffix and checkpoint_path.suffix != suffix:
        checkpoint_path = checkpoint_path.with_suffix(suffix)
    return checkpoint_path


def materialize_catmus_dataset(
    dataset_id: str,
    dataset_root: Path,
    max_samples_per_split: int | None,
    force: bool,
) -> None:
    metadata_path = dataset_root / ".catmus_materialized.json"
    if metadata_path.exists() and not force:
        print(f"📦 Reusing CATMuS dataset at {dataset_root}")
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with `uv add datasets` or run with `uv run --with datasets ...`."
        ) from e

    from benchmarking.annotations.ann_handler import create_pagexml_from_annotations

    print(f"📥 Streaming {dataset_id} from Hugging Face...")
    hf_dataset = load_dataset(dataset_id, streaming=True)

    dataset_root.mkdir(parents=True, exist_ok=True)
    annotations_dir = dataset_root / "annotations-diplomatic"
    processed_dir = annotations_dir / "processed_splits"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for hf_split, file_split in SPLIT_FILENAME_MAP.items():
        split_data = hf_dataset[hf_split]
        if max_samples_per_split is not None:
            split_iter = islice(split_data, max_samples_per_split)
        else:
            split_iter = split_data

        task_payloads = {
            task: {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": TASK_CATEGORY_IDS[task],
                        "name": "line" if task == "ocr" else "music_line",
                        "supercategory": "recognition",
                    }
                ],
            }
            for task in RECOGNITION_TASKS
        }

        kept_rows = 0
        for row_index, row in enumerate(split_iter):
            text = (row.get("text") or "").strip()
            if not text:
                continue

            image = row["im"]
            filename = f"catmus_{hf_split}_{row_index:06d}.png"
            image_path = dataset_root / filename
            image.save(image_path)
            width, height = image.size

            image_record = {
                "id": kept_rows,
                "file_name": filename,
                "width": int(width),
                "height": int(height),
            }
            ann_common = {
                "id": kept_rows,
                "image_id": kept_rows,
                "bbox": [0, 0, int(width), int(height)],
                "area": int(width) * int(height),
                "iscrowd": 0,
                "description": text,
            }

            for task in RECOGNITION_TASKS:
                task_payloads[task]["images"].append(dict(image_record))
                task_payloads[task]["annotations"].append(
                    {
                        **ann_common,
                        "category_id": TASK_CATEGORY_IDS[task],
                    }
                )
            kept_rows += 1

        split_counts[hf_split] = kept_rows
        print(f"   {hf_split}: {kept_rows} samples")

        for task in RECOGNITION_TASKS:
            json_path = annotations_dir / f"{task}_{file_split}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(task_payloads[task], f, ensure_ascii=False, indent=2)

            create_pagexml_from_annotations(
                json_path,
                processed_dir / f"pagexml_all_diplomatic_{task}",
                task,
                image_base_path=dataset_root,
            )

    metadata = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "max_samples_per_split": max_samples_per_split,
        "split_counts": split_counts,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ CATMuS dataset materialized at {dataset_root}")


def build_setup_dict(
    framework: str,
    task: str,
    model_name: str,
    model_identifier: str,
    dataset_root: Path,
    checkpoint_path: Path,
    results_dir: Path,
    augment: bool,
    debug: bool,
) -> dict:
    args = argparse.Namespace(
        edition="diplomatic",
        debug=debug,
        task=task,
        framework=framework,
        enable_pretrain=False,
        fold=0,
        model_name=model_name,
        sequential_step=None,
        sequential_strategy="cumulative",
        augment=augment,
        train_dir=dataset_root,
        test_dir=dataset_root,
        data_dir=None,
        pretrain_dir=None,
    )

    run_dir = results_dir / framework / task / model_name
    output_dir = run_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "args": args,
        "is_train_test_mode": True,
        "is_sequential": False,
        "output_dir": output_dir,
        "train_json": dataset_root / "annotations-diplomatic" / f"{task}_train.json",
        "val_json": dataset_root / "annotations-diplomatic" / f"{task}_val.json",
        "test_json": None,
        "save_model_path": checkpoint_path,
        "load_model_path": None,
        "model_identifier": model_identifier,
    }


def run_framework(framework: str, setup_dict: dict) -> None:
    module = importlib.import_module(f"benchmarking.train_test.train_test_{framework}")
    train_fn = getattr(module, f"train_test_{framework}")
    train_fn(**setup_dict)


def replace_framework_models(models: dict, replacements: dict) -> dict:
    updated = json.loads(json.dumps(models))
    for framework, task_map in replacements.items():
        if framework not in updated:
            continue
        for task, model_map in task_map.items():
            if task not in updated[framework]:
                continue
            for model_name, checkpoint in model_map.items():
                updated[framework][task][model_name] = checkpoint
    return updated


def main() -> None:
    args = parse_args()
    set_device(args.device)

    frameworks = normalize_frameworks(args.frameworks)
    models = load_models(args.models_file)

    missing = [fw for fw in frameworks if fw not in models]
    if missing:
        raise ValueError(f"Frameworks missing from {args.models_file}: {', '.join(missing)}")

    materialize_catmus_dataset(
        dataset_id=args.dataset_id,
        dataset_root=args.catmus_dir,
        max_samples_per_split=args.max_samples_per_split,
        force=args.force_download,
    )

    replacements: dict[str, dict[str, dict[str, str]]] = {}
    manifest: dict[str, dict[str, dict[str, str]]] = {}

    for framework in frameworks:
        recognition_entries = {
            task: models[framework][task]
            for task in RECOGNITION_TASKS
            if task in models[framework]
        }
        if not recognition_entries:
            raise ValueError(
                f"Framework '{framework}' has no OCR/OMR entries in {args.models_file}; CATMuS pretraining is only defined for recognition frameworks"
            )

        replacements.setdefault(framework, {})
        manifest.setdefault(framework, {})

        for task, model_map in recognition_entries.items():
            replacements[framework].setdefault(task, {})
            manifest[framework].setdefault(task, {})

            for model_name, model_identifier in model_map.items():
                checkpoint_path = checkpoint_path_for(
                    framework, task, model_name, args.checkpoint_dir
                )
                checkpoint_ref = repo_relative(checkpoint_path)
                manifest[framework][task][model_name] = checkpoint_ref

                if checkpoint_path.exists() and not args.force_train:
                    print(
                        f"⏭️  Skipping {framework}/{task}/{model_name}; checkpoint exists at {checkpoint_path}"
                    )
                    replacements[framework][task][model_name] = checkpoint_ref
                    continue

                print(
                    f"🚀 Pretraining {framework}/{task}/{model_name} from {model_identifier} on CATMuS"
                )
                setup_dict = build_setup_dict(
                    framework=framework,
                    task=task,
                    model_name=model_name,
                    model_identifier=model_identifier,
                    dataset_root=args.catmus_dir,
                    checkpoint_path=checkpoint_path,
                    results_dir=args.results_dir,
                    augment=args.augment,
                    debug=args.debug,
                )
                run_framework(framework, setup_dict)

                if not checkpoint_path.exists():
                    raise RuntimeError(
                        f"Training finished but checkpoint was not found at {checkpoint_path}"
                    )

                replacements[framework][task][model_name] = checkpoint_ref
                print(f"   ✅ Saved checkpoint: {checkpoint_ref}")

    updated_models = replace_framework_models(models, replacements)
    args.output_models_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_models_file, "w", encoding="utf-8") as f:
        json.dump(updated_models, f, ensure_ascii=False, indent=2)

    manifest_path = args.checkpoint_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {args.output_models_file}")
    print(f"✅ Wrote {manifest_path}")


if __name__ == "__main__":
    main()
