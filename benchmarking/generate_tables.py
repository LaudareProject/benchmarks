#!/usr/bin/env python3
"""
Generate publication-ready CSV tables from experiment results.

Tables use LaTeX syntax for special characters and collapsed format:
- OCR/OMR: rows grouped by task, columns grouped by edition
- Layout: rows by model, columns grouped by edition
- Layout metrics scaled to 0-100
"""

import argparse
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

try:
    from benchmarking.duplicate_guard import check_for_duplicate_relative_json_paths
except ImportError:
    from duplicate_guard import check_for_duplicate_relative_json_paths


# Metrics to scale from 0-1 to 0-100
SCALE_TO_100 = {"mAP", "mAP@0.5", "f1@0.50", "F1@0.50"}

# Metric definitions per task type (display name, JSON key)
TASK_METRICS = {
    "ocr": [
        ("CER", "CER"),
        ("WER", "WER"),
        ("p-CER", "p-CER"),
        ("WWER", "WWER"),
        ("R-WER-2", "R-WER-2"),
        ("R-WER-4", "R-WER-4"),
        ("R-WER-8", "R-WER-8"),
    ],
    "omr": [
        ("CER", "CER"),
        ("WER", "NER"),
        ("p-CER", "p-CER"),
        ("WWER", "WWER"),
        ("R-WER-2", "R-WER-2"),
        ("R-WER-4", "R-WER-4"),
        ("R-WER-8", "R-WER-8"),
    ],
    "layout": [("mAP", "mAP"), ("mAP@0.5", "mAP@0.5"), ("F1@0.50", "f1@0.50")],
}

# Framework display names for publication
FRAMEWORK_DISPLAY_NAMES = {
    # OCR/OMR frameworks
    "calamari_ocr_default": "Calamari",
    "calamari_omr_default": "Calamari",
    "kraken_ocr_default": "Kraken",
    "kraken_omr_default": "Kraken",
    "trocr_ocr_small": "TrOCR-Small",
    "trocr_ocr_large": "TrOCR-Large",
    "trocr_omr_small": "TrOCR-Small",
    "trocr_omr_large": "TrOCR-Large",
    # Layout frameworks
    "yolo_layout_yolov8n": "YOLOv8n",
    "yolo_layout_yolov8s": "YOLOv8s",
    "faster_rcnn_layout_mobilenet": "Faster R-CNN (MobileNet)",
    "faster_rcnn_layout_resnet50": "Faster R-CNN (ResNet-50)",
    "detr_layout_default": "DETR",
}


def get_display_name(framework: str) -> str:
    """Get publication-ready display name for a framework."""
    if framework in FRAMEWORK_DISPLAY_NAMES:
        return FRAMEWORK_DISPLAY_NAMES[framework]
    # Handle pretrained variants
    if framework.endswith("_pretrained"):
        base = framework[: -len("_pretrained")]
        base_name = FRAMEWORK_DISPLAY_NAMES.get(base)
        if base_name:
            return f"{base_name} (pretrained)"
    # Fallback: clean up the name
    name = framework.replace("_default", "").replace("_", " ").title()
    return name


def get_experiment_pattern(experiment_name: str) -> str:
    """Get the shared pattern for experiment folders."""
    parts = experiment_name.split("_")
    if len(parts) >= 4 and all(part.isdigit() for part in parts[-3:]):
        return "_".join(parts[:-3])
    return experiment_name


def get_initial_token(name: str) -> str:
    """Get initial token before first underscore."""
    return name.split("_", 1)[0]


def classify_pretrained_origin(
    experiment_dir: Path, eval_file: Path, dataset: str
) -> Optional[str]:
    """Classify pretrained file origin as cross-manuscript or synthetic."""
    if "pretrained" not in eval_file.stem:
        return None

    rel_parts = eval_file.relative_to(experiment_dir).parts
    if rel_parts and rel_parts[0] == "train_test":
        return "cross"

    source_token = get_initial_token(experiment_dir.name)
    dataset_token = get_initial_token(dataset)
    return "synthetic" if dataset_token == source_token else "cross"


def get_model_name_from_framework(framework: str) -> str:
    """Extract model name without task suffix for grouping."""
    # Remove task part to get base model name for display
    display = get_display_name(framework)
    return display


def load_evaluation_json(filepath: Path) -> Optional[Dict]:
    """Load evaluation JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "metrics" in data:
            return data["metrics"]
        return data
    except Exception as e:
        print(f"  Warning: Failed to load {filepath}: {e}")
        return None


def detect_task_from_framework(framework_name: str) -> Optional[str]:
    """Detect task type from framework directory name."""
    for task in ["ocr", "omr", "layout"]:
        if f"_{task}_" in framework_name or framework_name.endswith(f"_{task}"):
            return task
    return None


def get_evaluation_file(framework_dir: Path) -> Optional[Path]:
    """Find the evaluation JSON file in a framework directory."""
    patterns = [
        "*_evaluation.json",
        "ocr_evaluation.json",
        "omr_evaluation.json",
        "layout_evaluation.json",
    ]
    for pattern in patterns:
        files = list(framework_dir.glob(pattern))
        if files:
            return files[0]
    return None


def compute_ci_95(values: List[float]) -> Tuple[float, float]:
    """Compute mean and 95% CI half-width."""
    if len(values) < 2:
        return values[0] if values else 0.0, 0.0
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf(0.975, len(values) - 1)
    error = t_crit * se
    return mean, error


def format_mean_ci(mean: float, error: float, scale: bool = False) -> str:
    """Format mean $\pm$ error with LaTeX syntax."""
    if scale:
        mean *= 100
        error *= 100
    return f"${mean:.1f} \\pm {error:.1f}$"


def format_value(value: float, scale: bool = False) -> str:
    """Format single value."""
    if scale:
        value *= 100
    return f"${value:.1f}$"


def collect_5fold_results(experiment_dir: Path) -> Dict:
    """
    Collect 5-fold cross-validation results.
    Returns: {dataset: {edition: {framework: {metric: [values_per_fold]}}}}
    """
    results = {}

    for dataset_dir in experiment_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in [
            "tables",
            "train_test",
            "plots",
        ]:
            continue

        dataset = dataset_dir.name
        results[dataset] = {}

        for edition_dir in dataset_dir.iterdir():
            if not edition_dir.is_dir():
                continue
            edition = edition_dir.name
            if edition not in ["diplomatic", "editorial"]:
                continue

            results[dataset][edition] = {}
            fold_dirs = sorted(edition_dir.glob("fold_*"))

            for fold_dir in fold_dirs:
                if not fold_dir.is_dir():
                    continue

                for framework_dir in fold_dir.iterdir():
                    if not framework_dir.is_dir():
                        continue

                    framework = framework_dir.name
                    if framework not in results[dataset][edition]:
                        results[dataset][edition][framework] = {}

                    eval_file = get_evaluation_file(framework_dir)
                    if eval_file is None:
                        continue

                    metrics = load_evaluation_json(eval_file)
                    if metrics is None:
                        continue

                    for metric_name, value in metrics.items():
                        if not isinstance(value, (int, float)):
                            continue
                        if metric_name not in results[dataset][edition][framework]:
                            results[dataset][edition][framework][metric_name] = []
                        results[dataset][edition][framework][metric_name].append(value)

    return results


def merge_5fold_results(target: Dict, source: Dict) -> None:
    """Merge 5-fold results into target in-place."""
    for dataset, editions in source.items():
        if dataset not in target:
            target[dataset] = {}
        for edition, frameworks in editions.items():
            if edition not in target[dataset]:
                target[dataset][edition] = {}
            for framework, metrics in frameworks.items():
                if framework not in target[dataset][edition]:
                    target[dataset][edition][framework] = {}
                for metric_name, values in metrics.items():
                    target[dataset][edition][framework].setdefault(
                        metric_name, []
                    ).extend(values)


def collect_train_test_results(experiment_dir: Path) -> Dict:
    """
    Collect train-test results.
    Returns: {train_test_pair: {framework: {metric: value}}}
    """
    results = {}
    train_test_dir = experiment_dir / "train_test"

    if not train_test_dir.exists():
        return results

    for pair_dir in train_test_dir.iterdir():
        if not pair_dir.is_dir():
            continue

        pair_name = pair_dir.name
        results[pair_name] = {}

        for framework_dir in pair_dir.iterdir():
            if not framework_dir.is_dir():
                continue

            framework = framework_dir.name
            results[pair_name][framework] = {}

            eval_file = get_evaluation_file(framework_dir)
            if eval_file is None:
                continue

            metrics = load_evaluation_json(eval_file)
            if metrics is None:
                continue

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    results[pair_name][framework][metric_name] = value

    return results


def merge_train_test_results(target: Dict, source: Dict) -> None:
    """Merge train-test results into target in-place."""
    for pair_name, frameworks in source.items():
        if pair_name not in target:
            target[pair_name] = {}
        for framework, metrics in frameworks.items():
            if framework not in target[pair_name]:
                target[pair_name][framework] = {}
            target[pair_name][framework].update(metrics)


def generate_collapsed_ocr_omr_table(results: Dict, output_dir: Path):
    """
    Generate collapsed OCR/OMR table with:
    - Rows: Task (OCR/OMR) + Model
    - Columns: Edition (Diplomatic/Editorial) x Metrics (CER, WER/NER)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, editions in results.items():
        # Collect all models for OCR and OMR
        ocr_models = {}  # {model_name: {edition: {metric: (mean, error)}}}
        omr_models = {}

        for edition in ["diplomatic", "editorial"]:
            if edition not in editions:
                continue

            for framework, metrics in editions[edition].items():
                task = detect_task_from_framework(framework)
                if task not in ["ocr", "omr"]:
                    continue

                model_name = get_model_name_from_framework(framework)
                target = ocr_models if task == "ocr" else omr_models

                if model_name not in target:
                    target[model_name] = {}
                if edition not in target[model_name]:
                    target[model_name][edition] = {}

                # Get metrics for this task
                task_metric_defs = TASK_METRICS[task]
                for display_name, json_key in task_metric_defs:
                    values = metrics.get(json_key, [])
                    if values:
                        mean, error = compute_ci_95(values)
                        target[model_name][edition][display_name] = (mean, error)

        if not ocr_models and not omr_models:
            continue

        # Build CSV with 2-row header
        # Row 1: Edition grouping
        # Row 2: Metric names
        filepath = output_dir / f"5fold_ocr_omr_{dataset}.csv"

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row 1: Edition grouping
            header1 = [
                "",
                "",
                "Diplomatic",
                "",
                "",
                "",
                "",
                "",
                "",
                "Editorial",
                "",
                "",
                "",
                "",
                "",
            ]
            writer.writerow(header1)

            # Header row 2: Metric names
            header2 = [
                "Task",
                "Model",
                "CER",
                "WER",
                "p-CER",
                "WWER",
                "R-WER-2",
                "R-WER-4",
                "R-WER-8",
                "CER",
                "WER",
                "p-CER",
                "WWER",
                "R-WER-2",
                "R-WER-4",
                "R-WER-8",
            ]
            writer.writerow(header2)

            # Write OCR rows
            for model_name in sorted(ocr_models.keys()):
                row = ["OCR", model_name]
                for edition in ["diplomatic", "editorial"]:
                    for metric in [
                        "CER",
                        "WER",
                        "p-CER",
                        "WWER",
                        "R-WER-2",
                        "R-WER-4",
                        "R-WER-8",
                    ]:
                        if (
                            edition in ocr_models[model_name]
                            and metric in ocr_models[model_name][edition]
                        ):
                            mean, error = ocr_models[model_name][edition][metric]
                            row.append(format_mean_ci(mean, error, scale=False))
                        else:
                            row.append("-")
                writer.writerow(row)

            # Write OMR rows (NER maps to WER column position)
            for model_name in sorted(omr_models.keys()):
                row = ["OMR", model_name]
                for edition in ["diplomatic", "editorial"]:
                    for metric in [
                        "CER",
                        "WER",
                        "p-CER",
                        "WWER",
                        "R-WER-2",
                        "R-WER-4",
                        "R-WER-8",
                    ]:
                        if (
                            edition in omr_models[model_name]
                            and metric in omr_models[model_name][edition]
                        ):
                            mean, error = omr_models[model_name][edition][metric]
                            row.append(format_mean_ci(mean, error, scale=False))
                        else:
                            row.append("-")
                writer.writerow(row)

        print(f"  Created: {filepath}")


def generate_collapsed_layout_table(results: Dict, output_dir: Path):
    """
    Generate collapsed Layout table with:
    - Rows: Model
    - Columns: Edition (Diplomatic/Editorial) x Metrics (mAP, F1@0.50)
    Layout metrics scaled to 0-100.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, editions in results.items():
        # Collect all layout models
        layout_models = {}  # {model_name: {edition: {metric: (mean, error)}}}

        for edition in ["diplomatic", "editorial"]:
            if edition not in editions:
                continue

            for framework, metrics in editions[edition].items():
                task = detect_task_from_framework(framework)
                if task != "layout":
                    continue

                model_name = get_model_name_from_framework(framework)

                if model_name not in layout_models:
                    layout_models[model_name] = {}
                if edition not in layout_models[model_name]:
                    layout_models[model_name][edition] = {}

                # Get metrics
                for display_name, json_key in TASK_METRICS["layout"]:
                    values = metrics.get(json_key, [])
                    if values:
                        mean, error = compute_ci_95(values)
                        layout_models[model_name][edition][display_name] = (mean, error)

        if not layout_models:
            continue

        # Build CSV with 2-row header
        filepath = output_dir / f"5fold_layout_{dataset}.csv"

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header row 1: Edition grouping
            header1 = ["", "Diplomatic", "", "", "Editorial", "", ""]
            writer.writerow(header1)

            # Header row 2: Metric names
            header2 = [
                "Model",
                "mAP",
                "mAP@0.5",
                "F1@0.50",
                "mAP",
                "mAP@0.5",
                "F1@0.50",
            ]
            writer.writerow(header2)

            for model_name in sorted(layout_models.keys()):
                row = [model_name]
                for edition in ["diplomatic", "editorial"]:
                    for metric in ["mAP", "mAP@0.5", "F1@0.50"]:
                        if (
                            edition in layout_models[model_name]
                            and metric in layout_models[model_name][edition]
                        ):
                            mean, error = layout_models[model_name][edition][metric]
                            row.append(format_mean_ci(mean, error, scale=True))
                        else:
                            row.append("-")
                writer.writerow(row)

        print(f"  Created: {filepath}")


def generate_train_test_tables(results: Dict, output_dir: Path):
    """Generate train-test tables (one per task, collapsed by edition if available)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, frameworks in results.items():
        if not frameworks:
            continue

        # Group by task
        task_data = {"ocr": {}, "omr": {}, "layout": {}}

        for framework, metrics in frameworks.items():
            task = detect_task_from_framework(framework)
            if task is None:
                continue
            model_name = get_model_name_from_framework(framework)
            task_data[task][model_name] = metrics

        # Generate OCR/OMR combined table
        if task_data["ocr"] or task_data["omr"]:
            filepath = output_dir / f"train_test_ocr_omr_{pair_name}.csv"

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Task",
                        "Model",
                        "CER",
                        "WER/NER",
                        "p-CER",
                        "WWER",
                        "R-WER-2",
                        "R-WER-4",
                        "R-WER-8",
                    ]
                )

                for model_name in sorted(task_data["ocr"].keys()):
                    metrics = task_data["ocr"][model_name]
                    row = ["OCR", model_name]
                    for metric in [
                        "CER",
                        "WER",
                        "p-CER",
                        "WWER",
                        "R-WER-2",
                        "R-WER-4",
                        "R-WER-8",
                    ]:
                        if metric in metrics:
                            row.append(format_value(metrics[metric]))
                        else:
                            row.append("-")
                    writer.writerow(row)

                for model_name in sorted(task_data["omr"].keys()):
                    metrics = task_data["omr"][model_name]
                    row = ["OMR", model_name]
                    for metric in [
                        "CER",
                        "NER",
                        "p-CER",
                        "WWER",
                        "R-WER-2",
                        "R-WER-4",
                        "R-WER-8",
                    ]:
                        if metric in metrics:
                            row.append(format_value(metrics[metric]))
                        else:
                            row.append("-")
                    writer.writerow(row)

            print(f"  Created: {filepath}")

        # Generate Layout table
        if task_data["layout"]:
            filepath = output_dir / f"train_test_layout_{pair_name}.csv"

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Model", "mAP", "mAP@0.5", "F1@0.50"])

                for model_name in sorted(task_data["layout"].keys()):
                    metrics = task_data["layout"][model_name]
                    row = [model_name]
                    for metric in ["mAP", "mAP@0.5", "f1@0.50"]:
                        if metric in metrics:
                            row.append(format_value(metrics[metric], scale=True))
                        else:
                            row.append("-")
                    writer.writerow(row)

            print(f"  Created: {filepath}")


def collect_sequential_data(experiment_dir: Path) -> dict:
    """
    Collect all sequential evaluation data from one experiment directory.

    Returns nested dict: {dataset: {edition: {task: {framework_key: {seq_num: {metric: value}}}}}}
    """
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for seq_path in experiment_dir.rglob("sequential/random_sample/seq_*"):
        if not seq_path.is_dir():
            continue

        match = re.match(r"seq_(\d+)", seq_path.name)
        if not match:
            continue
        seq_num = int(match.group(1))

        parts = seq_path.parts
        try:
            sequential_idx = parts.index("sequential")
            edition = parts[sequential_idx - 1]
            dataset = parts[sequential_idx - 2]
        except (ValueError, IndexError):
            continue

        for framework_dir in seq_path.iterdir():
            if not framework_dir.is_dir():
                continue

            # Normalize framework folder name to a display key
            folder = framework_dir.name
            # Try direct lookup first, then fall back to cleaning
            if folder in FRAMEWORK_DISPLAY_NAMES:
                framework_key = folder
            else:
                # Strip trailing _default and try again
                stripped = re.sub(r"_default$", "", folder)
                framework_key = (
                    stripped if stripped in FRAMEWORK_DISPLAY_NAMES else folder
                )

            for eval_file in framework_dir.glob("*_evaluation.json"):
                filename = eval_file.stem
                is_pretrained = "pretrained" in filename
                task_match = re.match(r"(\w+?)_(?:pretrained_)?evaluation", filename)
                if not task_match:
                    continue
                task = task_match.group(1)

                with open(eval_file, "r") as f:
                    eval_data = json.load(f)
                metrics = eval_data.get("metrics", {})

                key = f"{framework_key}_pretrained" if is_pretrained else framework_key
                data[dataset][edition][task][key][seq_num] = metrics

    return data


def collect_filtered_sequential_data(
    experiment_dir: Path, mode: str, with_source_pair_key: bool = False
) -> dict:
    """
    Collect filtered sequential data.

    mode='in': in-manuscript, same initial token and non-pretrained files.
    mode='cross': pretrained files classified as cross-manuscript.
    mode='synthetic': pretrained files classified as synthetic pre-training.
    """
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    source_token = get_initial_token(experiment_dir.name)
    source_dataset = get_experiment_pattern(experiment_dir.name)

    for seq_path in experiment_dir.rglob("sequential/random_sample/seq_*"):
        if not seq_path.is_dir():
            continue

        match = re.match(r"seq_(\d+)", seq_path.name)
        if not match:
            continue
        seq_num = int(match.group(1))

        parts = seq_path.parts
        try:
            sequential_idx = parts.index("sequential")
            edition = parts[sequential_idx - 1]
            dataset = parts[sequential_idx - 2]
        except (ValueError, IndexError):
            continue

        dataset_token = get_initial_token(dataset)
        is_same_token = dataset_token == source_token
        if mode == "in" and not is_same_token:
            continue
        if mode == "cross" and is_same_token:
            continue

        for framework_dir in seq_path.iterdir():
            if not framework_dir.is_dir():
                continue

            folder = framework_dir.name
            if folder in FRAMEWORK_DISPLAY_NAMES:
                framework_key = folder
            else:
                stripped = re.sub(r"_default$", "", folder)
                framework_key = (
                    stripped if stripped in FRAMEWORK_DISPLAY_NAMES else folder
                )

            for eval_file in framework_dir.glob("*_evaluation.json"):
                filename = eval_file.stem
                is_pretrained = "pretrained" in filename

                if mode == "in" and is_pretrained:
                    continue

                if mode in {"cross", "synthetic"} and not is_pretrained:
                    continue

                pretrained_origin = None
                if is_pretrained:
                    pretrained_origin = classify_pretrained_origin(
                        experiment_dir, eval_file, dataset
                    )
                    if mode == "cross" and pretrained_origin != "cross":
                        continue
                    if mode == "synthetic" and pretrained_origin != "synthetic":
                        continue

                if is_pretrained:
                    task_match = re.match(r"(\w+?)_pretrained_evaluation", filename)
                else:
                    task_match = re.match(r"(\w+?)_evaluation", filename)
                if not task_match:
                    continue
                task = task_match.group(1)

                with open(eval_file, "r") as f:
                    eval_data = json.load(f)
                metrics = eval_data.get("metrics", {})

                key = f"{framework_key}_pretrained" if is_pretrained else framework_key
                dataset_key = dataset
                if mode == "cross" and with_source_pair_key:
                    dataset_key = f"{dataset}__FROM__{source_dataset}"
                data[dataset_key][edition][task][key][seq_num] = metrics

    return data


def merge_sequential_data(target: dict, source: dict) -> None:
    """Merge sequential data from source into target in-place."""
    for dataset, editions in source.items():
        for edition, tasks in editions.items():
            for task, frameworks in tasks.items():
                for framework, seq_data in frameworks.items():
                    for seq_num, metrics in seq_data.items():
                        target[dataset][edition][task].setdefault(framework, {})[
                            seq_num
                        ] = metrics


def compute_auc_over_10(
    framework_data: dict, metric: str, scale: bool
) -> Optional[float]:
    """Compute AUC/10 for one framework's sequential data."""
    seq_nums = sorted(framework_data.keys())
    values = []
    for seq_num in seq_nums:
        v = framework_data[seq_num].get(metric)
        if v is None:
            return None
        values.append(v * 100 if scale else v)
    if len(values) < 2:
        return None
    return float(np.trapezoid(values, seq_nums) / 10.0)


def compute_metric_at_10th(
    framework_data: dict, metric: str, scale: bool
) -> Optional[float]:
    """Get metric value at the 10th sequential point."""
    seq_nums = sorted(framework_data.keys())
    if len(seq_nums) < 10:
        return None
    seq_num = seq_nums[9]
    value = framework_data[seq_num].get(metric)
    if value is None:
        return None
    return float(value * 100 if scale else value)


def generate_sequential_ocr_omr_auc_table(
    data: dict,
    output_dir: Path,
    file_prefix: str,
    pretrained_only: Optional[bool],
    filename_suffix: str = "",
) -> None:
    """
    Write sequential AUC table for OCR/OMR.
    Columns: Task, Model, Diplomatic AUC/10, Diplomatic @10th, Editorial AUC/10, Editorial @10th
    """
    for dataset, editions in data.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"{file_prefix}_ocr_omr_{dataset}{filename_suffix}.csv"
        )

        # Gather all models per task
        task_models: Dict[str, set] = {"OCR": set(), "OMR": set()}
        for edition in ["diplomatic", "editorial"]:
            for task_key, task_upper in [("ocr", "OCR"), ("omr", "OMR")]:
                for fw in editions.get(edition, {}).get(task_key, {}).keys():
                    if pretrained_only is True and not fw.endswith("_pretrained"):
                        continue
                    if pretrained_only is False and fw.endswith("_pretrained"):
                        continue
                    task_models[task_upper].add(fw)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["", "", "Diplomatic", "", "Editorial", ""])
            writer.writerow(["Task", "Model", "AUC/10", "@10th", "AUC/10", "@10th"])

            for task_upper, task_key, metric in [
                ("OCR", "ocr", "CER"),
                ("OMR", "omr", "NER"),
            ]:
                for fw in sorted(task_models[task_upper]):
                    dipl = compute_auc_over_10(
                        editions.get("diplomatic", {}).get(task_key, {}).get(fw, {}),
                        metric,
                        False,
                    )
                    edit = compute_auc_over_10(
                        editions.get("editorial", {}).get(task_key, {}).get(fw, {}),
                        metric,
                        False,
                    )
                    dipl_10 = compute_metric_at_10th(
                        editions.get("diplomatic", {}).get(task_key, {}).get(fw, {}),
                        metric,
                        False,
                    )
                    edit_10 = compute_metric_at_10th(
                        editions.get("editorial", {}).get(task_key, {}).get(fw, {}),
                        metric,
                        False,
                    )
                    dipl_str = f"${dipl:.2f}$" if dipl is not None else "-"
                    edit_str = f"${edit:.2f}$" if edit is not None else "-"
                    dipl_10_str = f"${dipl_10:.2f}$" if dipl_10 is not None else "-"
                    edit_10_str = f"${edit_10:.2f}$" if edit_10 is not None else "-"
                    writer.writerow(
                        [
                            task_upper,
                            get_display_name(fw),
                            dipl_str,
                            dipl_10_str,
                            edit_str,
                            edit_10_str,
                        ]
                    )

        print(f"Saved: {output_path}")


def generate_sequential_layout_auc_table(
    data: dict,
    output_dir: Path,
    file_prefix: str,
    pretrained_only: Optional[bool],
    filename_suffix: str = "",
) -> None:
    """
    Write sequential AUC table for layout.
    Columns include AUC/10 and mAP@10th for each dataset/edition.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(data.keys())
    datasets_part = "_".join(datasets)
    output_path = (
        output_dir / f"{file_prefix}_layout_{datasets_part}{filename_suffix}.csv"
    )
    editions = ["diplomatic", "editorial"]

    # Collect all models across both datasets
    all_models: set = set()
    for dataset in datasets:
        for edition in editions:
            for fw in data.get(dataset, {}).get(edition, {}).get("layout", {}).keys():
                if pretrained_only is True and not fw.endswith("_pretrained"):
                    continue
                if pretrained_only is False and fw.endswith("_pretrained"):
                    continue
                all_models.add(fw)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header1 = [""]
        for dataset in datasets:
            header1.extend([dataset.replace("_", " "), "", "", ""])
        writer.writerow(header1)

        header2 = ["Model"]
        for _dataset in datasets:
            header2.extend(
                [
                    "Diplomatic AUC/10",
                    "Diplomatic mAP@10th",
                    "Editorial AUC/10",
                    "Editorial mAP@10th",
                ]
            )
        writer.writerow(header2)

        for fw in sorted(all_models):
            row = [get_display_name(fw)]
            for dataset in datasets:
                for edition in editions:
                    auc = compute_auc_over_10(
                        data.get(dataset, {})
                        .get(edition, {})
                        .get("layout", {})
                        .get(fw, {}),
                        "mAP",
                        True,  # scale mAP to 0-100
                    )
                    at_10 = compute_metric_at_10th(
                        data.get(dataset, {})
                        .get(edition, {})
                        .get("layout", {})
                        .get(fw, {}),
                        "mAP",
                        True,
                    )
                    row.append(f"${auc:.2f}$" if auc is not None else "-")
                    row.append(f"${at_10:.2f}$" if at_10 is not None else "-")
            writer.writerow(row)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready CSV tables from experiment results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory (default: results/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print(f"Scanning results in: {results_dir}")

    experiment_dirs = [
        p
        for p in results_dir.iterdir()
        if p.is_dir() and p.name not in {"plots", "tables"}
    ]
    pattern_groups = {}
    for exp_dir in experiment_dirs:
        pattern = get_experiment_pattern(exp_dir.name)
        pattern_groups.setdefault(pattern, []).append(exp_dir)

    if check_for_duplicate_relative_json_paths(experiment_dirs):
        return 1

    # Process each pattern group
    for pattern, exp_dirs in pattern_groups.items():
        print(f"\nProcessing pattern: {pattern}")

        output_dir = results_dir / "tables" / pattern

        fold_results = {}
        train_test_results = {}

        for exp_dir in exp_dirs:
            print(f"  Merging experiment: {exp_dir.name}")

            # Collect 5-fold results
            fold_data = collect_5fold_results(exp_dir)
            merge_5fold_results(fold_results, fold_data)

            # Collect train-test results
            train_test_data = collect_train_test_results(exp_dir)
            merge_train_test_results(train_test_results, train_test_data)

        if fold_results:
            print("  Generating collapsed OCR/OMR table...")
            generate_collapsed_ocr_omr_table(fold_results, output_dir)
            print("  Generating collapsed Layout table...")
            generate_collapsed_layout_table(fold_results, output_dir)

        if train_test_results:
            print("  Generating train-test tables...")
            generate_train_test_tables(train_test_results, output_dir)

    # Sequential in-manuscript AUC tables
    print("\nCollecting in-manuscript sequential data for AUC tables...")
    all_sequential_in = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for exp_dirs in pattern_groups.values():
        for exp_dir in exp_dirs:
            merge_sequential_data(
                all_sequential_in, collect_filtered_sequential_data(exp_dir, mode="in")
            )

    if all_sequential_in:
        seq_output_dir = results_dir / "tables"

        for dataset in sorted(all_sequential_in.keys()):
            dataset_output_dir = seq_output_dir / dataset
            generate_sequential_ocr_omr_auc_table(
                {dataset: all_sequential_in[dataset]},
                dataset_output_dir,
                file_prefix="sequential_in_manuscript_auc",
                pretrained_only=False,
            )

        generate_sequential_layout_auc_table(
            all_sequential_in,
            seq_output_dir,
            file_prefix="sequential_in_manuscript_auc",
            pretrained_only=False,
        )

    # Sequential cross-manuscript AUC tables
    print("\nCollecting cross-manuscript sequential data for AUC tables...")
    all_sequential_cross = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for exp_dirs in pattern_groups.values():
        for exp_dir in exp_dirs:
            merge_sequential_data(
                all_sequential_cross,
                collect_filtered_sequential_data(
                    exp_dir, mode="cross", with_source_pair_key=True
                ),
            )

    if all_sequential_cross:
        seq_output_dir = results_dir / "tables"

        for pair_key in sorted(all_sequential_cross.keys()):
            target_dataset, source_dataset = pair_key.split("__FROM__", 1)
            dataset_output_dir = seq_output_dir / target_dataset
            generate_sequential_ocr_omr_auc_table(
                {target_dataset: all_sequential_cross[pair_key]},
                dataset_output_dir,
                file_prefix="sequential_cross_manuscript_auc",
                pretrained_only=True,
                filename_suffix=f"_{source_dataset}",
            )

            generate_sequential_layout_auc_table(
                {target_dataset: all_sequential_cross[pair_key]},
                dataset_output_dir,
                file_prefix="sequential_cross_manuscript_auc",
                pretrained_only=True,
                filename_suffix=f"_{source_dataset}",
            )

    # Sequential synthetic pre-training AUC tables
    print("\nCollecting synthetic pre-training sequential data for AUC tables...")
    all_sequential_synthetic = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for exp_dirs in pattern_groups.values():
        for exp_dir in exp_dirs:
            merge_sequential_data(
                all_sequential_synthetic,
                collect_filtered_sequential_data(exp_dir, mode="synthetic"),
            )

    if all_sequential_synthetic:
        seq_output_dir = results_dir / "tables"

        for dataset in sorted(all_sequential_synthetic.keys()):
            dataset_output_dir = seq_output_dir / dataset
            generate_sequential_ocr_omr_auc_table(
                {dataset: all_sequential_synthetic[dataset]},
                dataset_output_dir,
                file_prefix="sequential_synthetic_pretraining_auc",
                pretrained_only=True,
            )

        generate_sequential_layout_auc_table(
            all_sequential_synthetic,
            seq_output_dir,
            file_prefix="sequential_synthetic_pretraining_auc",
            pretrained_only=True,
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
