"""
Generate publication-ready line plots for sequential training results.

Reads evaluation JSONs from sequential training experiments and creates
plots showing performance metrics vs number of training steps.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict

try:
    from benchmarking.duplicate_guard import check_for_duplicate_relative_json_paths
except ImportError:
    from duplicate_guard import check_for_duplicate_relative_json_paths

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use LaTeX-style rendering for publication quality
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# Colorblind-friendly IBM Design palette
COLORS = {
    "blue": "#648FFF",
    "purple": "#785EF0",
    "magenta": "#DC267F",
    "orange": "#FE6100",
    "yellow": "#FFB000",
}

# Publication-ready display names for frameworks
FRAMEWORK_DISPLAY_NAMES = {
    # HTR/HMR
    "trocr_small": "TrOCR-Small",
    "trocr_large": "TrOCR-Large",
    "calamari": "Calamari",
    "kraken": "Kraken",
    # Pretrained variants
    "trocr_small_pretrained": "TrOCR-Small (pretrained)",
    "trocr_large_pretrained": "TrOCR-Large (pretrained)",
    "calamari_pretrained": "Calamari (pretrained)",
    "kraken_pretrained": "Kraken (pretrained)",
    # Layout
    "yolo_yolov8n": "YOLOv8n",
    "yolo_yolov8s": "YOLOv8s",
    "faster_rcnn_mobilenet": "Faster R-CNN (MobileNet)",
    "faster_rcnn_resnet50": "Faster R-CNN (ResNet-50)",
    "faster_rcnn_layout_mobilenet": "Faster R-CNN (MobileNet)",
    "faster_rcnn_layout_resnet50": "Faster R-CNN (ResNet-50)",
    "detr": "DETR",
}

# Framework/model to color mapping
FRAMEWORK_COLORS = {
    "trocr_small": COLORS["blue"],
    "trocr_large": COLORS["purple"],
    "calamari": COLORS["magenta"],
    "kraken": COLORS["orange"],
    "yolo_yolov8n": COLORS["blue"],
    "yolo_yolov8s": COLORS["purple"],
    "faster_rcnn_mobilenet": COLORS["magenta"],
    "faster_rcnn_resnet50": COLORS["orange"],
    "faster_rcnn_layout_mobilenet": COLORS["magenta"],
    "faster_rcnn_layout_resnet50": COLORS["orange"],
    "detr": COLORS["yellow"],
}

# Metric display names with proper formatting
METRIC_DISPLAY = {
    "CER": "CER (%)",
    "WER": "WER (%)",
    "NER": "NER (%)",
    "mAP": "mAP (%)",
    "mAP@0.5": "mAP@0.5 (%)",
    "f1@0.50": "F1@0.50 (%)",
}

# Metrics that need scaling from 0-1 to 0-100
SCALE_TO_100 = {"mAP", "mAP@0.5", "f1@0.50"}

# Line styles for pretrained vs non-pretrained
LINE_STYLES = {
    "normal": "-",
    "pretrained": "--",
}

# Line styles for distinct cross-manuscript sources
CROSS_SOURCE_STYLES = ["-", "--", ":", "-."]

# Markers for different frameworks
MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]


def split_cross_manuscript_key(framework_key: str) -> tuple[str, str | None]:
    """Split a cross-manuscript framework key into base key and source dataset."""
    if "__FROM__" in framework_key:
        base_key, source_dataset = framework_key.split("__FROM__", 1)
        return base_key, source_dataset
    return framework_key, None


def get_display_name(framework_key: str) -> str:
    """Get publication-ready display name for a framework."""
    framework_key, _ = split_cross_manuscript_key(framework_key)
    if framework_key in FRAMEWORK_DISPLAY_NAMES:
        return FRAMEWORK_DISPLAY_NAMES[framework_key]
    # Fallback: clean up the name
    name = framework_key.replace("_pretrained", " (pretrained)")
    name = name.replace("_default", "").replace("_", " ").title()
    return name


def parse_framework_task_model(folder_name: str) -> tuple:
    """Parse folder name like 'trocr_ocr_small' into (framework, task, model)."""
    parts = folder_name.split("_")
    if len(parts) >= 3:
        framework = parts[0]
        task = parts[1]
        model = "_".join(parts[2:])
        return framework, task, model
    return None, None, None


def get_framework_key(folder_name: str) -> str:
    """Get a normalized key for the framework from folder name."""
    framework, task, model = parse_framework_task_model(folder_name)
    if framework is None:
        return folder_name

    # Create keys like 'trocr_small', 'yolo_yolov8n', 'faster_rcnn_mobilenet'
    if framework in ["trocr", "calamari", "kraken"]:
        return f"{framework}_{model}" if model != "default" else framework
    elif framework in ["yolo", "faster_rcnn", "detr"]:
        return f"{framework}_{model}" if model != "default" else framework
    return folder_name


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
) -> str | None:
    """Classify pretrained file origin as cross-manuscript or synthetic."""
    if "pretrained" not in eval_file.stem:
        return None

    rel_parts = eval_file.relative_to(experiment_dir).parts
    if rel_parts and rel_parts[0] == "train_test":
        return "cross"

    source_token = get_initial_token(experiment_dir.name)
    dataset_token = get_initial_token(dataset)
    return "synthetic" if dataset_token == source_token else "cross"


def collect_sequential_data(results_dir: Path) -> dict:
    """
    Collect all sequential evaluation data from results directory.

    Returns nested dict: {dataset: {edition: {task: {framework: {seq_num: {metric: value}}}}}}
    """
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    # Find all sequential directories
    for seq_path in results_dir.rglob("sequential/random_sample/seq_*"):
        if not seq_path.is_dir():
            continue

        # Extract seq number from folder name (seq_01 -> 1, seq_02 -> 2, etc.)
        match = re.match(r"seq_(\d+)", seq_path.name)
        if not match:
            continue
        seq_num = int(match.group(1))

        # Parse path to get dataset and edition
        # Path structure: results/{experiment_id}/{dataset}/{edition}/sequential/random_sample/seq_XX/
        parts = seq_path.parts
        try:
            random_sample_idx = parts.index("random_sample")
            sequential_idx = parts.index("sequential")
            edition = parts[sequential_idx - 1]
            dataset = parts[sequential_idx - 2]
        except (ValueError, IndexError):
            print(f"Warning: Could not parse path structure for {seq_path}")
            continue

        # Find all evaluation JSONs in this seq folder
        for framework_dir in seq_path.iterdir():
            if not framework_dir.is_dir():
                continue

            framework_key = get_framework_key(framework_dir.name)

            for eval_file in framework_dir.glob("*_evaluation.json"):
                # Determine task and if pretrained
                filename = eval_file.stem
                is_pretrained = "pretrained" in filename

                # Extract task from filename (ocr_evaluation.json -> ocr)
                task_match = re.match(r"(\w+?)_(?:pretrained_)?evaluation", filename)
                if not task_match:
                    continue
                task = task_match.group(1)

                # Load metrics
                try:
                    with open(eval_file, "r") as f:
                        eval_data = json.load(f)
                    metrics = eval_data.get("metrics", {})
                except Exception as e:
                    print(f"Warning: Could not load {eval_file}: {e}")
                    continue

                # Store data with pretrained suffix if applicable
                key = f"{framework_key}_pretrained" if is_pretrained else framework_key
                data[dataset][edition][task][key][seq_num] = metrics

                print(f"Found: {dataset}/{edition}/{task}/{key} seq_{seq_num:02d}")

    return data


def collect_filtered_sequential_data(
    experiment_dir: Path, mode: str, with_source_pair_key: bool = False
) -> dict:
    """
    Collect filtered sequential evaluation data from one experiment directory.

    mode='in': non-pretrained files
    mode='cross': pretrained files classified as cross-manuscript
    mode='synthetic': pretrained files classified as synthetic pre-training
    """
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
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

        for framework_dir in seq_path.iterdir():
            if not framework_dir.is_dir():
                continue

            framework_key = get_framework_key(framework_dir.name)

            for eval_file in framework_dir.glob("*_evaluation.json"):
                filename = eval_file.stem
                is_pretrained = "pretrained" in filename

                if mode == "in" and is_pretrained:
                    continue
                if mode in {"cross", "synthetic"} and not is_pretrained:
                    continue

                if is_pretrained:
                    origin = classify_pretrained_origin(
                        experiment_dir, eval_file, dataset
                    )
                    if mode == "cross" and origin != "cross":
                        continue
                    if mode == "synthetic" and origin != "synthetic":
                        continue

                task_match = re.match(r"(\w+?)_(?:pretrained_)?evaluation", filename)
                if not task_match:
                    continue
                task = task_match.group(1)

                try:
                    with open(eval_file, "r") as f:
                        eval_data = json.load(f)
                    metrics = eval_data.get("metrics", {})
                except Exception:
                    continue

                key = f"{framework_key}_pretrained" if is_pretrained else framework_key
                dataset_key = dataset
                if mode == "cross" and with_source_pair_key:
                    dataset_key = f"{dataset}__FROM__{source_dataset}"

                data[dataset_key][edition][task][key][seq_num] = metrics

    return data


def merge_sequential_data(target: dict, source: dict) -> None:
    """Merge sequential data into target in-place."""
    for dataset, editions in source.items():
        for edition, tasks in editions.items():
            for task, frameworks in tasks.items():
                for framework, seq_data in frameworks.items():
                    for seq_num, metrics in seq_data.items():
                        target[dataset][edition][task].setdefault(framework, {})[
                            seq_num
                        ] = metrics


def get_framework_styles(task_data: dict) -> dict:
    """Assign consistent colors/markers to frameworks in a panel."""
    styles = {}
    color_idx = 0
    marker_idx = 0
    frameworks = sorted(task_data.keys(), key=lambda x: (x.endswith("_pretrained"), x))
    for framework in frameworks:
        base_framework, _ = split_cross_manuscript_key(framework)
        base_framework = base_framework.replace("_pretrained", "")
        if base_framework not in styles:
            color = FRAMEWORK_COLORS.get(
                base_framework, list(COLORS.values())[color_idx % len(COLORS)]
            )
            marker = MARKERS[marker_idx % len(MARKERS)]
            styles[base_framework] = {"color": color, "marker": marker}
            color_idx += 1
            marker_idx += 1
    return styles


def plot_task_panel(
    ax,
    task_data: dict,
    metric: str,
    ylabel: str,
    title: str,
    legend_handles: Dict[str, object],
) -> None:
    """Plot one task panel."""
    if not task_data:
        ax.set_title(title)
        ax.set_xlabel("Number of Training Steps")
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return {}

    styles = get_framework_styles(task_data)
    frameworks = sorted(task_data.keys(), key=lambda x: (x.endswith("_pretrained"), x))
    all_seq_nums = set()

    for framework in frameworks:
        base_framework, source_dataset = split_cross_manuscript_key(framework)
        base_framework = base_framework.replace("_pretrained", "")
        is_pretrained = framework.endswith("_pretrained")
        style = styles[base_framework]
        if source_dataset is None:
            linestyle = (
                LINE_STYLES["pretrained"] if is_pretrained else LINE_STYLES["normal"]
            )
        else:
            source_keys = sorted(
                {
                    split_cross_manuscript_key(name)[1]
                    for name in frameworks
                    if split_cross_manuscript_key(name)[1] is not None
                }
            )
            source_idx = source_keys.index(source_dataset)
            linestyle = CROSS_SOURCE_STYLES[source_idx % len(CROSS_SOURCE_STYLES)]

        seq_nums = sorted(task_data[framework].keys())
        all_seq_nums.update(seq_nums)
        values = []
        scale = metric in SCALE_TO_100
        for seq_num in seq_nums:
            value = task_data[framework][seq_num].get(metric)
            if value is not None and scale:
                value *= 100
            values.append(value if value is not None else np.nan)

        if not seq_nums or all(np.isnan(v) for v in values):
            continue

        label = get_display_name(framework)
        (line,) = ax.plot(
            seq_nums,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )
        legend_handles.setdefault(label, line)

    ax.set_title(title)
    ax.set_xlabel("Number of Training Steps")
    ax.set_ylabel(ylabel)
    if all_seq_nums:
        ax.set_xticks(sorted(all_seq_nums))


def save_with_shared_legend(
    fig,
    output_path: Path,
    legend_handles: Dict[str, object],
    legend_fontsize: int = 10,
    legend_ncol: int | None = None,
) -> None:
    """Save figure with one common legend for all subplots."""
    handles = list(legend_handles.values())
    labels = list(legend_handles.keys())
    if handles:
        ncol = legend_ncol if legend_ncol is not None else min(4, len(labels))
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            bbox_to_anchor=(0.5, -0.02),
            fontsize=legend_fontsize,
        )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_ocr_omr_2x2_plot(data: dict, dataset: str, output_path: Path) -> None:
    """Create one 2x2 plot for HTR/HMR by edition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    legend_handles: Dict[str, object] = {}

    tasks = [("ocr", "CER", "HTR"), ("omr", "NER", "HMR")]
    editions = ["editorial", "diplomatic"]

    for row, (task, metric, task_title) in enumerate(tasks):
        for col, edition in enumerate(editions):
            panel_data = data.get(dataset, {}).get(edition, {}).get(task, {})
            title = f"{task_title} - {edition.capitalize()}"
            ylabel = METRIC_DISPLAY.get(metric, metric)
            plot_task_panel(
                axes[row, col],
                panel_data,
                metric,
                ylabel,
                title,
                legend_handles,
            )

    fig.suptitle(f"Sequential Learning - {dataset.replace('_', ' ')}", fontsize=15)
    save_with_shared_legend(fig, output_path, legend_handles)


def create_synthetic_ocr_omr_2x2_plot(data: dict, output_path: Path) -> None:
    """Create one 2x2 OCR/OMR plot for synthetic pre-training by dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    legend_handles: Dict[str, object] = {}

    tasks = [("ocr", "CER", "HTR"), ("omr", "NER", "HMR")]
    datasets = ["I-Ct_91", "I-Fn_BR_18"]
    edition = "diplomatic"

    for row, (task, metric, task_title) in enumerate(tasks):
        for col, dataset in enumerate(datasets):
            panel_data = data.get(dataset, {}).get(edition, {}).get(task, {})
            title = f"{task_title} - {dataset.replace('_', ' ')}"
            ylabel = METRIC_DISPLAY.get(metric, metric)
            plot_task_panel(
                axes[row, col],
                panel_data,
                metric,
                ylabel,
                title,
                legend_handles,
            )

    fig.suptitle("Sequential Learning - Synthetic Pre-training", fontsize=15)
    save_with_shared_legend(fig, output_path, legend_handles)


def create_cross_ocr_omr_2x2_plot(
    data: dict, target_dataset: str, source_dataset: str, output_path: Path
) -> None:
    """Create one 2x2 OCR/OMR plot for a cross-manuscript source-target pair."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    legend_handles: Dict[str, object] = {}

    tasks = [("ocr", "CER", "HTR"), ("omr", "NER", "HMR")]
    editions = ["editorial", "diplomatic"]

    for row, (task, metric, task_title) in enumerate(tasks):
        for col, edition in enumerate(editions):
            panel_data = data.get(target_dataset, {}).get(edition, {}).get(task, {})
            title = f"{task_title} - {edition.capitalize()}"
            ylabel = METRIC_DISPLAY.get(metric, metric)
            plot_task_panel(
                axes[row, col],
                panel_data,
                metric,
                ylabel,
                title,
                legend_handles,
            )

    fig.suptitle(
        f"Cross-manuscript Pre-training - {target_dataset} → {source_dataset}",
        fontsize=15,
    )
    save_with_shared_legend(fig, output_path, legend_handles)


def create_cross_manuscript_plots(data: dict, output_dir: Path) -> None:
    """Create one OCR/OMR plot and one layout plot for cross-manuscript runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_ocr_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    combined_layout_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for pair_key, pair_data in data.items():
        target_dataset, source_dataset = pair_key.split("__FROM__", 1)
        if target_dataset not in combined_ocr_data:
            combined_ocr_data[target_dataset] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )
        if target_dataset not in combined_layout_data:
            combined_layout_data[target_dataset] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )

        for edition, tasks in pair_data.items():
            for task, frameworks in tasks.items():
                for framework, seq_data in frameworks.items():
                    for seq_num, metrics in seq_data.items():
                        if task in {"ocr", "omr"}:
                            key = f"{framework}__FROM__{source_dataset}"
                            combined_ocr_data[target_dataset][edition][task][key][
                                seq_num
                            ] = metrics
                        elif task == "layout":
                            key = f"{framework}__FROM__{source_dataset}"
                            combined_layout_data[target_dataset][edition][task][key][
                                seq_num
                            ] = metrics

    if combined_ocr_data:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        legend_handles: Dict[str, object] = {}
        tasks = [("ocr", "CER", "HTR"), ("omr", "NER", "HMR")]
        datasets = ["I-Ct_91", "I-Fn_BR_18"]
        source_targets = {
            "I-Ct_91": "I-Fn_BR_18",
            "I-Fn_BR_18": "I-Ct_91",
        }
        edition = "diplomatic"

        for row, (task, metric, task_title) in enumerate(tasks):
            for col, dataset in enumerate(datasets):
                panel_data = (
                    combined_ocr_data.get(dataset, {}).get(edition, {}).get(task, {})
                )
                title = (
                    f"{task_title} - {source_targets[dataset].replace('_', ' ')} "
                    f"→ {dataset.replace('_', ' ')}"
                )
                plot_task_panel(
                    axes[row, col],
                    panel_data,
                    metric,
                    METRIC_DISPLAY.get(metric, metric),
                    title,
                    legend_handles,
                )

        fig.suptitle("Cross-manuscript Pre-training", fontsize=15)
        save_with_shared_legend(
            fig,
            output_dir / "sequential_cross_manuscript_ocr_omr.png",
            legend_handles,
        )

    if combined_layout_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        legend_handles = {}
        datasets = ["I-Fn_BR_18", "I-Ct_91"]
        source_targets = {
            "I-Fn_BR_18": "I-Ct_91",
            "I-Ct_91": "I-Fn_BR_18",
        }
        edition = "diplomatic"

        for col, dataset in enumerate(datasets):
            panel_data = (
                combined_layout_data.get(dataset, {}).get(edition, {}).get("layout", {})
            )
            title = (
                f"Layout - {source_targets[dataset].replace('_', ' ')} "
                f"→ {dataset.replace('_', ' ')}"
            )
            plot_task_panel(
                axes[col],
                panel_data,
                "mAP",
                METRIC_DISPLAY["mAP"],
                title,
                legend_handles,
            )

        fig.suptitle("Cross-manuscript Pre-training", fontsize=15)
        save_with_shared_legend(
            fig,
            output_dir / "sequential_cross_manuscript_layout.png",
            legend_handles,
            legend_fontsize=12,
            legend_ncol=2,
        )


def create_layout_2x2_plot(data: dict, output_path: Path) -> None:
    """Create one 2x2 layout plot by dataset and edition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    legend_handles: Dict[str, object] = {}

    datasets = ["I-Fn_BR_18", "I-Ct_91"]
    editions = ["diplomatic", "editorial"]

    for row, dataset in enumerate(datasets):
        for col, edition in enumerate(editions):
            panel_data = data.get(dataset, {}).get(edition, {}).get("layout", {})
            title = f"{dataset.replace('_', ' ')} - {edition.capitalize()}"
            plot_task_panel(
                axes[row, col],
                panel_data,
                "mAP",
                METRIC_DISPLAY["mAP"],
                title,
                legend_handles,
            )

    fig.suptitle("Sequential Learning - Layout", fontsize=15)
    save_with_shared_legend(
        fig,
        output_path,
        legend_handles,
        legend_fontsize=12,
        legend_ncol=2,
    )


def generate_plots(results_dir: Path) -> int:
    """Generate compact 2x2 plots for sequential training results."""

    print(f"Scanning results directory: {results_dir}")
    experiment_dirs = [
        p
        for p in results_dir.iterdir()
        if p.is_dir() and p.name not in {"plots", "tables"}
    ]
    pattern_groups = defaultdict(list)
    for exp_dir in experiment_dirs:
        pattern_groups[get_experiment_pattern(exp_dir.name)].append(exp_dir)

    if check_for_duplicate_relative_json_paths(experiment_dirs):
        return 1

    if not pattern_groups:
        print("No experiment directories found!")
        return 0

    pattern_data_in = {}
    pattern_data_synthetic = {}
    pattern_data_cross = {}
    for pattern, exp_dirs in pattern_groups.items():
        print(f"\nProcessing pattern: {pattern}")
        data_in = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        data_synthetic = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        data_cross = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        for exp_dir in exp_dirs:
            merge_sequential_data(
                data_in, collect_filtered_sequential_data(exp_dir, "in")
            )
            merge_sequential_data(
                data_synthetic,
                collect_filtered_sequential_data(exp_dir, "synthetic"),
            )
            merge_sequential_data(
                data_cross,
                collect_filtered_sequential_data(
                    exp_dir, "cross", with_source_pair_key=True
                ),
            )

        if not data_in and not data_synthetic:
            print(f"No sequential data found for pattern {pattern}!")
            continue

        pattern_data_in[pattern] = data_in
        pattern_data_synthetic[pattern] = data_synthetic
        pattern_data_cross[pattern] = data_cross

        plots_dir = results_dir / "plots" / pattern

        if pattern == "I-Ct_91" and data_in:
            create_ocr_omr_2x2_plot(
                data_in,
                "I-Ct_91",
                plots_dir / "sequential_ocr_omr_I-Ct_91.png",
            )

        if pattern == "I-Fn_BR_18" and data_in:
            create_ocr_omr_2x2_plot(
                data_in,
                "I-Fn_BR_18",
                plots_dir / "sequential_ocr_omr_I-Fn_BR_18.png",
            )

    combined_layout_data_in = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    combined_layout_data_synthetic = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for key in ["I-Fn_BR_18", "I-Ct_91"]:
        if key in pattern_data_in:
            merge_sequential_data(combined_layout_data_in, pattern_data_in[key])
        if key in pattern_data_synthetic:
            merge_sequential_data(
                combined_layout_data_synthetic, pattern_data_synthetic[key]
            )

    if combined_layout_data_synthetic:
        create_synthetic_ocr_omr_2x2_plot(
            combined_layout_data_synthetic,
            results_dir / "plots" / "sequential_synthetic_pretraining_ocr_omr.png",
        )

    combined_cross_pair_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for cross_data in pattern_data_cross.values():
        merge_sequential_data(combined_cross_pair_data, cross_data)

    if combined_cross_pair_data:
        create_cross_manuscript_plots(
            combined_cross_pair_data, results_dir / "plots" / "cross_manuscript"
        )

    if combined_layout_data_in:
        create_layout_2x2_plot(
            combined_layout_data_in,
            results_dir / "plots" / "sequential_layout_I-Fn_BR_18_I-Ct_91.png",
        )

    print("\nPlot generation complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate line plots for sequential training results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory (default: results)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    return generate_plots(args.results_dir)


if __name__ == "__main__":
    exit(main())
