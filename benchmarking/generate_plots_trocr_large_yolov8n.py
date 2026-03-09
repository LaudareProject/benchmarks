"""
Generate sequential-learning plots limited to TrOCR-Large and YOLOv8n.

Produces:
- one HTR plot (all TrOCR-Large lines)
- one HMR plot (all TrOCR-Large lines)
- one Layout plot (all YOLOv8n lines)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from benchmarking.duplicate_guard import check_for_duplicate_relative_json_paths
except ImportError:
    from duplicate_guard import check_for_duplicate_relative_json_paths

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

COLORS = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
    "#1B9E77",
    "#D95F02",
    "#7570B3",
]

MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*"]

METRIC_DISPLAY = {
    "CER": "CER (%)",
    "NER": "NER (%)",
    "mAP": "mAP (%)",
}

SCALE_TO_100 = {"mAP", "mAP@0.5", "f1@0.50"}


def parse_framework_task_model(folder_name: str):
    parts = folder_name.split("_")
    if len(parts) >= 3:
        framework = parts[0]
        task = parts[1]
        model = "_".join(parts[2:])
        return framework, task, model
    return None, None, None


def get_framework_key(folder_name: str) -> str:
    framework, _task, model = parse_framework_task_model(folder_name)
    if framework is None:
        return folder_name
    return f"{framework}_{model}" if model != "default" else framework


def get_experiment_pattern(experiment_name: str) -> str:
    parts = experiment_name.split("_")
    if len(parts) >= 4 and all(part.isdigit() for part in parts[-3:]):
        return "_".join(parts[:-3])
    return experiment_name


def collect_sequential_data(results_dir: Path) -> dict:
    data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for seq_path in results_dir.rglob("sequential/random_sample/seq_*"):
        if not seq_path.is_dir():
            continue

        match = re.match(r"seq_(\d+)", seq_path.name)
        if not match:
            continue
        seq_num = int(match.group(1))

        parts = seq_path.parts
        sequential_idx = parts.index("sequential")
        edition = parts[sequential_idx - 1]
        dataset = parts[sequential_idx - 2]

        for framework_dir in seq_path.iterdir():
            if not framework_dir.is_dir():
                continue

            framework_key = get_framework_key(framework_dir.name)

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


def merge_sequential_data(target: dict, source: dict) -> None:
    for dataset, editions in source.items():
        for edition, tasks in editions.items():
            for task, frameworks in tasks.items():
                for framework, seq_data in frameworks.items():
                    for seq_num, metrics in seq_data.items():
                        target[dataset][edition][task].setdefault(framework, {})[
                            seq_num
                        ] = metrics


def iter_series(data: dict, task: str, allowed_frameworks: set[str]):
    for dataset in sorted(data.keys()):
        for edition in sorted(data[dataset].keys()):
            task_data = data[dataset][edition].get(task, {})
            for framework in sorted(task_data.keys()):
                if framework not in allowed_frameworks:
                    continue
                yield dataset, edition, framework, task_data[framework]


def plot_single_task(
    data: dict,
    task: str,
    metric: str,
    allowed_frameworks: set[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots()

    all_seq_nums = set()
    datasets = sorted(data.keys())
    dataset_colors = {
        dataset: COLORS[idx % len(COLORS)] for idx, dataset in enumerate(datasets)
    }
    edition_markers = {
        "diplomatic": "o",
        "editorial": "s",
    }

    for dataset, edition, framework, seq_metrics in iter_series(
        data, task, allowed_frameworks
    ):
        color = dataset_colors[dataset]
        marker = edition_markers.get(edition, "^")
        is_pretrained = framework.endswith("_pretrained")
        linestyle = (0, (6, 3)) if is_pretrained else "-"

        seq_nums = sorted(seq_metrics.keys())
        all_seq_nums.update(seq_nums)
        values = []
        for seq_num in seq_nums:
            value = seq_metrics[seq_num].get(metric)
            if value is not None and metric in SCALE_TO_100:
                value *= 100
            values.append(value if value is not None else np.nan)

        if not seq_nums or all(np.isnan(v) for v in values):
            continue

        framework_label = (
            "TrOCR-Large" if framework.startswith("trocr_large") else "YOLOv8n"
        )
        suffix = " (cross-manuscript)" if is_pretrained else ""
        label = f"{dataset} | {edition} | {framework_label}{suffix}"

        ax.plot(
            seq_nums,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Number of Training Steps")
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
    if all_seq_nums:
        ax.set_xticks(sorted(all_seq_nums))
    handles, labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2),
        Line2D([0], [0], color="black", linestyle=(0, (6, 3)), linewidth=2),
    ]
    style_labels = ["within-manuscript", "cross-manuscript"]
    ax.legend(
        handles + style_handles,
        labels + style_labels,
        loc="best",
        frameon=True,
        handlelength=3.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_plots(results_dir: Path) -> int:
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

    combined_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for _pattern, exp_dirs in pattern_groups.items():
        data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        for exp_dir in exp_dirs:
            merge_sequential_data(data, collect_sequential_data(exp_dir))
        merge_sequential_data(combined_data, data)

    trocr_large_only = {"trocr_large", "trocr_large_pretrained"}
    yolov8n_only = {"yolo_yolov8n", "yolo_yolov8n_pretrained"}

    plots_dir = results_dir / "plots"
    plot_single_task(
        combined_data,
        task="ocr",
        metric="CER",
        allowed_frameworks=trocr_large_only,
        title="Sequential Learning - HTR (TrOCR-Large)",
        output_path=plots_dir / "sequential_htr_trocr_large.png",
    )
    plot_single_task(
        combined_data,
        task="omr",
        metric="NER",
        allowed_frameworks=trocr_large_only,
        title="Sequential Learning - HMR (TrOCR-Large)",
        output_path=plots_dir / "sequential_hmr_trocr_large.png",
    )
    plot_single_task(
        combined_data,
        task="layout",
        metric="mAP",
        allowed_frameworks=yolov8n_only,
        title="Sequential Learning - Layout (YOLOv8n)",
        output_path=plots_dir / "sequential_layout_yolov8n.png",
    )

    print("Plot generation complete!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sequential plots for TrOCR-Large and YOLOv8n only."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory (default: results)",
    )
    args = parser.parse_args()

    return generate_plots(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
