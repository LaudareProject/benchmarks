import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import scipy.stats

# Use experiment ID for isolation of results, matching utils.py
EXPERIMENT_ID = os.environ.get("LAUDARE_EXPERIMENT_ID", "default")


def get_frameworks_for_task(task):
    """Returns a list of frameworks applicable to a given task."""
    if task == "layout":
        return ["kraken", "faster_rcnn", "yolo", "detr"]
    elif task == "ocr":
        return ["kraken", "calamari", "trocr"]
    elif task == "omr":
        return ["kraken", "calamari", "trocr"]
    return []


def get_eval_path_and_metrics(args):
    """Determines the evaluation file path and key metrics based on args."""
    task = args.task
    fw = args.framework
    model_name = args.model_name

    if task == "layout":
        key_metrics = ["mAP", "mAP@0.5", "mAP@0.75", "f1@0.50", "f1@0.75"]
        return "layout_evaluation.json", key_metrics
    elif task == "ocr":
        key_metrics = ["WER", "CER"]
        return "ocr_evaluation.json", key_metrics
    elif task == "omr":
        key_metrics = ["NER", "CER"]
        return "omr_evaluation.json", key_metrics

    raise ValueError(f"Unsupported combination: task={task}, framework={fw}")


def analyze_single_framework(args):
    """Analyzes results for a single framework and returns summary stats."""
    print(f"\n--- Analyzing: {args.framework.upper()} for {args.task.upper()} ---")

    results_base_dir = (
        Path("results") / EXPERIMENT_ID / args.data_dir.name / args.edition
    )
    try:
        path_suffix, key_metrics = get_eval_path_and_metrics(args)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return {}

    collected_metrics = {metric: [] for metric in key_metrics}

    for i in range(args.num_folds):
        fold_dir = results_base_dir / f"fold_{i}"
        eval_file = (
            fold_dir / f"{args.framework}_{args.task}_{args.model_name}" / path_suffix
        )

        if not eval_file.exists():
            print(f"   - Fold {i}: Evaluation file not found at {eval_file}")
            continue

        with open(eval_file, "r") as f:
            data = json.load(f)

        metrics_data = data.get("metrics", data)

        for metric in key_metrics:
            if metric in metrics_data:
                collected_metrics[metric].append(metrics_data[metric])

    output_data = {}

    for metric, values in collected_metrics.items():
        if not values:
            print(f"Metric '{metric}': No data found.")
            continue

        values = np.array(values)
        mean = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        if len(values) > 1:
            ci = scipy.stats.t.interval(
                0.95,
                len(values) - 1,
                loc=np.mean(values),
                scale=scipy.stats.sem(values),
            )
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = None, None

        output_data[metric] = {
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "95_ci": (ci_lower, ci_upper) if ci_lower is not None else None,
            "values": values.tolist(),
        }

    if args.output_file:
        print(f"   -> Saving aggregated JSON results to: {args.output_file}")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(
            f"   -> Aggregated results for {args.framework.upper()} saved to {args.output_file}"
        )

    return output_data


def create_summary_table(args):
    """Creates a summary CSV table comparing all relevant frameworks for a task."""
    print("\n--- Aggregated Cross-Framework Results ---")

    frameworks = get_frameworks_for_task(args.task)

    # Use a dummy framework from the list to get the key metrics for the header
    dummy_args = argparse.Namespace(**vars(args))
    dummy_args.framework = frameworks[0]
    _, key_metrics = get_eval_path_and_metrics(dummy_args)

    header = ["Framework", "Metric", "Mean", "Min", "Max", "95% CI"]
    table_data = [header]

    print(
        f"\n{'-' * 80}\nTask: {args.task.upper()}, Dataset: {args.edition}\n{'-' * 80}"
    )
    print(
        f"{'Framework':<15} | {'Metric':<10} | {'Mean':<8} | {'Min':<8} | {'Max':<8} | {'95% CI':<20}"
    )
    print("=" * 80)

    for fw in frameworks:
        fw_args = argparse.Namespace(**vars(args))
        fw_args.framework = fw
        fw_args.output_file = None

        fw_stats = analyze_single_framework(fw_args)

        if not fw_stats:
            print(f"{fw:<15} | {'No data found':<63}")
            continue

        for metric in key_metrics:
            if metric in fw_stats:
                stats = fw_stats[metric]
                mean = stats["mean"]
                min_val = stats["min"]
                max_val = stats["max"]
                ci = stats["95_ci"]
                ci_str = (
                    f"({ci[0]:.4f}, {ci[1]:.4f})" if ci and ci[0] is not None else "N/A"
                )

                row = [
                    fw,
                    metric,
                    f"{mean:.4f}",
                    f"{min_val:.4f}",
                    f"{max_val:.4f}",
                    ci_str,
                ]
                table_data.append(row)
                print(
                    f"{fw:<15} | {metric:<10} | {mean:<8.4f} | {min_val:<8.4f} | {max_val:<8.4f} | {ci_str:<20}"
                )

    # Save to CSV
    csv_output_dir = (
        Path("results")
        / EXPERIMENT_ID
        / args.data_dir.name
        / args.edition
        / "aggregated"
    )
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_file = csv_output_dir / f"all_frameworks_{args.task}_summary.csv"

    print(f"\n   -> Saving aggregated CSV summary to: {csv_output_file}")

    with open(csv_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table_data)
    print(f"\n✅ Summary table saved to {csv_output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze N-fold cross-validation results."
    )
    parser.add_argument(
        "--edition",
        type=str,
        required=True,
        choices=["diplomatic", "editorial"],
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["kraken", "calamari", "faster_rcnn", "yolo", "trocr", "detr", "all"],
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["ocr", "omr", "layout"]
    )
    parser.add_argument(
        "--model-index",
        type=int,
        help="Model index used for models that have different versions (e.g., yolo, faster_rcnn).",
    )
    parser.add_argument(
        "--num-folds", type=int, required=True, help="Number of folds that were run"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="File to save the aggregated JSON results (single framework only)",
    )
    parser.add_argument(
        "--model-name", type=str, help="The model name that must be used"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to the data directory"
    )

    args = parser.parse_args()

    if args.framework == "all":
        create_summary_table(args)
    else:
        stats = analyze_single_framework(args)
        if stats:
            print("\n--- Summary ---")
            for metric, values in stats.items():
                ci_str = (
                    f"({values['95_ci'][0]:.4f}, {values['95_ci'][1]:.4f})"
                    if values["95_ci"] and values["95_ci"][0] is not None
                    else "N/A"
                )
                print(f"Metric: {metric}")
                print(
                    f"  - Mean: {values['mean']:.4f}, Min: {values['min']:.4f}, Max: {values['max']:.4f}, 95% CI: {ci_str}"
                )
        else:
            print("No results to display.")


if __name__ == "__main__":
    main()
