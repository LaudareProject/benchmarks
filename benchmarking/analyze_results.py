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
        return ["detr", "faster_rcnn", "yolo", "doclayout_yolo"]
    elif task == "ocr":
        return ["kraken", "calamari", "trocr", "paddleocr_vl", "vlt"]
    elif task == "omr":
        return ["kraken", "calamari", "trocr", "paddleocr_vl", "vlt", "bgk"]
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
    """Analyze one framework without aggregating partial fold coverage."""
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
    missing_metric_folds = {metric: [] for metric in key_metrics}
    available_folds = []
    missing_folds = []

    for i in range(args.num_folds):
        fold_dir = results_base_dir / f"fold_{i}"
        eval_file = (
            fold_dir / f"{args.framework}_{args.task}_{args.model_name}" / path_suffix
        )

        if not eval_file.exists():
            print(f"   - Fold {i}: Evaluation file not found at {eval_file}")
            missing_folds.append(i)
            for metric in key_metrics:
                missing_metric_folds[metric].append(i)
            continue

        available_folds.append(i)
        with open(eval_file, "r") as f:
            data = json.load(f)

        metrics_data = data.get("metrics", data)

        for metric in key_metrics:
            if metric in metrics_data:
                collected_metrics[metric].append(metrics_data[metric])
            else:
                missing_metric_folds[metric].append(i)

    output_metrics = {}

    for metric, values in collected_metrics.items():
        missing = missing_metric_folds[metric]
        if missing:
            print(
                f"Metric '{metric}': incomplete; missing from fold(s) "
                f"{', '.join(map(str, missing))}."
            )
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
            ci_interval = tuple(float(bound) for bound in ci)
        else:
            ci_interval = None

        output_metrics[metric] = {
            "mean": float(mean),
            "min": float(min_val),
            "max": float(max_val),
            "95_ci": ci_interval,
            "values": values.tolist(),
        }

    incomplete_metrics = {
        metric: missing
        for metric, missing in missing_metric_folds.items()
        if missing
    }
    is_complete = not missing_folds and not incomplete_metrics
    result = {
        "status": "complete" if is_complete else "incomplete",
        "expected_folds": args.num_folds,
        "available_folds": available_folds,
        "missing_folds": missing_folds,
        "missing_metrics": incomplete_metrics,
        "metrics": output_metrics,
    }
    print(
        f"Aggregation status: {result['status']} "
        f"({len(available_folds)}/{args.num_folds} evaluation files found)."
    )

    if args.output_file:
        print(f"   -> Saving aggregation report to: {args.output_file}")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"   -> Aggregation report saved to {args.output_file}")

    return result


def create_summary_table(args):
    """Create a cross-framework CSV with fold coverage for each metric."""
    print("\n--- Aggregated Cross-Framework Results ---")

    frameworks = get_frameworks_for_task(args.task)

    # Use a dummy framework from the list to get the key metrics for the header
    dummy_args = argparse.Namespace(**vars(args))
    dummy_args.framework = frameworks[0]
    _, key_metrics = get_eval_path_and_metrics(dummy_args)

    header = [
        "Framework",
        "Framework Status",
        "Metric",
        "Metric Status",
        "Fold Coverage",
        "Mean",
        "Min",
        "Max",
        "95% CI",
        "Missing Folds",
        "Missing Metric Folds",
    ]
    table_data = [header]

    print(
        f"\n{'-' * 80}\nTask: {args.task.upper()}, Dataset: {args.edition}\n{'-' * 80}"
    )

    for fw in frameworks:
        fw_args = argparse.Namespace(**vars(args))
        fw_args.framework = fw
        fw_args.output_file = None

        fw_result = analyze_single_framework(fw_args)
        global_missing = ", ".join(map(str, fw_result["missing_folds"]))

        for metric in key_metrics:
            stats = fw_result["metrics"].get(metric)
            metric_missing = fw_result["missing_metrics"].get(metric, [])
            metric_status = "complete" if stats is not None else "incomplete"
            metric_coverage = args.num_folds - len(metric_missing)
            fold_coverage = f"{metric_coverage}/{args.num_folds}"

            if stats is None:
                mean = min_val = max_val = ci_str = ""
                print(
                    f"{fw:<15} | {metric:<10} | incomplete | "
                    f"{fold_coverage} folds; missing metric folds "
                    f"{', '.join(map(str, metric_missing))}"
                )
            else:
                mean = f"{stats['mean']:.4f}"
                min_val = f"{stats['min']:.4f}"
                max_val = f"{stats['max']:.4f}"
                ci = stats["95_ci"]
                ci_str = (
                    f"({ci[0]:.4f}, {ci[1]:.4f})"
                    if ci and ci[0] is not None
                    else "N/A"
                )
                print(
                    f"{fw:<15} | {metric:<10} | {metric_status:<10} | "
                    f"{fold_coverage} | {mean} | {min_val} | {max_val} | {ci_str}"
                )

            table_data.append(
                [
                    fw,
                    fw_result["status"],
                    metric,
                    metric_status,
                    fold_coverage,
                    mean,
                    min_val,
                    max_val,
                    ci_str,
                    global_missing,
                    ", ".join(map(str, metric_missing)),
                ]
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
        choices=[
            "kraken",
            "calamari",
            "faster_rcnn",
            "yolo",
            "doclayout_yolo",
            "trocr",
            "paddleocr_vl",
            "vlt",
            "detr",
            "bgk",
            "all",
        ],
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
        "--num-folds",
        type=int,
        required=True,
        help="Number of folds expected in the aggregate",
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

    if args.num_folds <= 0:
        parser.error("--num-folds must be greater than zero")

    if args.framework == "all":
        create_summary_table(args)
        return

    result = analyze_single_framework(args)
    if result["metrics"]:
        print("\n--- Summary ---")
        for metric, values in result["metrics"].items():
            ci_str = (
                f"({values['95_ci'][0]:.4f}, {values['95_ci'][1]:.4f})"
                if values["95_ci"] and values["95_ci"][0] is not None
                else "N/A"
            )
            print(f"Metric: {metric}")
            print(
                f"  - Mean: {values['mean']:.4f}, Min: {values['min']:.4f}, "
                f"Max: {values['max']:.4f}, 95% CI: {ci_str}"
            )
    else:
        print("No complete metric results to display.")

    if result["status"] == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
