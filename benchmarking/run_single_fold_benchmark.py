import argparse
import sys
import importlib
from pathlib import Path

from .utils import (
    setup_with_args,
    get_pretrained_model_path,
)
from .evaluation import evaluate_predictions_entry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.absolute()


def setup_pretrain_training(args, task: str,
                            pretrained_model_path: Path) -> dict:
    """Setup training configuration for pretraining on custom or synthetic data."""
    # Create a copy of args for pretrain setup
    pretrain_args = argparse.Namespace(**vars(args))

    # Use custom pretrain directory if provided, otherwise default to synthetic data
    if hasattr(args, "pretrain_dir") and args.pretrain_dir:
        # Use custom directory for pretraining
        pretrain_task_dir = args.pretrain_dir
        print(f"   📂 Using custom pretrain directory: {pretrain_task_dir}")
    else:
        # Override paths for synthetic data - use train_test mode
        pretrain_base_dir = PROJECT_ROOT / "data" / "pretrain_data"
        pretrain_task_dir = pretrain_base_dir / task
        print(f"   🧪 Using synthetic pretrain data: {pretrain_task_dir}")

    # Clear sequential mode attributes for pretraining
    pretrain_args.sequential_step = None
    pretrain_args.sequential_strategy = None

    # For pretrain, we fake a train/test mode, so we need to specify both train_dir and test_dir, but
    # later we will remove the test.json, so no test is actually performed.
    pretrain_args.train_dir = pretrain_task_dir
    pretrain_args.test_dir = pretrain_task_dir
    pretrain_args.data_dir = None

    # Set up pretrain output directory
    pretrained_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Call setup_with_args for pretrain configuration
    setup_dict = setup_with_args(pretrain_args)

    # Override save path to our pretrained models directory
    setup_dict["save_model_path"] = pretrained_model_path
    setup_dict["test_json"] = None  # No evaluation during pretraining

    return setup_dict


def load_and_run_framework(framework: str, setup_dict: dict):
    """Dynamically load and execute framework training function"""
    module_name = f"train_test_{framework}"
    function_name = f"train_test_{framework}"

    try:
        # Dynamic import using importlib for better error handling
        module = importlib.import_module(
            f"benchmarking.train_test.{module_name}")
        train_test_func = getattr(module, function_name)
        train_test_func(**setup_dict)

    except ImportError as e:
        raise RuntimeError(f"Could not import {module_name}: {e}")
    except AttributeError as e:
        raise RuntimeError(
            f"Function {function_name} not found in {module_name}: {e}")


def run_framework_task(args):
    """Modified function to use direct function calls instead of subprocess."""

    print(
        f"📝 Training {args.task.upper()} model with {args.framework.upper()}..."
    )
    print(f"   - Model name: {args.model_name}")
    if args.enable_pretrain:
        print("   🔄 Pre-training enabled")

    # Handle pretraining workflow if enabled
    if args.enable_pretrain:
        pretrained_model_path = get_pretrained_model_path(
            args.framework,
            args.task,
            args.model_name,
            getattr(args, "pretrain_dir", None),
        )
        if pretrained_model_path.exists():
            print(
                f"   ✅ Found existing pretrained model: {pretrained_model_path}"
            )
        else:
            print(
                "   🏗️ No pretrained model found, training on synthetic data..."
            )

            # Setup and run pretraining
            pretrain_setup_dict = setup_pretrain_training(
                args, args.task, pretrained_model_path)

            print(
                f"   📚 Pre-training {args.framework} on synthetic {args.task} data..."
            )
            load_and_run_framework(args.framework, pretrain_setup_dict)
            print(
                f"   ✅ Pre-training complete, model saved to: {pretrain_setup_dict['save_model_path']}"
            )

    # Call setup function that handles all path configuration
    setup_dict = setup_with_args(args)

    # Use function call instead of subprocess
    load_and_run_framework(args.framework, setup_dict)

    print(f"✅ {args.task.upper()} training complete")

    # Run evaluation if needed
    if setup_dict[
            "test_json"] is not None:  # Only evaluate if we have test data
        print(f"📊 Evaluating {args.task.upper()} model...")
        pred_dir = setup_dict[
            "output_dir"]  # This already points to output_dir/predictions
        if pred_dir is not None:
            eval_output_file = Path(
                pred_dir
            ).parent / f"{args.task}_{'pretrained_' if args.enable_pretrain else ''}evaluation.json"

            results = evaluate_predictions_entry(
                predictions_dir=str(pred_dir),
                ground_truth_json=str(setup_dict["test_json"]),
                task=args.task,
                output_file=str(eval_output_file),
                debug=args.debug,
            )
            print(
                f"✅ {args.task.upper()} evaluation complete. Results in {eval_output_file}"
            )
            return results
        else:
            print("⏭️ Skipping evaluation (no output directory)")
            return None
    else:
        print("⏭️ Skipping evaluation (no test data)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run single fold benchmark for specified dataset.")
    parser.add_argument(
        "--edition",
        type=str,
        default="diplomatic",
        choices=["diplomatic", "editorial"],
        help="Dataset type to use (diplomatic or editorial).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced training settings.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run only a specific task (e.g. ocr, omr, layout).",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="all",
        help="Framework to benchmark.",
    )
    parser.add_argument(
        "--enable-pretrain",
        action="store_true",
        help="Enable pre-training on synthetic data",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number to run the benchmark on.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="default",
        help="Name of the model to use from models.json.",
    )
    parser.add_argument(
        "--sequential-step",
        type=int,
        help="The index of the sequential step to run.",
    )
    parser.add_argument(
        "--sequential-strategy",
        type=str,
        default="cumulative",
        help="Strategy for sequential learning.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use data augmentation.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        help=
        "Directory containing training data (uses gt.json from annotations subdirectory).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help=
        "Directory containing test data (uses gt.json from annotations subdirectory).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=
        "Base directory for dataset (the one containing images, according to the Laudare structure).",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        help=
        "Custom directory for pretraining data (overrides default synthetic data).",
    )
    args = parser.parse_args()

    print(
        f"🔬 Benchmarking: {args.framework}-{args.task} ({args.edition.title()})"
    )
    if args.debug:
        print("🐛 DEBUG MODE: Reduced training settings")
    if args.framework != "all":
        print(f"🚀 Framework: {args.framework.upper()}")
    print(f"💪 Augmentation: {args.augment}")
    print("")

    # Use global pre-training data if enabled
    if args.enable_pretrain:
        if hasattr(args, "pretrain_dir") and args.pretrain_dir:
            # Check custom pretrain directory
            if not args.pretrain_dir.exists():
                print(
                    f"❌ Custom pre-training directory does not exist: {args.pretrain_dir}"
                )
                sys.exit(1)
            print(
                f"📚 Using custom pre-training data from: {args.pretrain_dir}")
        else:
            # Check default synthetic data directory
            pretrain_base_dir = PROJECT_ROOT / "data" / "pretrain_data"
            if not pretrain_base_dir.exists():
                print(
                    f"❌ Pre-training enabled but no data found at: {pretrain_base_dir}"
                )
                print(
                    "   Run the workflow with --task synthesis first to generate pre-training data"
                )
                sys.exit(1)
            print(
                f"📚 Using synthetic pre-training data from: {pretrain_base_dir}"
            )

    if not args.framework:
        print("❌ no framework specified!")
        exit(1)

    if not args.task:
        print("❌ no task specified!")
        exit(1)

    # Run benchmarks for each framework-task combination
    print(f"🚀 {args.framework.upper()} BENCHMARKS")
    print("─" * (len(args.framework) + 12))

    if args.data_dir is None and (args.test_dir is None
                                  or args.train_dir is None):
        print(
            "❌ Must specify either data directory or both train and test directories!"
        )
        exit(1)
    if bool(args.test_dir) != bool(args.train_dir):
        print("❌ Must specify both train and test directories!")
        exit(1)
    run_framework_task(args)

    print("")

    print(
        f"✅ Benchmark complete for {args.framework}-{args.task} ({args.edition})"
    )


if __name__ == "__main__":
    main()
