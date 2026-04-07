import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
import tempfile
from concurrent.futures import ThreadPoolExecutor

from ..utils import path_json2pagexml, get_adaptive_num_workers, get_adaptive_batch_size

# Define paths to executables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

KRAKEN_EXEC = PROJECT_ROOT / ".venv-kraken" / "bin" / "kraken"
KETOS_EXEC = PROJECT_ROOT / ".venv-kraken" / "bin" / "ketos"


def run_kraken_command(
    command: list, cwd: Optional[Path] = None, log_file: Optional[Path] = None
):
    """Runs a Kraken shell command and logs its output."""
    command = [str(c) for c in command]  # Ensure all parts are strings
    try:
        if log_file:
            with open(log_file, "a") as lf:
                lf.write(f"Running command: {' '.join(command)}\n")
                result = subprocess.run(
                    command, cwd=cwd, text=True, capture_output=True, check=False
                )
                lf.write(f"Stdout: {result.stdout}\n")
                if result.stderr:
                    lf.write(f"Stderr: {result.stderr}\n")
                lf.write(f"Return code: {result.returncode}\n")
        else:
            result = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, check=False
            )

        if result.returncode != 0:
            error_message = f"   ❌ Command failed with exit code {result.returncode}: {' '.join(command)}\n   Stderr: {result.stderr.strip()}"
            print(error_message)
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout, stderr=result.stderr
            )

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   ❌ Error running command: {e}")
        raise e


def predict_single_file(
    xml_file: Path, output_path: Path, best_model: Path, cuda: bool, log_file: Path
):
    """Helper function to predict a single file."""
    output_xml_file = output_path / f"{xml_file.stem}.xml"
    predict_cmd = [str(KRAKEN_EXEC)]

    if cuda and os.environ.get("CUDA_VISIBLE_DEVICES"):
        predict_cmd.extend(["--device", "cuda:0"])
    else:
        predict_cmd.extend(["--device", "cpu"])

    predict_cmd += [
        "--pagexml",
        "-f",
        "xml",
        "-i",
        xml_file,
        output_xml_file,
        "ocr",
        "-m",
        best_model,
    ]
    run_kraken_command(predict_cmd, log_file=log_file)
    return xml_file.stem


def train_kraken_ocr_omr(
    save_model_path: Path,
    artifacts_path: Path,
    train_json: Path,
    val_json: Path,
    test_json: Optional[Path],
    output_path: Path,
    log_file: Path,
    task: str,
    edition: str,
    train_data_dir: Path,
    test_data_dir: Path,
    debug: bool = False,
    cuda: bool = True,
    pretrained_model: Optional[str] = None,
    augment: bool = False,
):
    """
    Trains a Kraken model for OCR or OMR with evaluation.
    """

    # Get adaptive parameters with LR scaling
    base_batch_size = 4  # Kraken default
    base_lr = 0.0001  # Kraken default (1e-4)

    adaptive_batch_size = get_adaptive_batch_size(
        train_json, train_data_dir, base_batch_size=base_batch_size
    )
    adaptive_workers = get_adaptive_num_workers(train_json, train_data_dir)

    # Linear scaling rule for learning rate
    scaled_lr = base_lr * (adaptive_batch_size / base_batch_size)

    print(f"   📊 Adaptive batch size: {adaptive_batch_size}")
    print(f"   📈 Scaled learning rate: {scaled_lr:.2e} (base: {base_lr:.2e})")
    print(f"   👷 Adaptive workers for training: {adaptive_workers}")

    # Get training and validation files from JSON
    training_files = path_json2pagexml(train_json, task, edition, train_data_dir)
    val_files = path_json2pagexml(val_json, task, edition, train_data_dir)

    # Limit training files in debug mode
    if debug:
        training_files = training_files[:10]
        val_files = val_files[:3]
        print(f"   🐛 Using {len(training_files)} files (debug mode)")
    else:
        print(f"   📄 Found {len(training_files)} training files")

    # Create file lists for ketos
    train_list_file = artifacts_path / "train_files.txt"
    with open(train_list_file, "w") as f:
        for file_path in training_files:
            f.write(f"{file_path}\n")

    val_list_file = artifacts_path / "val_files.txt"
    with open(val_list_file, "w") as f:
        for file_path in val_files:
            f.write(f"{file_path}\n")

    # Set epochs based on debug mode
    epochs = "2" if debug else "100"

    print(f"   ⚙️  Training model ({epochs} epochs)...")

    # Build training command
    train_cmd = [
        str(KETOS_EXEC),
        "train",
        "-f",
        "xml",
    ]

    # Add device
    if cuda and os.environ.get("CUDA_VISIBLE_DEVICES"):
        train_cmd.extend(["--device", "cuda:0"])
    else:
        train_cmd.extend(["--device", "cpu"])

    train_cmd.extend(
        [
            "-o",
            str(artifacts_path / "model"),
            "--epochs",
            epochs,
            "--freq",
            "1",
            "-t",
            str(train_list_file),
            "-e",
            str(val_list_file),
            "--resize",
            "union",
            "--workers",
            str(adaptive_workers),
            "--batch-size",
            str(adaptive_batch_size),
            "--lrate",
            str(scaled_lr),
        ]
    )

    # Add pretrained model if specified
    if pretrained_model:
        train_cmd.extend(["-i", pretrained_model])
        print(f"   🔄 Fine-tuning from pretrained model: {pretrained_model}")
    else:
        print("   🆕 Training from scratch")

    # Add augmentation if enabled
    if augment:
        train_cmd.append("--augment")
        print("   💪 Augmentation enabled.")

    # Add early stopping: disable for debug mode, enable with patience=1 for testing
    if debug:
        train_cmd.extend(["--quit", "early", "--lag", "1"])
    else:
        train_cmd.extend(["--quit", "early", "--lag", "15"])

    # Run training
    run_kraken_command(train_cmd, log_file=log_file)

    # Find trained model
    possible_models = [
        artifacts_path / "model_best.mlmodel",
        artifacts_path / "model.mlmodel",
        artifacts_path / "model_0.mlmodel",
    ]

    best_model = None
    for model_path in possible_models:
        if model_path.exists():
            best_model = model_path
            break

    if best_model is None:
        print("   ❌ No trained model found!")
        return

    print("   ✅ Model trained successfully")

    # Copy model to standardized path if specified
    if save_model_path:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, save_model_path)
        print(f"   📁 Model saved to standardized path: {save_model_path}")

    if test_json is None:
        return

    test_files = path_json2pagexml(test_json, task, edition, test_data_dir)
    if debug:
        test_files = test_files[:3]
    print(f"   🔮 Running predictions on {len(test_files)} test files...")
    output_path.mkdir(parents=True, exist_ok=True)

    n_jobs = min(len(test_files), int(os.environ.get("N_JOBS", "10")))
    print(f"   🚀 Using {n_jobs} parallel jobs for predictions")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                predict_single_file, xml_file, output_path, best_model, cuda, log_file
            )
            for xml_file in test_files
        ]

        for future in futures:
            future.result()


def train_test_kraken(
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
    """Entry point for Kraken training with expanded parameter signature."""

    # Create log file
    log_file = (
        PROJECT_ROOT
        / "logs"
        / f"kraken_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cuda_available = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))

    artifacts_path = Path(tempfile.mkdtemp(prefix="kraken_artifacts_"))

    if load_model_path is None:
        # we use path as identifier
        load_model_path = model_identifier

    train_kraken_ocr_omr(
        save_model_path=Path(save_model_path),
        artifacts_path=Path(artifacts_path),
        train_json=Path(train_json),
        val_json=Path(val_json),
        test_json=Path(test_json) if test_json else None,
        output_path=output_dir,
        log_file=log_file,
        task=args.task,
        edition=args.edition,
        train_data_dir=args.data_dir or args.train_dir,
        test_data_dir=args.data_dir or args.test_dir,
        debug=args.debug,
        cuda=cuda_available,
        pretrained_model=load_model_path,
        augment=args.augment,
    )

    shutil.rmtree(artifacts_path)
