import argparse
from pathlib import Path


def find_best_model(directory: Path):
    """
    Finds the best model file in a given directory.
    Priority 1: .mlmodel, .ckpt, .pt files based on modification time.
    Priority 2: HuggingFace checkpoint-* directories based on step number.
    """
    if not directory.is_dir():
        return ""

    # Priority 1: Find file-based models and sort by modification time
    file_models = (
        list(directory.glob("**/*best.mlmodel"))
        + list(directory.glob("**/*.ckpt"))
        + list(directory.glob("**/*best.pt"))
        + list(directory.glob("**/*.pth"))
    )

    if file_models:
        latest_file_model = max(file_models, key=lambda p: p.stat().st_mtime)
        return str(latest_file_model)

    # Priority 2: Find directory-based models (huggingface checkpoints)
    try:
        dir_models = sorted(
            [p for p in directory.glob("**/checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if dir_models:
            return str(dir_models[-1])
    except (ValueError, IndexError):
        # Handles cases where checkpoint name is not as expected
        pass

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best model in an output directory."
    )
    parser.add_argument(
        "directory", type=Path, help="The directory to search for models."
    )
    args = parser.parse_args()

    best_model_path = find_best_model(args.directory)
    if best_model_path:
        print(best_model_path)
