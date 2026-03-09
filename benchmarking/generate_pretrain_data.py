#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# Add annotations directory to path
from .annotations.data_synthesis import generate_dataset
from .annotations.ann_handler import create_all_pagexml_files


def main():
    parser = argparse.ArgumentParser(description="Generate pre-training data")
    parser.add_argument("--task", required=True, choices=["ocr", "omr"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--font-dir", help="Font directory for OCR")
    parser.add_argument("--capitals-dir", help="Capitals directory for OCR")
    parser.add_argument(
        "--lines-per-image",
        type=int,
        default=15,
        help="Number of text/music lines per image",
    )
    parser.add_argument(
        "--image-height", type=int, default=1280, help="Height of generated images"
    )
    parser.add_argument(
        "--latex-jobs",
        type=int,
        default=4,
        help="Number of parallel LaTeX jobs for OMR.",
    )

    args = parser.parse_args()

    if args.debug:
        args.num_samples = min(args.num_samples, 10)
        args.lines_per_image = min(args.lines_per_image, 5)  # Fewer lines in debug

    if args.task == "ocr":
        mode = "text"
        if not args.font_dir:
            raise ValueError("OCR pre-training requires --font-dir")
    elif args.task == "omr":
        mode = "music"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(
        f"   📝 Generating {args.num_samples} {args.task.upper()} pre-training samples..."
    )
    if args.task == "ocr":
        print(f"   📄 {args.lines_per_image} lines per image")
    else:
        print(f"   💻 Using {args.latex_jobs} parallel LaTeX jobs for OMR")
        print(f"   📄 {args.lines_per_image} staves per image")

    train_json_path, val_json_path = generate_dataset(
        output_dir=args.output_dir,
        font_dir=args.font_dir,
        capitals_dir=args.capitals_dir,
        mode=mode,
        task=args.task,
        num_samples=args.num_samples,
        width=1024,
        height=args.image_height,
        lines_per_image=args.lines_per_image,  # if args.task == "ocr" else 1,
        latex_jobs=args.latex_jobs,
        debug=args.debug,
    )

    create_all_pagexml_files(
        train_json_path.parent / "processed_splits",
        json.load(open(train_json_path)),
        "diplomatic",
        image_base_path=train_json_path.parent.parent,
        target_tasks=args.task,
    )

    create_all_pagexml_files(
        val_json_path.parent / "processed_splits",
        json.load(open(val_json_path)),
        "diplomatic",
        image_base_path=val_json_path.parent.parent,
        target_tasks=args.task,
    )

    print(f"   ✅ Pre-training data generated: {args.output_dir}")


if __name__ == "__main__":
    main()
