import argparse
import copy
import json
import sys
from pathlib import Path

from .evaluation import evaluate_predictions_entry, extract_layout_from_pagexml
from .run_single_fold_benchmark import load_and_run_framework, setup_pretrain_training
from .utils import RESULTS_DIR, get_pretrained_model_path, setup_with_args

PROJECT_ROOT = Path(__file__).resolve().parent.parent.absolute()

_TEXT_TASKS = ("ocr", "omr")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _pipeline_pred_dir(setup_dict: dict, pipeline_id: str) -> dict:
    """Return a copy of setup_dict with output_dir rerouted to a pipeline-specific subdir.

    save_model_path is intentionally kept as-is so trained weights are shared
    with (and reusable by) standalone runs of the same framework+task+model.
    """
    old = Path(setup_dict["output_dir"])
    run_dir = old.parent  # .../fw_task_model
    new_run_dir = run_dir.parent / f"pipeline_{pipeline_id}__{run_dir.name}"
    new_pred_dir = new_run_dir / "predictions"
    new_pred_dir.mkdir(parents=True, exist_ok=True)
    return {**setup_dict, "output_dir": new_pred_dir}


# ---------------------------------------------------------------------------
# PageXML → COCO conversion
# ---------------------------------------------------------------------------


def _pagexml_preds_to_coco(layout_preds_dir: Path, reference_json: Path) -> dict:
    """Convert layout PageXML predictions to a COCO JSON for OCR/OMR input.

    Bboxes come from predicted regions; text fields are left empty (not used
    during prediction, only ground-truth JSON is used for evaluation).
    """
    with open(reference_json) as f:
        ref = json.load(f)

    stem_to_image = {Path(img["file_name"]).stem: img for img in ref["images"]}
    categories = ref.get("categories", [{"id": 1, "name": "line", "supercategory": "layout"}])
    default_cat_id = categories[0]["id"] if categories else 1

    annotations: list[dict] = []
    ann_id = 1

    for xml_file in sorted(layout_preds_dir.glob("*.xml")):
        stem = xml_file.stem
        if stem not in stem_to_image:
            continue
        image_id = stem_to_image[stem]["id"]

        regions = extract_layout_from_pagexml(xml_file)
        if not regions:
            continue

        # Prefer TextLine-level crops; fall back to all regions when absent
        lines = [r for r in regions if r.get("type") in ("TextLine", "line")]
        crop_regions = lines if lines else regions

        for region in crop_regions:
            bbox = region["bbox"]  # [x, y, w, h]
            w, h = bbox[2], bbox[3]
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": default_cat_id,
                    "bbox": bbox,
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                    "description": "",
                    "text": "",
                }
            )
            ann_id += 1

    return {"images": ref["images"], "categories": categories, "annotations": annotations}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _resolve_tasks(requested_task: str | None, available: tuple = _TEXT_TASKS) -> list[str]:
    if requested_task is None:
        return list(available)
    if requested_task in available:
        return [requested_task]
    return []


def _make_stage_args(base_args, framework: str, task: str, model_name: str):
    a = copy.copy(base_args)
    a.framework = framework
    a.task = task
    a.model_name = model_name
    return a


def _run_pretrain_if_needed(args, framework: str, task: str, model_name: str):
    if not getattr(args, "enable_pretrain", False):
        return
    pretrained_path = get_pretrained_model_path(
        framework, task, model_name, getattr(args, "pretrain_dir", None)
    )
    if pretrained_path.exists():
        print(f"   ✅ Existing pretrained model: {pretrained_path}")
        return
    print("   🏗️  No pretrained model found, training on synthetic data…")
    stage_args = _make_stage_args(args, framework, task, model_name)
    pretrain_setup = setup_pretrain_training(stage_args, task, pretrained_path)
    load_and_run_framework(framework, pretrain_setup)
    print(f"   ✅ Pre-training complete: {pretrain_setup['save_model_path']}")


# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------


def run_pipeline(args, layout_fw: str, text_fw: str, layout_model: str, text_model: str):
    pipeline_id = f"{layout_fw}+{text_fw}"

    # ── Stage 1: layout ───────────────────────────────────────────────────
    print(f"\n🏗️  Stage 1 — layout  [{layout_fw}  model={layout_model}]")
    _run_pretrain_if_needed(args, layout_fw, "layout", layout_model)
    layout_args = _make_stage_args(args, layout_fw, "layout", layout_model)
    layout_setup = setup_with_args(layout_args)
    load_and_run_framework(layout_fw, layout_setup)

    layout_preds_dir: Path = layout_setup["output_dir"]
    layout_test_json: Path | None = layout_setup["test_json"]

    if layout_test_json is not None and layout_preds_dir is not None:
        eval_file = layout_preds_dir.parent / "layout_evaluation.json"
        results = evaluate_predictions_entry(
            predictions_dir=str(layout_preds_dir),
            ground_truth_json=str(layout_test_json),
            task="layout",
            output_file=str(eval_file),
            debug=args.debug,
        )
        print(f"   📊 Layout evaluation → {eval_file}")
    else:
        print("   ⏭️  Skipping layout evaluation (no test data)")

    # ── Stages 2+3: OCR / OMR on predicted regions ───────────────────────
    tasks = _resolve_tasks(args.task, _TEXT_TASKS)
    if not tasks:
        # user passed --task layout only; nothing more to do
        return

    for task in tasks:
        print(f"\n🔤 Stage 2/{task} — build pipeline test JSON  [{task.upper()}]")
        _run_pretrain_if_needed(args, text_fw, task, text_model)
        text_args = _make_stage_args(args, text_fw, task, text_model)
        text_setup = setup_with_args(text_args)

        gt_test_json: Path | None = text_setup["test_json"]
        if gt_test_json is None or not gt_test_json.exists():
            print(f"   ⏭️  No ground-truth test JSON for {task}, skipping")
            continue

        # Build COCO JSON from layout predictions
        pipeline_coco = _pagexml_preds_to_coco(layout_preds_dir, gt_test_json)
        n_anns = len(pipeline_coco["annotations"])
        pipeline_test_json = layout_preds_dir.parent / f"pipeline_{task}_test.json"
        with open(pipeline_test_json, "w") as f:
            json.dump(pipeline_coco, f)
        print(f"   📄 {pipeline_test_json}  ({n_anns} predicted regions)")

        # Override: test on predicted crops, train/val unchanged
        text_setup["test_json"] = pipeline_test_json
        text_setup = _pipeline_pred_dir(text_setup, pipeline_id)

        print(f"\n🔤 Stage 3/{task} — [{text_fw}  model={text_model}]")
        load_and_run_framework(text_fw, text_setup)

        pred_dir = text_setup["output_dir"]
        if pred_dir is not None:
            eval_file = pred_dir.parent / f"{task}_evaluation.json"
            evaluate_predictions_entry(
                predictions_dir=str(pred_dir),
                ground_truth_json=str(gt_test_json),
                task=task,
                output_file=str(eval_file),
                debug=args.debug,
            )
            print(f"   📊 {task.upper()} evaluation → {eval_file}")
        else:
            print(f"   ⏭️  Skipping {task.upper()} evaluation (no output directory)")


# ---------------------------------------------------------------------------
# E2E mode
# ---------------------------------------------------------------------------


def run_e2e(args, framework: str, model_name: str):
    tasks = _resolve_tasks(args.task, _TEXT_TASKS)
    if not tasks:
        print(f"❌ --task must be one of {_TEXT_TASKS} in e2e mode")
        sys.exit(1)

    for task in tasks:
        print(f"\n🔤 E2E — {framework}  task={task}  model={model_name}")
        _run_pretrain_if_needed(args, framework, task, model_name)
        stage_args = _make_stage_args(args, framework, task, model_name)
        setup = setup_with_args(stage_args)
        load_and_run_framework(framework, setup)

        if setup["test_json"] is not None and setup["output_dir"] is not None:
            eval_file = Path(setup["output_dir"]).parent / f"{task}_evaluation.json"
            evaluate_predictions_entry(
                predictions_dir=str(setup["output_dir"]),
                ground_truth_json=str(setup["test_json"]),
                task=task,
                output_file=str(eval_file),
                debug=args.debug,
            )
            print(f"   📊 {task.upper()} evaluation → {eval_file}")
        else:
            print(f"   ⏭️  Skipping {task.upper()} evaluation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full-pipeline benchmark.\n"
            "  Pipeline mode:  --framework yolo+trocr  --model-name yolov8n+large\n"
            "  E2E mode:       --framework paddleocr_vl_e2e  --model-name default"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--framework",
        required=True,
        help="'layout_fw+text_fw' for pipeline mode, or 'e2e_fw' for end-to-end.",
    )
    p.add_argument(
        "--model-name",
        default="default",
        help="'layout_model+text_model' (text_model optional, defaults to 'default').",
    )
    p.add_argument(
        "--edition",
        default="diplomatic",
        choices=["diplomatic", "editorial"],
    )
    p.add_argument(
        "--task",
        default=None,
        choices=["ocr", "omr", "layout"],
        help="Restrict to a single task. Omit to run all applicable tasks.",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--enable-pretrain", action="store_true")
    p.add_argument("--sequential-step", type=int)
    p.add_argument("--sequential-strategy", type=str, default="cumulative")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--train-dir", type=Path)
    p.add_argument("--test-dir", type=Path)
    p.add_argument("--pretrain-dir", type=Path)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.data_dir is None and (args.train_dir is None or args.test_dir is None):
        parser.error("Specify --data-dir or both --train-dir and --test-dir.")
    if bool(args.train_dir) != bool(args.test_dir):
        parser.error("--train-dir and --test-dir must be specified together.")

    fw_parts = args.framework.split("+", 1)
    model_parts = args.model_name.split("+", 1)

    if len(fw_parts) == 2:
        layout_fw, text_fw = fw_parts
        layout_model = model_parts[0]
        text_model = model_parts[1] if len(model_parts) == 2 else "default"
        print(f"🔬 Pipeline mode: {layout_fw}+{text_fw}  models={layout_model}+{text_model}")
        run_pipeline(args, layout_fw, text_fw, layout_model, text_model)
    else:
        framework = fw_parts[0]
        model_name = model_parts[0]
        print(f"🔬 E2E mode: {framework}  model={model_name}")
        run_e2e(args, framework, model_name)

    print("\n✅ Pipeline benchmark complete")


if __name__ == "__main__":
    main()
