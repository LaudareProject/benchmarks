# Custom Model and Dataset Integration Guide

This guide covers custom datasets and model/framework integration. For a custom
dataset, provide COCO annotations under `data/<dataset>/annotations-<edition>/`,
prepare them with `run.sh --task annotations`, then run a benchmark. The sections
below give the exact layout, annotation rules, and commands.

## Pipeline Benchmarking

### Overview

End-to-end pipeline evaluation: layout detection → crop extraction → OCR/OMR recognition.

### Usage

```bash
# Pipeline: layout framework + text framework (both required)
uv run python3 -m benchmarking.run_pipeline_benchmark \
  --framework yolo+trocr \
  --model-name yolov8n+large \
  --data-dir data/I-Ct_91 \
  --fold 0

# E2E: single framework, text tasks only (no layout evaluation)
uv run python3 -m benchmarking.run_pipeline_benchmark \
  --framework paddleocr_vl_e2e \
  --data-dir data/I-Ct_91
```

### Architecture

**Pipeline (2-fw mode)**: `--framework fw1+fw2 --model-name m1+m2` (m2 optional → default)
1. Stage 1: Layout (`fw1`) trains on GT, predicts regions → PageXML
2. Convert PageXML → COCO JSON (predicted bboxes per task)
3. Stage 2a: OCR (`fw2`) trains on GT crops, tests on predicted crops
4. Stage 2b: OMR (`fw2`) trains on GT crops, tests on predicted crops
5. Evaluate: layout (Stage 1 PageXML vs GT), OCR/OMR (Stage 2 text vs GT)

**E2E (1-fw mode)**: `--framework fw_e2e`—no layout, OCR/OMR only.

### Key details

- **Isolation**: Pipeline predictions go to `pipeline_{fw1}+{fw2}__{fw_task_model}/predictions/` (separate from standalone runs)
- **Weights reuse**: `save_model_path` shared; trained weights are cached across pipeline/standalone
- **Ground-truth matching**: OCR/OMR train on GT, test on predicted regions; evaluation still uses original GT JSON for metrics
- **Region selection**: TextLine elements preferred from PageXML; falls back to all regions if absent
- **Empty results**: handled gracefully (zero annotations → test skip)

### Flags

- `--task layout`/`ocr`/`omr`: restrict to single stage (omit for all applicable)
- `--enable-pretrain`: pre-train on synthetic data before fine-tuning
- `--sequential-step`: sequential learning within pipeline
- All other flags from `run_single_fold_benchmark.py` supported

### Output

Each stage emits:
- `{stage}_evaluation.json`: metrics (AP for layout, WER/CER for text)
- `predictions/`: PageXML (layout) or `.pred.txt` (text)

## Add a Custom Dataset

Put images under `data/<dataset>/` and the master COCO file at
`data/<dataset>/annotations-<edition>/gt.json`, where edition is `diplomatic` or
`editorial`. Prepare annotations and run a benchmark:

```bash
./run.sh --task annotations --data-dir data/my_dataset --edition diplomatic
./run.sh --framework kraken --model-name default --task ocr \
  --data-dir data/my_dataset --edition diplomatic --fold 0
```

Annotation preparation creates five folds, both sequential strategies, and
`train.json` / `val.json` beside `gt.json`. It performs real processing; there is
no dry-run mode. The details below describe the expected data and category IDs.

### Directory and COCO Format

```text
data/my_dataset/
├── images/
│   └── page_001.png
└── annotations-diplomatic/
    └── gt.json
```

In each COCO image entry, `file_name` is relative to the dataset root, such as
`images/page_001.png`. Include `images`, `annotations`, and `categories` arrays.
Each annotation needs an `image_id` matching an image and a positive
`[x, y, width, height]` bounding box. Text transcription can be stored as `text`
or `description`; PageXML generation copies `description` to `text` when `text`
is absent.

### Category IDs and Task Filtering

The annotation and OCR/OMR loading code uses fixed category IDs; these are not
configurable per dataset:

- OCR uses category ID `6`.
- OMR uses category ID `5` for generated PageXML. OMR split JSON also includes
  IDs `1`, `2`, `3`, and `9`.
- OCMR split JSON uses IDs `5` and `6`, but `run.sh` does not accept `ocmr` as a
  valid `--task`.
- Layout split JSON includes categories whose `supercategory` is `"layout"`.
  Generated layout PageXML handles IDs `4` and `7` as regions, and IDs `5` and
  `6` as lines; other IDs are not represented by that PageXML conversion.

Use category definitions and IDs that match the task and the framework’s input
requirements. A COCO layout split accepting other layout IDs does not mean the
PageXML conversion or every model supports those IDs.

For example, this is an OCR line category:

```json
{
  "id": 6,
  "name": "line",
  "supercategory": "layout"
}
```

### Prepare and Run

`--data-dir` is the dataset root; `--edition` selects the
`annotations-<edition>` directory. The same `--data-dir` and `--edition` must be
used when preparing annotations and running the benchmark. Use `./run.sh --help`
and `benchmarking/models.json` to choose a supported framework and model.

The annotation handler infers the image root from the location of `gt.json`.
Images referenced by `file_name` must exist beneath that root.

### Generated Files

Fold JSON files, sequential split files, and generated PageXML are written under
`data/<dataset>/annotations-<edition>/processed_splits/`. Running the handler on
`gt.json` also writes `train.json` and `val.json` next to it. Models and results
are stored under `models/${LAUDARE_EXPERIMENT_ID:-default}/` and
`results/${LAUDARE_EXPERIMENT_ID:-default}/`; benchmark results are further
organized by dataset and edition.

`--debug` does not reduce the five-fold split: `split_into_folds` currently ignores
its debug argument. Sequential split processing does honor `--debug` and limits
that processing to the first 15 images. Neither mode is a full-dataset validation.

### Troubleshooting

- Missing images: verify each `file_name` relative to `data/<dataset>/`.
- Empty task splits: verify annotations use the category IDs listed above.
- Missing PageXML transcription: provide `text` or `description` on OCR
  annotations.
- Missing processed files: confirm `gt.json` exists at the selected edition
  path and rerun annotation preparation.

