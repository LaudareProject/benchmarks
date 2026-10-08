#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != "--debug" ]]; then
  printf 'Usage: %s --debug\n' "$0" >&2
  exit 2
fi

FRAMEWORK="calamari"
TASK="ocr"
MODEL_NAME="default"
EDITION="diplomatic"
DATA_DIR="data/I-Ct_91"
EXPECTED_FOLDS=2
TEST_EXPERIMENT_ID="${LAUDARE_TEST_EXPERIMENT_ID:-test-sh-debug}"

START_TIME=$(date +%s)
echo "Running bounded two-fold Calamari OCR aggregation smoke test (${TEST_EXPERIMENT_ID})."
LAUDARE_EXPERIMENT_ID="$TEST_EXPERIMENT_ID" "$HOME/.local/bin/mise" exec uv -- ./run.sh \
  --framework "$FRAMEWORK" \
  --task "$TASK" \
  --model-name "$MODEL_NAME" \
  --edition "$EDITION" \
  --data-dir "$DATA_DIR" \
  --device cuda:1 \
  --n-fold \
  --debug

DATASET_NAME=$(basename "$DATA_DIR")
EXPERIMENT_ID="$TEST_EXPERIMENT_ID"
AGGREGATE_FILE="results/${EXPERIMENT_ID}/${DATASET_NAME}/${EDITION}/aggregated/${FRAMEWORK}_${TASK}_${MODEL_NAME}_evaluation.json"

echo "Checking aggregate fold coverage in ${AGGREGATE_FILE}."
if ! jq -e --argjson expected "$EXPECTED_FOLDS" '
  .status == "complete" and
  .expected_folds == $expected and
  .available_folds == [range(0; $expected)] and
  .missing_folds == [] and
  .missing_metrics == {} and
  (.metrics.WER.values | length == $expected) and
  (.metrics.CER.values | length == $expected)
' "$AGGREGATE_FILE" >/dev/null; then
  printf 'N-fold debug aggregation check failed for %s.\n' "$AGGREGATE_FILE" >&2
  exit 1
fi

END_TIME=$(date +%s)
printf 'Two-fold aggregation smoke test passed in %ss.\n' "$((END_TIME - START_TIME))"
