#!/bin/env bash
# exit at first error
set -e

# Record the total start time
TOTAL_START_TIME=$(date +%s)

# Timing helpers
STEP_SUMMARY=""
STEP_START_TIME=0
STEP_NAME=""

step_start() {
  STEP_NAME="$1"
  STEP_START_TIME=$(date +%s)
}

step_end() {
  local STEP_END_TIME=$(date +%s)
  local ELAPSED=$((STEP_END_TIME - STEP_START_TIME))
  STEP_SUMMARY="$STEP_SUMMARY\n$(printf "%-30s %6ds" "$STEP_NAME" "$ELAPSED")"
  echo "Completed step: $STEP_NAME in ${ELAPSED}s" >>times.txt
}
#
# # testing frameworks
# step_start "kraken ocr"
# ./run.sh --framework kraken --task ocr --edition diplomatic --data-dir data/I-Ct_91 "$@"
# step_end

# step_start "calamari ocr"
# ./run.sh --framework calamari --task ocr --edition editorial --data-dir data/I-Ct_91 "$@"
# step_end

# step_start "trocr ocr"
# ./run.sh --framework trocr --task ocr --model-name "large" --edition editorial --data-dir data/I-Ct_91 "$@"
# step_end
#
# step_start "faster_rcnn layout"
# ./run.sh --framework faster_rcnn --task layout --model-name "resnet50" --edition diplomatic --data-dir data/I-Ct_91 "$@"
# step_end
#
# step_start "detr layout"
# ./run.sh --framework detr --task layout --edition diplomatic --data-dir data/I-Ct_91 "$@"
# step_end
#
# step_start "yolo layout"
# ./run.sh --framework yolo --task layout --model-name "yolov8n" --edition diplomatic --data-dir data/I-Ct_91 "$@"
# step_end

# # testing n-fold
# step_start "faster_rcnn layout n_fold"
# ./run.sh --framework faster_rcnn --task layout --model-name "resnet50" --edition diplomatic --data-dir data/I-Ct_91 --n-fold --debug
# step_end

# testing sequential strategies
step_start "sequential random_sample"
# ./run.sh --framework faster_rcnn --task layout --model-name "resnet50" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework faster_rcnn --task layout --model-name "mobilenet" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework yolo --task layout --model-name "yolov8s" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework yolo --task layout --model-name "yolov8n" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework detr --task layout --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
./run.sh --framework calamari --task ocr --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework kraken --task ocr --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework trocr --task ocr --model-name "large" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework trocr --task ocr --model-name "small" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework calamari --task omr --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework kraken --task omr --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework trocr --task omr --model-name "large" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
# ./run.sh --framework trocr --task omr --model-name "small" --edition diplomatic --data-dir data/I-Ct_91 --strategy random_sample --debug
step_end

# testing train-test mode
# ./run.sh --framework faster_rcnn --task layout --model-name "resnet50" --edition diplomatic --train-dir data/I-Ct_91 --test-dir data/I-Ct_91 --debug
# ./run.sh --framework detr --task layout --edition diplomatic --train-dir data/I-Ct_91 --test-dir data/I-Ct_91 --debug
# ./run.sh --framework yolo --task layout --model-name "yolov8s" --edition diplomatic --train-dir data/I-Ct_91 --test-dir data/I-Ct_91 --debug
# ./run.sh --framework trocr --task ocr --model-name "small" --edition diplomatic --train-dir data/I-Ct_91 --test-dir data/I-Ct_91 --debug
# ./run.sh --framework calamari --task ocr --edition diplomatic --train-dir data/I-Ct_91 --test-dir data/I-Ct_91 --debug

# testing synthesis
# ./run.sh --task synthesis --debug

# testing pretraining and pretrained model usage
# ./run.sh --framework trocr --task ocr --model-name large --edition diplomatic --data-dir data/I-Ct_91 --enable-pretrain --strategy random_sample --debug
# ./run.sh --framework trocr --task omr --model-name small --edition diplomatic --data-dir data/I-Fn_BR_18 --enable-pretrain --strategy random_sample --debug

# Print timing summary
TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo
printf "\n%-30s %6s\n" "Step" "Time (s)"
printf "%-30s %6s\n" "------------------------------" "------"
echo -e "$STEP_SUMMARY"
printf "%-30s %6ds\n" "Total" "$TOTAL_ELAPSED"
