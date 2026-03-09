#!/bin/bash

# experiments.sh - Run all combinations of frameworks, tasks, model indices, sequential strategies, and editions
# This script runs the benchmark workflow for every possible combination without loops

# Use this like:
# ./experiments.sh data/I-Ct_91
# ./experiments.sh data/I-Fn_BR_18 --debug
# ./experiments.sh data/I-Ct_91 --device cuda:1
# ./experiments.sh data/I-Fn_BR_18 --debug --device cpu

#===================
# Argument parsing
#===================

# Initialize variables
DATA_DIR=""
DEBUG_FLAG=""
DEVICE_FLAG=""
ALLOWED_DATA_DIRS=("data/I-Ct_91" "data/I-Fn_BR_18")

# Function to show usage
show_usage() {
    echo "Usage: $0 <data_directory> [--debug] [--device <device>]"
    echo "  data_directory: Must be one of: ${ALLOWED_DATA_DIRS[*]}"
    echo "  --debug:        Enable debug mode (optional)"
    echo "  --device:       Specify device (e.g., cuda:0, cpu) (optional)"
    exit 1
}

# Function to validate data directory
validate_data_dir() {
    local dir="$1"
    for allowed_dir in "${ALLOWED_DATA_DIRS[@]}"; do
        if [ "$dir" = "$allowed_dir" ]; then
            return 0
        fi
    done
    return 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        --device)
            if [ -z "$2" ]; then
                echo "Error: --device requires a value"
                show_usage
            fi
            DEVICE_FLAG="--device $2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_usage
            ;;
        *)
            if [ -z "$DATA_DIR" ]; then
                DATA_DIR="$1"
            else
                echo "Error: Multiple data directories specified"
                show_usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ]; then
    echo "Error: Data directory is required"
    show_usage
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

if ! validate_data_dir "$DATA_DIR"; then
    echo "Error: Data directory must be one of: ${ALLOWED_DATA_DIRS[*]}"
    exit 1
fi

# Build the common arguments string
COMMON_ARGS="$DEBUG_FLAG $DEVICE_FLAG"

#@@@@@@@@@@@@@@@
#@ Annotations @
#@@@@@@@@@@@@@@@
# These are the commands used to generate annotations and pre-trained data
# ./run.sh --task annotations --edition diplomatic --data-dir data/I-Ct_91
# ./run.sh --task annotations --edition editorial --data-dir data/I-Ct_91
# ./run.sh --task annotations --edition diplomatic --data-dir data/I-Fn_BR_18
# ./run.sh --task annotations --edition editorial --data-dir data/I-Fn_BR_18

#@@@@@@@@@@
#@ 5-fold @
#@@@@@@@@@@


./run.sh --framework yolo --task layout --model-name "yolov8n" --edition diplomatic --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework yolo --task layout --model-name "yolov8n" --edition editorial --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS

./run.sh --framework trocr --task ocr --model-name "large" --edition diplomatic --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task omr --model-name "large" --edition diplomatic --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task ocr --model-name "large" --edition editorial --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task omr --model-name "large" --edition editorial --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS

#@@@@@@@@@@@@@@
#@ Sequential @
#@@@@@@@@@@@@@@

./run.sh --framework yolo --task layout --model-name "yolov8n" --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework yolo --task layout --model-name "yolov8n" --edition editorial --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS

./run.sh --framework trocr --task ocr --model-name "large" --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task omr --model-name "large" --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task ocr --model-name "large" --edition editorial --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS
./run.sh --framework trocr --task omr --model-name "large" --edition editorial --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS

#@@@@@@@@@@@@@@
#@ train-test @
#@@@@@@@@@@@@@@

# Determine train and test directories based on data directory
if [ "$DATA_DIR" == "data/I-Ct_91" ]; then
  train_dir="data/I-Ct_91"
  # test_dir="data/I-Ct_91"
  test_dir="data/I-Fn_BR_18"
else
  train_dir="data/I-Fn_BR_18"
  # train_dir="data/I-Ct_91"
  test_dir="data/I-Ct_91"
fi

./run.sh --framework trocr --model-name "large" --task ocr --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS
./run.sh --framework trocr --model-name "large" --task omr --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS
./run.sh --framework yolo --task layout --model-name "yolov8n" --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS

./run.sh --framework yolo --model-name "yolov8n" --task layout --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS
./run.sh --framework trocr --model-name "large" --task ocr --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS
./run.sh --framework trocr --model-name "large" --task omr --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS
