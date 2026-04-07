#!/bin/bash

set -e

# Parse command line arguments
initial_argc=$#

DEBUG_MODE=false
DEVICE="cuda:0" # Default to GPU
TASK=""
EDITION=""
FRAMEWORK=""
DATA_DIR=""
TRAIN_DIR=""
TEST_DIR=""
SEQUENTIAL_STRATEGY=""
ENABLE_PRETRAIN=false
N_FOLD_MODE=false
SEQUENTIAL_MODE=false
OCR_PRETRAIN_SAMPLES=5000
OMR_PRETRAIN_SAMPLES=5000
FONT_DIR="./fonts"
CAPITALS_DIR="./fonts/capitals"
PRETRAIN_DIR=""
AUGMENT="true"
LATEX_JOBS=8
FOLD_NUM=0
MODEL_NAME="default"
while [[ $# -gt 0 ]]; do
  case $1 in
  --help)
    cat "$(dirname "$0")/run_help.txt"
    exit 0
    ;;
  --debug)
    DEBUG_MODE=true
    shift
    ;;
  --device)
    DEVICE="$2"
    shift
    shift
    ;;
  --framework)
    FRAMEWORK="$2"
    shift
    shift
    ;;
  --edition)
    EDITION="$2"
    case $EDITION in
    diplomatic | editorial) ;;
    *)
      echo "Invalid --edition option: $EDITION"
      echo "Valid options: diplomatic, editorial"
      exit 1
      ;;
    esac
    shift
    shift
    ;;
  --data-dir)
    DATA_DIR="$2"
    shift
    shift
    ;;
  --train-dir)
    TRAIN_DIR="$2"
    shift
    shift
    ;;
  --test-dir)
    TEST_DIR="$2"
    shift
    shift
    ;;
  --n-fold)
    N_FOLD_MODE=true
    shift
    ;;
  --strategy)
    SEQUENTIAL_STRATEGY="$2"
    SEQUENTIAL_MODE=true
    shift
    shift
    ;;
  --task)
    TASK="$2"
    case $TASK in
    annotations | ocr | omr | layout | synthesis) ;;
    *)
      echo "Invalid --task option: $TASK"
      echo "Valid options: annotations, ocr, omr, layout, ocmr"
      exit 1
      ;;
    esac
    shift
    shift
    ;;
  --enable-pretrain)
    ENABLE_PRETRAIN=true
    shift
    ;;
  --pretrain-samples)
    OMR_PRETRAIN_SAMPLES="$(($2 / 20))"
    OCR_PRETRAIN_SAMPLES="$2"
    shift
    shift
    ;;
  --font-dir)
    FONT_DIR="$2"
    shift
    shift
    ;;
  --capitals-dir)
    CAPITALS_DIR="$2"
    shift
    shift
    ;;
  --pretrain-dir)
    PRETRAIN_DIR="$2"
    shift
    shift
    ;;
  --latex-jobs)
    LATEX_JOBS="$2"
    shift
    shift
    ;;
  --fold)
    FOLD_NUM="$2"
    shift
    shift
    ;;
  --model-name)
    MODEL_NAME="$2"
    shift
    shift
    ;;
  --editorial)
    EDITION="editorial"
    shift
    ;;
  --augment)
    AUGMENT="$2"
    shift
    shift
    ;;
  *)
    echo "Unknown option $1"
    exit 1
    ;;
  esac
done

# Validate mandatory parameters
if [ "$initial_argc" -eq 0 ]; then
  echo "Error: No arguments provided. Use --help to see usage information."
  exit 1
fi

# Check mandatory parameters
missing_params=()

if [ -z "$TASK" ]; then
  missing_params+=("--task")
fi

# if task is not annotations nor synthesis
if [[ "$TASK" != "annotations" && "$TASK" != "synthesis" ]]; then
  if [ -z "$FRAMEWORK" ]; then
    missing_params+=("--framework")
  fi

  if [ -z "$EDITION" ]; then
    missing_params+=("--edition")
  fi
fi

if [ ${#missing_params[@]} -gt 0 ]; then
  echo "Error: Missing mandatory parameters: ${missing_params[*]}"
  echo "Use --help to see usage information."
  exit 1
fi

# Define Project Root and other key paths
# Assuming this script is in the project root
PROJECT_ROOT=$(pwd)
BENCHMARKING_DIR="benchmarking"

# Input COCO annotation files (adjust names if different)
# These should be the master COCO files for the entire dataset
COCO_MASTER="${DATA_DIR}/annotations-${EDITION}/gt.json"

# Output directories for processed data (folds, sequential splits)
# ann_handler.py will create subfolders like 'fold_00', 'train_test_0' inside these
OUTPUT_PROCESSED_BASE="${DATA_DIR}/annotations-${EDITION}/processed_splits"

# --- 1. Data Preparation ---
echo "=================="
echo "LAUDARE BENCHMARKS"
echo "=================="
echo ""
if [ -n "$LAUDARE_EXPERIMENT_ID" ]; then
  echo "🔬 Experiment ID: ${LAUDARE_EXPERIMENT_ID}"
fi
if [ "$DEBUG_MODE" = true ]; then
  echo "🐛 DEBUG MODE: Using reduced dataset"
  export LAUDARE_DEBUG=true
fi
# Set CUDA_VISIBLE_DEVICES based on the DEVICE variable
if [[ $DEVICE == cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES=${DEVICE#cuda:}
else
  export CUDA_VISIBLE_DEVICES="" # Set to empty to use CPU
fi
if [ -n "$DATA_DIR" ]; then
  echo "📊 Dataset: ${EDITION} (${DATA_DIR})"
else
  echo "📊 Train/Test Mode: ${TRAIN_DIR} -> ${TEST_DIR}"
fi
echo "🚀 Framework: ${FRAMEWORK}"
echo "🎯 Task: ${TASK}"
echo "Ⓜ️  Model name: ${MODEL_NAME}"

if [ "$N_FOLD_MODE" = true ]; then
  echo "🔄 N-Fold Mode Enabled"
fi
if [ "$SEQUENTIAL_MODE" = true ]; then
  echo "🔁 Sequential Mode Enabled (Strategy: ${SEQUENTIAL_STRATEGY})"
fi
if [ "$ENABLE_PRETRAIN" = true ]; then
  echo "📚 Pre-training enabled!"
fi
echo ""

# --- 1. Data Preparation ---
if [ "$TASK" = "annotations" ]; then
  # This condition ensures data prep runs for "annotations" step.
  echo "📊 Data Preparation"
  echo "──────────────────────────"

  # Create base output directories if they don't exist
  mkdir -p "${OUTPUT_PROCESSED_BASE}"

  # Run ann_handler.py to create folds for annotations.
  # This uses the default environment with only core dependencies.
  echo "⚙️  Processing ${EDITION} annotations..."
  if [ "$DEBUG_MODE" = true ]; then
    cmd="uv run python -m ${BENCHMARKING_DIR}.annotations.ann_handler \
    ${COCO_MASTER} \
    ${OUTPUT_PROCESSED_BASE} \
    --folds \
    --debug"
    echo "$cmd"
    $cmd
  else
    uv run python -m "${BENCHMARKING_DIR}.annotations.ann_handler" \
      "${COCO_MASTER}" \
      "${OUTPUT_PROCESSED_BASE}" \
      --folds
  fi
  echo "✅ ${EDITION} annotations processed"

  echo "⚙️  Processing Annotations for ALL Sequential Splits..."

  # Define all available strategies
  ALL_STRATEGIES=("random_sample" "sequential_sample")

  for strategy in "${ALL_STRATEGIES[@]}"; do
    echo "   - Strategy: ${strategy}"
    ann_args=(-m "${BENCHMARKING_DIR}.annotations.ann_handler" "${COCO_MASTER}" "${OUTPUT_PROCESSED_BASE}" --seq --strategy "${strategy}")
    if [ "$DEBUG_MODE" = true ]; then
      ann_args+=(--debug)
    fi
    uv run python "${ann_args[@]}"
  done
  echo "✅ Data preparation complete"
  echo ""
fi

# Generate pre-training data if the task is "synthesis"
if [ "$TASK" == "synthesis" ]; then
  echo "📚 STEP 1.5: Generating Pre-training Data"
  echo "─────────────────────────────────────────"

  PRETRAIN_DATA_DIR="${PROJECT_ROOT}/data/pretrain_data"
  mkdir -p "${PRETRAIN_DATA_DIR}"

  # Generate OMR synthetic data
  echo "🎵 Generating OMR synthetic data..."
  OMR_ARGS=(--task omr --output-dir "${PRETRAIN_DATA_DIR}/omr" --num-samples "$OMR_PRETRAIN_SAMPLES" --latex-jobs "$LATEX_JOBS")
  if [ "$DEBUG_MODE" = true ]; then
    OMR_ARGS+=(--debug)
  fi
  uv run python -m "${BENCHMARKING_DIR}.generate_pretrain_data" "${OMR_ARGS[@]}"

  echo "✅ Pre-training data generation complete"
  echo ""

  # Generate OCR synthetic data
  if [ -n "$FONT_DIR" ]; then
    echo "📝 Generating OCR synthetic data..."
    OCR_ARGS=(--task ocr --output-dir "${PRETRAIN_DATA_DIR}/ocr" --num-samples "$OCR_PRETRAIN_SAMPLES")
    if [ "$DEBUG_MODE" = true ]; then
      OCR_ARGS+=(--debug)
    fi
    OCR_ARGS+=(--font-dir "$FONT_DIR")
    if [ -n "$CAPITALS_DIR" ]; then
      OCR_ARGS+=(--capitals-dir "$CAPITALS_DIR")
    fi
    uv run python -m "${BENCHMARKING_DIR}.generate_pretrain_data" "${OCR_ARGS[@]}"
  else
    echo "⚠️  Skipping OCR synthetic data generation - no --font-dir provided"
  fi
fi

# Function to run benchmark for a specific fold
run_benchmarks_for_fold() {
  local fold_to_run=$1
  local model_name_to_run=$2
  shift 2
  local base_args=("$@")

  local -A python_exec_map=(
    ["kraken"]="-p ${PROJECT_ROOT}/.venv-kraken/bin/python"
    ["calamari"]="-p ${PROJECT_ROOT}/.venv-calamari/bin/python"
  )

  echo "🚀 Launching benchmark for ${FRAMEWORK}/${TASK} with model name ${model_name_to_run}..."
  local run_args=("${base_args[@]}" --fold "$fold_to_run" --framework "$FRAMEWORK" --task "$TASK" --model-name "$model_name_to_run")

  local exec_args=${python_exec_map[$FRAMEWORK]}
  command="uv run $exec_args python -m ${BENCHMARKING_DIR}.run_single_fold_benchmark ${run_args[@]}"
  echo "Running command: $command"
  $command
}

# --- 2. Run Benchmarks ---
if [[ "$TASK" =~ ^(ocr|omr|layout|ocmr)$ ]]; then
  # Build base arguments for run_single_fold_benchmark.py
  benchmark_args=(--edition "${EDITION}")
  if [ "$AUGMENT" = "true" ]; then benchmark_args+=(--augment); fi
  if [ "$DEBUG_MODE" = true ]; then benchmark_args+=(--debug); fi
  if [ -n "$TASK" ]; then benchmark_args+=(--task "$TASK"); fi
  if [ "$ENABLE_PRETRAIN" = true ]; then benchmark_args+=(--enable-pretrain); fi
  if [ -n "$PRETRAIN_DIR" ]; then benchmark_args+=(--pretrain-dir "$PRETRAIN_DIR"); fi

  if [ -n "$DATA_DIR" ]; then
    benchmark_args+=(--data-dir "$DATA_DIR")
  fi

  if [ "$N_FOLD_MODE" = true ]; then
    # N-FOLD CROSS-VALIDATION MODE
    echo "🚀 Running N-Fold Benchmark"
    echo "──────────────────────────────────"

    NUM_FOLDS=5
    if [ "$DEBUG_MODE" = true ]; then
      NUM_FOLDS=2 # Run only 2 folds in debug mode to speed up testing
      echo "🐛 DEBUG: Running for ${NUM_FOLDS} folds only."
    fi

    for i in $(seq 0 $((NUM_FOLDS - 1))); do
      run_benchmarks_for_fold "$i" "$MODEL_NAME" "${benchmark_args[@]}"
    done

    echo ""
    echo "📊 Aggregating Results"
    echo "──────────────────────────────"

    # Define output file for aggregated results
    RESULTS_DIR="${PROJECT_ROOT}/results"
    DATA_DIR_NAME=$(basename "$DATA_DIR")
    EXP_ID="${LAUDARE_EXPERIMENT_ID:-default}"
    AGG_OUTPUT_DIR="${RESULTS_DIR}/${EXP_ID}/${DATA_DIR_NAME}/${EDITION}/aggregated"
    mkdir -p "$AGG_OUTPUT_DIR"
    AGG_OUTPUT_FILE="${AGG_OUTPUT_DIR}/${FRAMEWORK}_${TASK}_${MODEL_NAME}_evaluation.json"

    # Build analysis command
    analyze_args=(
      --edition "$EDITION" --framework "$FRAMEWORK" --task "$TASK"
      --num-folds "$NUM_FOLDS" --model-name "$MODEL_NAME" --data-dir "$DATA_DIR"
    )

    # Save to a JSON file
    analyze_args+=(--output-file "$AGG_OUTPUT_FILE")

    # The python script will handle file naming and saving logic.
    uv run python "${BENCHMARKING_DIR}/analyze_results.py" "${analyze_args[@]}"

  fi

  # SEQUENTIAL LEARNING MODE
  if [ "$SEQUENTIAL_MODE" = true ]; then
    echo "🚀 Running Sequential Benchmark"
    echo "─────────────────────────────────────"

    # Use the specified strategy (always set since it's mandatory)
    current_strategy="$SEQUENTIAL_STRATEGY"
    echo "  === Running Sequential Benchmark for Strategy: ${current_strategy} ==="

    SEQ_STRATEGY_DIR="${OUTPUT_PROCESSED_BASE}/${current_strategy}"
    SEQ_DIRS=($(find "$SEQ_STRATEGY_DIR" -maxdepth 1 -type d -name "seq_*" | sort))
    if [ ${#SEQ_DIRS[@]} -eq 0 ]; then
      echo "❌ Error: No sequential directories (seq_*) found in ${SEQ_STRATEGY_DIR}"
      exit 1
    fi

    NUM_STEPS=${#SEQ_DIRS[@]}
    if [ "$DEBUG_MODE" = true ]; then
      NUM_STEPS=2 # Run only 2 steps in debug mode to speed up testing
      echo "🐛 DEBUG: Running for ${NUM_STEPS} steps only."
    fi

    for i in $(seq 0 $((NUM_STEPS - 1))); do
      echo "    --- Running Sequential Step $((i + 1))/$NUM_STEPS for ${current_strategy} ---"

      declare -A python_exec_map=(
        ["kraken"]="-p ${PROJECT_ROOT}/.venv-kraken/bin/python"
        ["calamari"]="-p ${PROJECT_ROOT}/.venv-calamari/bin/python"
      )

      # Use the specified model name for sequential mode
      seq_model_name="$MODEL_NAME"
      echo "    🚀 Launching benchmark for ${FRAMEWORK}/${TASK} with model name ${seq_model_name}..."
      seq_run_args=("${benchmark_args[@]}" "--framework" "$FRAMEWORK" --task "$TASK" "--sequential-step" "$i" "--sequential-strategy" "$current_strategy")
      seq_run_args+=("--model-name" "$seq_model_name")

      exec_args=${python_exec_map[$FRAMEWORK]}
      uv run $exec_args python -m "${BENCHMARKING_DIR}.run_single_fold_benchmark" "${seq_run_args[@]}"

    done

    echo "  ✅ Completed Sequential Benchmark for Strategy: ${current_strategy}"
    # Aggregation step would go here
  fi

  if [ "$N_FOLD_MODE" = false ] && [ "$SEQUENTIAL_MODE" = false ]; then
    if [ -n "$TRAIN_DIR" ] && [ -n "$TEST_DIR" ]; then
      # TRAIN/TEST DIRECTORY MODE
      echo "🚀 Running Train/Test Benchmark"
      echo "─────────────────────────────────────"
      echo "📂 Training on: ${TRAIN_DIR}"
      echo "🧪 Testing on: ${TEST_DIR}"

      declare -A python_exec_map=(
        ["kraken"]="-p ${PROJECT_ROOT}/.venv-kraken/bin/python"
        ["calamari"]="-p ${PROJECT_ROOT}/.venv-calamari/bin/python"
      )
      train_test_args=("${benchmark_args[@]}" --framework "$FRAMEWORK" --task "$TASK" --model-name "$MODEL_NAME" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR")
      exec_args=${python_exec_map[$FRAMEWORK]}
      uv run $exec_args python -m "${BENCHMARKING_DIR}.run_single_fold_benchmark" "${train_test_args[@]}"
    else
      # SINGLE FOLD MODE
      echo "🚀 Running Benchmark for Single Fold ${FOLD_NUM}"
      echo "─────────────────────────────────────────────────"
      run_benchmarks_for_fold "$FOLD_NUM" "$MODEL_NAME" "${benchmark_args[@]}"
    fi
  fi
fi

echo ""
echo "🧽 CLEANING UP LARGE FILES"
if [ -d "${PROJECT_ROOT}/results" ]; then
  for file in $(find "${PROJECT_ROOT}/results" -type f ! -name "*_evaluation.json" -size +1M); do
    printf '%s\n' "PLACEHOLDER: original file was larger than 1MB" > "$file"
  done
fi
echo "🎉 RUN COMPLETE"
