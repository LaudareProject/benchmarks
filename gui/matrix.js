/*
 * matrix.js - experiment matrix configuration and pure script generation.
 *
 * Works both as a plain browser script (defines the global `LaudareMatrix`)
 * and as a Node CommonJS module (`require("./matrix.js")`).
 * This file MUST stay free of DOM access.
 */
'use strict';

const TASK_GROUPS = [
  { task: 'layout', label: 'Layout' },
  { task: 'ocr', label: 'OCR' },
  { task: 'omr', label: 'OMR' }
];

/* Rows: framework/task/model combinations, in display order. */
const ROWS = [
  { id: 'detr_layout', task: 'layout', framework: 'detr', model: null, label: 'DETR · Layout' },
  { id: 'faster_rcnn_mobilenet_layout', task: 'layout', framework: 'faster_rcnn', model: 'mobilenet', label: 'Faster R-CNN (MobileNet) · Layout' },
  { id: 'faster_rcnn_resnet50_layout', task: 'layout', framework: 'faster_rcnn', model: 'resnet50', label: 'Faster R-CNN (ResNet50) · Layout' },
  { id: 'yolo_yolo26m_layout', task: 'layout', framework: 'yolo', model: 'yolo26m', label: 'YOLO26m · Layout' },
  { id: 'yolo_yolo26s_layout', task: 'layout', framework: 'yolo', model: 'yolo26s', label: 'YOLO26s · Layout' },
  { id: 'doclayout_yolo_layout', task: 'layout', framework: 'doclayout_yolo', model: null, label: 'DocLayout-YOLO · Layout' },

  { id: 'kraken_ocr', task: 'ocr', framework: 'kraken', model: null, label: 'Kraken · OCR' },
  { id: 'calamari_ocr', task: 'ocr', framework: 'calamari', model: null, label: 'Calamari · OCR' },
  { id: 'trocr_small_ocr', task: 'ocr', framework: 'trocr', model: 'small', label: 'TrOCR (small) · OCR' },
  { id: 'trocr_large_ocr', task: 'ocr', framework: 'trocr', model: 'large', label: 'TrOCR (large) · OCR' },
  { id: 'paddleocr_vl_ocr', task: 'ocr', framework: 'paddleocr_vl', model: null, label: 'PaddleOCR-VL · OCR' },
  { id: 'vlt_ocr', task: 'ocr', framework: 'vlt', model: null, label: 'VLT · OCR' },

  { id: 'kraken_omr', task: 'omr', framework: 'kraken', model: null, label: 'Kraken · OMR' },
  { id: 'calamari_omr', task: 'omr', framework: 'calamari', model: null, label: 'Calamari · OMR' },
  { id: 'trocr_small_omr', task: 'omr', framework: 'trocr', model: 'small', label: 'TrOCR (small) · OMR' },
  { id: 'trocr_large_omr', task: 'omr', framework: 'trocr', model: 'large', label: 'TrOCR (large) · OMR' },
  { id: 'paddleocr_vl_omr', task: 'omr', framework: 'paddleocr_vl', model: null, label: 'PaddleOCR-VL · OMR' },
  { id: 'vlt_omr', task: 'omr', framework: 'vlt', model: null, label: 'VLT · OMR' },
  { id: 'bgk_omr', task: 'omr', framework: 'bgk', model: null, label: 'BGK · OMR' }
];

const DISABLED_REASONS = {
  syntheticNoLayoutData: 'Synthetic pretraining has no layout data: data/pretrain_data contains only ocr/omr.',
  catmusOcrOnly: 'CATMuS is an OCR-only pretraining corpus (no music or layout annotations).'
};

/*
 * Columns: A..I. `args` holds the column-specific run.sh arguments in emission
 * order (edition first, then mode flags, matching experiments.sh conventions).
 * `dataDir` decides how the dataset argument is emitted:
 *   'standard'    -> append  --data-dir "$DATA_DIR"
 *   'train-test'  -> no --data-dir (column uses --train-dir/--test-dir)
 *   'embedded'    -> --data-dir already present in `args`, nothing appended
 */
const COLUMNS = [
  {
    id: 'fold_diplomatic',
    letter: 'A',
    label: '5-fold · Diplomatic',
    group: '5-fold',
    args: ['--edition', 'diplomatic', '--n-fold'],
    dataDir: 'standard',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'fold_editorial',
    letter: 'B',
    label: '5-fold · Editorial',
    group: '5-fold',
    args: ['--edition', 'editorial', '--n-fold'],
    dataDir: 'standard',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'sequential_diplomatic',
    letter: 'C',
    label: 'Sequential · Diplomatic',
    group: 'Sequential',
    args: ['--edition', 'diplomatic', '--strategy', 'random_sample'],
    dataDir: 'standard',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'sequential_editorial',
    letter: 'D',
    label: 'Sequential · Editorial',
    group: 'Sequential',
    args: ['--edition', 'editorial', '--strategy', 'random_sample'],
    dataDir: 'standard',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'train_test_cross_manuscript',
    letter: 'E',
    label: 'Train/test · Cross-manuscript',
    group: 'Train/test',
    args: ['--edition', 'diplomatic', '--train-dir', '"$train_dir"', '--test-dir', '"$test_dir"'],
    dataDir: 'train-test',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'pretrain_synthetic_sequential',
    letter: 'F',
    label: 'Pretrain synthetic → sequential',
    group: 'Pretrain → sequential',
    args: ['--edition', 'diplomatic', '--strategy', 'random_sample', '--enable-pretrain'],
    dataDir: 'standard',
    disabledForTasks: ['layout'],
    enabledTasks: null,
    disabledReason: DISABLED_REASONS.syntheticNoLayoutData
  },
  {
    id: 'pretrain_catmus_sequential',
    letter: 'G',
    label: 'Pretrain CATMuS → sequential',
    group: 'Pretrain → sequential',
    args: ['--edition', 'diplomatic', '--pretrain-dir', 'data/CATMuS_medieval', '--strategy', 'random_sample', '--enable-pretrain'],
    dataDir: 'standard',
    disabledForTasks: [],
    enabledTasks: ['ocr'],
    disabledReason: DISABLED_REASONS.catmusOcrOnly
  },
  {
    id: 'pretrain_train_test',
    letter: 'H',
    label: 'Pretrain train manuscript → train/test',
    group: 'Pretrain → train/test',
    args: ['--edition', 'diplomatic', '--pretrain-dir', '"$train_dir"', '--data-dir', '"$test_dir"', '--strategy', 'random_sample', '--enable-pretrain'],
    dataDir: 'embedded',
    disabledForTasks: [],
    enabledTasks: null,
    disabledReason: null
  },
  {
    id: 'pretrain_catmus_train_test',
    letter: 'I',
    label: 'Pretrain CATMuS → train/test',
    group: 'Pretrain → train/test',
    args: ['--edition', 'diplomatic', '--pretrain-dir', 'data/CATMuS_medieval', '--data-dir', '"$test_dir"', '--strategy', 'random_sample', '--enable-pretrain'],
    dataDir: 'embedded',
    disabledForTasks: [],
    enabledTasks: ['ocr'],
    disabledReason: DISABLED_REASONS.catmusOcrOnly
  }
];

const COLUMN_GROUPS = [
  { label: '5-fold', columns: ['fold_diplomatic', 'fold_editorial'] },
  { label: 'Sequential', columns: ['sequential_diplomatic', 'sequential_editorial'] },
  { label: 'Train/test', columns: ['train_test_cross_manuscript'] },
  { label: 'Pretrain → sequential', columns: ['pretrain_synthetic_sequential', 'pretrain_catmus_sequential'] },
  { label: 'Pretrain → train/test', columns: ['pretrain_train_test', 'pretrain_catmus_train_test'] }
];

const LEGEND = [
  DISABLED_REASONS.syntheticNoLayoutData,
  DISABLED_REASONS.catmusOcrOnly
];

const SCRIPT_FILENAME = 'experiments.sh';

/* ------------------------------------------------------------------ helpers */

function findRow(rowId) {
  for (let i = 0; i < ROWS.length; i++) {
    if (ROWS[i].id === rowId) return ROWS[i];
  }
  return null;
}

function findColumn(colId) {
  for (let i = 0; i < COLUMNS.length; i++) {
    if (COLUMNS[i].id === colId || COLUMNS[i].letter === colId) return COLUMNS[i];
  }
  return null;
}

function cellKey(rowId, colId) {
  return rowId + '|' + colId;
}

/* Returns the disabled reason for a cell, or null when the cell is enabled. */
function disabledReason(row, column) {
  if (column.disabledForTasks && column.disabledForTasks.indexOf(row.task) !== -1) {
    return column.disabledReason;
  }
  if (column.enabledTasks && column.enabledTasks.indexOf(row.task) === -1) {
    return column.disabledReason;
  }
  return null;
}

function isDisabled(row, column) {
  return disabledReason(row, column) !== null;
}

function isCellEnabled(rowId, colId) {
  const row = findRow(rowId);
  const column = findColumn(colId);
  if (!row || !column) return false;
  return !isDisabled(row, column);
}

/* Enabled cell keys of one column, in row display order. */
function enabledCellsInColumn(colId) {
  const column = findColumn(colId);
  if (!column) return [];
  const out = [];
  ROWS.forEach(function (row) {
    if (!isDisabled(row, column)) out.push(cellKey(row.id, column.id));
  });
  return out;
}

/* Enabled cell keys of one row, in column order. */
function enabledCellsInRow(rowId) {
  const row = findRow(rowId);
  if (!row) return [];
  const out = [];
  COLUMNS.forEach(function (column) {
    if (!isDisabled(row, column)) out.push(cellKey(row.id, column.id));
  });
  return out;
}

/* ----------------------------------------------------------------- commands */

function buildCommand(row, column) {
  const parts = ['./run.sh', '--framework', row.framework, '--task', row.task];
  if (row.model) {
    parts.push('--model-name', '"' + row.model + '"');
  }
  column.args.forEach(function (arg) { parts.push(arg); });
  if (column.dataDir === 'standard') {
    parts.push('--data-dir', '"$DATA_DIR"');
  }
  parts.push('$COMMON_ARGS');
  return parts.join(' ');
}

/* Accepts an array/Set of "rowId|colId" keys, or {rowId, colId} objects.
 * Returns a Set of valid, enabled keys. */
function normalizeSelection(selection) {
  const set = new Set();
  if (!selection) return set;
  const list = (selection instanceof Set) ? Array.from(selection)
    : (Array.isArray(selection) ? selection : Object.keys(selection));
  list.forEach(function (entry) {
    let key = null;
    if (typeof entry === 'string') {
      key = entry;
    } else if (entry && typeof entry === 'object') {
      if (entry.key) key = entry.key;
      else if (entry.rowId && entry.colId) key = cellKey(entry.rowId, entry.colId);
      else if (entry.row && entry.col) key = cellKey(
        typeof entry.row === 'string' ? entry.row : entry.row.id,
        typeof entry.col === 'string' ? entry.col : entry.col.id
      );
    }
    if (!key) return;
    const sep = key.indexOf('|');
    if (sep === -1) return;
    const row = findRow(key.slice(0, sep));
    const column = findColumn(key.slice(sep + 1));
    if (!row || !column) return;
    if (isDisabled(row, column)) return;
    set.add(cellKey(row.id, column.id));
  });
  return set;
}

/* Selected cells in canonical order: columns A..I, rows in display order. */
function orderedSelection(selection) {
  const set = normalizeSelection(selection);
  const cells = [];
  COLUMNS.forEach(function (column) {
    ROWS.forEach(function (row) {
      if (set.has(cellKey(row.id, column.id))) {
        cells.push({ row: row, column: column, key: cellKey(row.id, column.id) });
      }
    });
  });
  return cells;
}

/* ------------------------------------------------------------- scaffolding */

const SCAFFOLD_HEADER = [
  '# experiments.sh - Run all combinations of frameworks, tasks, model indices, sequential strategies, and editions',
  '# This script runs the selected benchmark combinations without loops',
  '',
  '# Use this like:',
  '# ./experiments.sh data/I-Ct_91',
  '# ./experiments.sh data/I-Fn_BR_18 --debug',
  '# ./experiments.sh data/I-Ct_91 --device cuda:1',
  '# ./experiments.sh data/I-Fn_BR_18 --debug --device cpu',
  '',
  '#===================',
  '# Argument parsing',
  '#===================',
  '',
  '# Initialize variables',
  'DATA_DIR=""',
  'DEBUG_FLAG=""',
  'DEVICE_FLAG=""',
  'ALLOWED_DATA_DIRS=("data/I-Ct_91" "data/I-Fn_BR_18")',
  '',
  '# Function to show usage',
  'show_usage() {',
  '    echo "Usage: $0 <data_directory> [--debug] [--device <device>]"',
  '    echo "  data_directory: Must be one of: ${ALLOWED_DATA_DIRS[*]}"',
  '    echo "  --debug:        Enable debug mode (optional)"',
  '    echo "  --device:       Specify device (e.g., cuda:0, cpu) (optional)"',
  '    exit 1',
  '}',
  '',
  '# Function to validate data directory',
  'validate_data_dir() {',
  '    local dir="$1"',
  '    for allowed_dir in "${ALLOWED_DATA_DIRS[@]}"; do',
  '        if [ "$dir" = "$allowed_dir" ]; then',
  '            return 0',
  '        fi',
  '    done',
  '    return 1',
  '}',
  '',
  '# Parse arguments',
  'while [[ $# -gt 0 ]]; do',
  '    case $1 in',
  '        --debug)',
  '            set -e',
  '            DEBUG_FLAG="--debug"',
  '            shift',
  '            ;;',
  '        --device)',
  '            if [ -z "$2" ]; then',
  '                echo "Error: --device requires a value"',
  '                show_usage',
  '            fi',
  '            DEVICE_FLAG="--device $2"',
  '            shift 2',
  '            ;;',
  '        -*)',
  '            echo "Error: Unknown option $1"',
  '            show_usage',
  '            ;;',
  '        *)',
  '            if [ -z "$DATA_DIR" ]; then',
  '                DATA_DIR="$1"',
  '            else',
  '                echo "Error: Multiple data directories specified"',
  '                show_usage',
  '            fi',
  '            shift',
  '            ;;',
  '    esac',
  'done',
  '',
  '# Validate required arguments',
  'if [ -z "$DATA_DIR" ]; then',
  '    echo "Error: Data directory is required"',
  '    show_usage',
  'fi',
  '',
  'if [ ! -d "$DATA_DIR" ]; then',
  '    echo "Error: Data directory \'$DATA_DIR\' does not exist"',
  '    exit 1',
  'fi',
  '',
  'if ! validate_data_dir "$DATA_DIR"; then',
  '    echo "Error: Data directory must be one of: ${ALLOWED_DATA_DIRS[*]}"',
  '    exit 1',
  'fi',
  '',
  '# Generate experiment ID that includes dataset name for traceability',
  '# This ensures concurrent experiments on the same or different datasets don\'t conflict',
  'DATASET_NAME=$(basename "$DATA_DIR")',
  'export LAUDARE_EXPERIMENT_ID="${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)_$$"',
  'echo "🔬 Experiment ID: $LAUDARE_EXPERIMENT_ID"',
  'echo ""',
  '',
  'mkdir -p models/$LAUDARE_EXPERIMENT_ID',
  'mkdir -p results/$LAUDARE_EXPERIMENT_ID',
  'echo "Now you can inject pretrained models and results."',
  'echo "Experiment ID: $LAUDARE_EXPERIMENT_ID"',
  'if [[ -t 0 ]]; then',
  '    read -p "Press [enter] to continue or CTRL-C to exit > "',
  'else',
  '    echo "Non-interactive session detected; continuing without pause."',
  'fi',
  '',
  '# Build the common arguments string',
  'COMMON_ARGS="$DEBUG_FLAG $DEVICE_FLAG"'
];

const ANNOTATIONS_BLOCK = [
  '# #@@@@@@@@@@@@@@@',
  '# #@ Annotations @',
  '# #@@@@@@@@@@@@@@@',
  '# # These are the commands used to generate annotations and pre-trained data',
  '# # ./run.sh --task synthesis',
  '# # ./run.sh --task annotations --edition diplomatic --data-dir data/I-Ct_91',
  '# # ./run.sh --task annotations --edition editorial --data-dir data/I-Ct_91',
  '# # ./run.sh --task annotations --edition diplomatic --data-dir data/I-Fn_BR_18',
  '# # ./run.sh --task annotations --edition editorial --data-dir data/I-Fn_BR_18'
];

const TRAIN_TEST_BLOCK = [
  '#@@@@@@@@@@@@@@',
  '#@ train-test @',
  '#@@@@@@@@@@@@@@',
  '',
  '# Determine train and test directories based on data directory',
  'if [ "$DATA_DIR" == "data/I-Ct_91" ]; then',
  '  train_dir="data/I-Ct_91"',
  '  # test_dir="data/I-Ct_91"',
  '  test_dir="data/I-Fn_BR_18"',
  'else',
  '  train_dir="data/I-Fn_BR_18"',
  '  # train_dir="data/I-Ct_91"',
  '  test_dir="data/I-Ct_91"',
  'fi'
];

function sectionHeader(column) {
  const title = '#@ ' + column.letter + ' · ' + column.label;
  const bar = '#' + new Array(Math.max(title.length + 2, 30)).join('@');
  return [bar, title, bar];
}

/* ----------------------------------------------------------- buildScript */

function buildScript(selection, options) {
  const opts = options || {};
  const timestamp = opts.timestamp || new Date().toISOString();
  const cells = orderedSelection(selection);

  const commandCount = cells.length;
  const rowCount = new Set(cells.map(function (c) { return c.row.id; })).size;
  const columnCount = new Set(cells.map(function (c) { return c.column.id; })).size;
  const usesTrainTest = cells.some(function (c) { return c.column.dataDir !== 'standard'; });

  const lines = [];
  lines.push('#!/bin/bash');
  lines.push('');
  lines.push('# Generated by the Laudare experiment matrix on ' + timestamp);
  lines.push('# Selection: ' + commandCount + ' cell(s), ' + commandCount + ' command(s), ' +
             rowCount + ' row(s), ' + columnCount + ' column(s)');
  if (commandCount === 0) {
    lines.push('# No cells selected: this script only sets up the experiment environment.');
  }
  lines.push('');
  lines.push.apply(lines, SCAFFOLD_HEADER);
  lines.push('');
  lines.push.apply(lines, ANNOTATIONS_BLOCK);
  lines.push('');

  if (usesTrainTest) {
    lines.push.apply(lines, TRAIN_TEST_BLOCK);
    lines.push('');
  }

  COLUMNS.forEach(function (column) {
    const columnCells = cells.filter(function (c) { return c.column.id === column.id; });
    if (columnCells.length === 0) return;
    lines.push.apply(lines, sectionHeader(column));
    lines.push('');
    columnCells.forEach(function (cell) {
      lines.push(buildCommand(cell.row, cell.column));
    });
    lines.push('');
  });

  let text = lines.join('\n');
  if (text.charAt(text.length - 1) !== '\n') text += '\n';
  return text;
}

/* ------------------------------------------------------------------- API */

const TOTAL_CELLS = ROWS.length * COLUMNS.length;

const LaudareMatrix = {
  TASK_GROUPS: TASK_GROUPS,
  ROWS: ROWS,
  COLUMNS: COLUMNS,
  COLUMN_GROUPS: COLUMN_GROUPS,
  LEGEND: LEGEND,
  DISABLED_REASONS: DISABLED_REASONS,
  SCRIPT_FILENAME: SCRIPT_FILENAME,
  TOTAL_CELLS: TOTAL_CELLS,
  findRow: findRow,
  findColumn: findColumn,
  cellKey: cellKey,
  disabledReason: disabledReason,
  isDisabled: isDisabled,
  isCellEnabled: isCellEnabled,
  enabledCellsInColumn: enabledCellsInColumn,
  enabledCellsInRow: enabledCellsInRow,
  buildCommand: buildCommand,
  normalizeSelection: normalizeSelection,
  orderedSelection: orderedSelection,
  buildScript: buildScript
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = LaudareMatrix;
}
