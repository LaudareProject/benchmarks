#!/usr/bin/env node
/*
 * test_generate.cjs - Node test for gui/matrix.js generation logic.
 * Run with: node gui/test_generate.cjs
 * No dependencies beyond Node's standard library.
 */
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const M = require('./matrix.js');

const SAMPLE_PATH = '/tmp/laudare-experiments.generated.sh';

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log('ok   - ' + name);
  } catch (error) {
    failures.push({ name, error });
    console.log('FAIL - ' + name);
    console.log('       ' + String(error && error.message ? error.message : error).split('\n').join('\n       '));
  }
}

function key(rowId, colId) {
  return M.cellKey(rowId, colId);
}

function commandLines(script) {
  return script.split('\n').filter((line) => line.indexOf('./run.sh') === 0);
}

function bashSyntax(script, label) {
  const file = path.join(os.tmpdir(), 'laudare-test-' + label + '.sh');
  fs.writeFileSync(file, script, 'utf8');
  execFileSync('bash', ['-n', file], { stdio: 'pipe' });
  return file;
}

/* ------------------------------------------------------------------ rules */

test('F is disabled for all layout rows (synthetic data has no layout)', () => {
  const layoutRows = M.ROWS.filter((row) => row.task === 'layout');
  assert.strictEqual(layoutRows.length, 6, 'expected 6 layout rows');
  const column = M.findColumn('pretrain_synthetic_sequential');
  layoutRows.forEach((row) => {
    assert.strictEqual(M.disabledReason(row, column), M.DISABLED_REASONS.syntheticNoLayoutData,
      row.id + ' x F must be disabled (data/pretrain_data has no layout data)');
    assert.strictEqual(M.isCellEnabled(row.id, 'pretrain_synthetic_sequential'), false);
  });
});

test('H is enabled for every row, including layout rows', () => {
  const column = M.findColumn('pretrain_train_test');
  M.ROWS.forEach((row) => {
    assert.strictEqual(M.disabledReason(row, column), null,
      row.id + ' x H must be enabled (pretrains on the train manuscript annotations)');
  });
  assert.strictEqual(M.isCellEnabled('detr_layout', 'pretrain_train_test'), true,
    'detr_layout x H must be enabled');
  assert.strictEqual(M.isCellEnabled('doclayout_yolo_layout', 'pretrain_train_test'), true,
    'doclayout_yolo_layout x H must be enabled');
});

test('F is enabled for every OCR and OMR row', () => {
  const rows = M.ROWS.filter((row) => row.task === 'ocr' || row.task === 'omr');
  assert.strictEqual(rows.length, 13, 'expected 13 ocr/omr rows');
  const column = M.findColumn('pretrain_synthetic_sequential');
  rows.forEach((row) => {
    assert.strictEqual(M.disabledReason(row, column), null,
      row.id + ' x F must be enabled');
  });
});

test('G and I are enabled only for OCR rows', () => {
  M.ROWS.forEach((row) => {
    ['pretrain_catmus_sequential', 'pretrain_catmus_train_test'].forEach((colId) => {
      const column = M.findColumn(colId);
      if (row.task === 'ocr') {
        assert.strictEqual(M.disabledReason(row, column), null, row.id + ' x ' + colId + ' must be enabled');
      } else {
        assert.strictEqual(M.disabledReason(row, column), M.DISABLED_REASONS.catmusOcrOnly,
          row.id + ' x ' + colId + ' must be disabled (CATMuS is OCR-only)');
      }
    });
  });
});

test('all other cells are enabled', () => {
  const freeColumns = ['fold_diplomatic', 'fold_editorial', 'sequential_diplomatic',
    'sequential_editorial', 'train_test_cross_manuscript', 'pretrain_train_test'];
  M.ROWS.forEach((row) => {
    freeColumns.forEach((colId) => {
      assert.strictEqual(M.disabledReason(row, M.findColumn(colId)), null,
        row.id + ' x ' + colId + ' must be enabled');
    });
  });
});

test('matrix shape: 19 rows, 9 columns, 139 enabled cells of 171', () => {
  assert.strictEqual(M.ROWS.length, 19);
  assert.strictEqual(M.COLUMNS.length, 9);
  assert.strictEqual(M.TOTAL_CELLS, 171);
  let enabled = 0;
  M.ROWS.forEach((row) => { enabled += M.enabledCellsInRow(row.id).length; });
  assert.strictEqual(enabled, 139);
  assert.strictEqual(M.TOTAL_CELLS - enabled, 32, 'expected 32 disabled cells');
  assert.strictEqual(M.LEGEND.length, 2, 'legend needs both disabled reasons');
  M.LEGEND.forEach((text) => {
    assert.ok(typeof text === 'string' && text.length > 10, 'legend entry must be a reason string');
  });
});

test('row ids and column letters are unique and stable', () => {
  const ids = new Set(M.ROWS.map((row) => row.id));
  assert.strictEqual(ids.size, M.ROWS.length, 'duplicate row id');
  assert.deepStrictEqual(M.COLUMNS.map((column) => column.letter),
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']);
});

/* --------------------------------------------------------------- commands */

test('single command strings match experiments.sh conventions', () => {
  const expectations = [
    ['detr_layout', 'fold_diplomatic',
      './run.sh --framework detr --task layout --edition diplomatic --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['faster_rcnn_mobilenet_layout', 'fold_editorial',
      './run.sh --framework faster_rcnn --task layout --model-name "mobilenet" --edition editorial --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['kraken_ocr', 'sequential_diplomatic',
      './run.sh --framework kraken --task ocr --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['trocr_large_omr', 'sequential_editorial',
      './run.sh --framework trocr --task omr --model-name "large" --edition editorial --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['doclayout_yolo_layout', 'train_test_cross_manuscript',
      './run.sh --framework doclayout_yolo --task layout --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS'],
    ['paddleocr_vl_ocr', 'pretrain_synthetic_sequential',
      './run.sh --framework paddleocr_vl --task ocr --edition diplomatic --strategy random_sample --enable-pretrain --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['calamari_ocr', 'pretrain_catmus_sequential',
      './run.sh --framework calamari --task ocr --edition diplomatic --pretrain-dir data/CATMuS_medieval --strategy random_sample --enable-pretrain --data-dir "$DATA_DIR" $COMMON_ARGS'],
    ['vlt_omr', 'pretrain_train_test',
      './run.sh --framework vlt --task omr --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS'],
    ['detr_layout', 'pretrain_train_test',
      './run.sh --framework detr --task layout --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS'],
    ['kraken_ocr', 'pretrain_catmus_train_test',
      './run.sh --framework kraken --task ocr --edition diplomatic --pretrain-dir data/CATMuS_medieval --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS']
  ];
  expectations.forEach(([rowId, colId, expected]) => {
    const row = M.findRow(rowId);
    const column = M.findColumn(colId);
    assert.ok(row && column, 'missing ' + rowId + ' or ' + colId);
    assert.strictEqual(M.buildCommand(row, column), expected);
  });
});

/* ------------------------------------------------------------ buildScript */

const REPRESENTATIVE = [
  key('detr_layout', 'fold_diplomatic'),                              // A
  key('yolo_yolo26s_layout', 'fold_editorial'),                       // B
  key('kraken_ocr', 'sequential_diplomatic'),                         // C
  key('trocr_large_omr', 'sequential_editorial'),                     // D
  key('doclayout_yolo_layout', 'train_test_cross_manuscript'),        // E
  key('paddleocr_vl_ocr', 'pretrain_synthetic_sequential'),           // F
  key('calamari_ocr', 'pretrain_catmus_sequential'),                  // G
  key('vlt_omr', 'pretrain_train_test'),                              // H
  key('kraken_ocr', 'pretrain_catmus_train_test')                     // I
];

const EXPECTED_REPRESENTATIVE_LINES = [
  './run.sh --framework detr --task layout --edition diplomatic --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework yolo --task layout --model-name "yolo26s" --edition editorial --n-fold --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework kraken --task ocr --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework trocr --task omr --model-name "large" --edition editorial --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework doclayout_yolo --task layout --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS',
  './run.sh --framework paddleocr_vl --task ocr --edition diplomatic --strategy random_sample --enable-pretrain --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework calamari --task ocr --edition diplomatic --pretrain-dir data/CATMuS_medieval --strategy random_sample --enable-pretrain --data-dir "$DATA_DIR" $COMMON_ARGS',
  './run.sh --framework vlt --task omr --edition diplomatic --pretrain-dir "$train_dir" --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS',
  './run.sh --framework kraken --task ocr --edition diplomatic --pretrain-dir data/CATMuS_medieval --data-dir "$test_dir" --strategy random_sample --enable-pretrain $COMMON_ARGS'
];

test('representative selection: exact commands, column order, no duplicates', () => {
  const script = M.buildScript(REPRESENTATIVE, { timestamp: '2026-10-22T00:00:00.000Z' });
  const lines = commandLines(script);
  assert.deepStrictEqual(lines, EXPECTED_REPRESENTATIVE_LINES);
  assert.strictEqual(lines.length, 9);
  assert.strictEqual(new Set(lines).size, lines.length, 'duplicate command lines');
});

test('selection order in input does not affect output order', () => {
  const shuffled = REPRESENTATIVE.slice().reverse();
  const a = M.buildScript(REPRESENTATIVE, { timestamp: 'x' });
  const b = M.buildScript(shuffled, { timestamp: 'x' });
  assert.strictEqual(a, b);
});

test('script header carries timestamp, counts and scaffolding', () => {
  const script = M.buildScript(REPRESENTATIVE, { timestamp: '2026-10-22T10:11:12.345Z' });
  assert.ok(script.indexOf('#!/bin/bash\n') === 0, 'must start with shebang');
  assert.ok(script.includes('# Generated by the Laudare experiment matrix on 2026-10-22T10:11:12.345Z'));
  assert.ok(script.includes('# Selection: 9 cell(s), 9 command(s), 8 row(s), 9 column(s)'));
  assert.ok(script.includes('ALLOWED_DATA_DIRS=("data/I-Ct_91" "data/I-Fn_BR_18")'));
  assert.ok(script.includes('export LAUDARE_EXPERIMENT_ID="${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)_$$"'));
  assert.ok(script.includes('COMMON_ARGS="$DEBUG_FLAG $DEVICE_FLAG"'));
  assert.ok(script.includes('mkdir -p models/$LAUDARE_EXPERIMENT_ID'));
  assert.ok(script.includes('mkdir -p results/$LAUDARE_EXPERIMENT_ID'));
  assert.ok(script.includes('read -p "Press [enter] to continue or CTRL-C to exit > "'));
  assert.ok(script.includes('# # ./run.sh --task synthesis'), 'annotations block preserved');
  assert.ok(script.includes('#@ Annotations @'), 'annotations block preserved');
  assert.ok(script.includes('validate_data_dir'));
  assert.ok(script.includes('--debug)'), 'debug parsing preserved');
  assert.ok(script.includes('--device)'), 'device parsing preserved');
});

test('one section comment per selected column, in A..I order', () => {
  const script = M.buildScript(REPRESENTATIVE, { timestamp: 'x' });
  const markers = ['A · 5-fold · Diplomatic', 'B · 5-fold · Editorial', 'C · Sequential · Diplomatic',
    'D · Sequential · Editorial', 'E · Train/test · Cross-manuscript', 'F · Pretrain synthetic → sequential',
    'G · Pretrain CATMuS → sequential', 'H · Pretrain train manuscript → train/test',
    'I · Pretrain CATMuS → train/test'];
  let cursor = -1;
  markers.forEach((marker) => {
    const at = script.indexOf('#@ ' + marker);
    assert.ok(at !== -1, 'missing section comment for ' + marker);
    assert.ok(at > cursor, 'section ' + marker + ' out of order');
    cursor = at;
  });
  // Only one command section header per letter, no empty sections.
  const letters = (script.match(/^#@ [A-I] · /gm) || []);
  assert.strictEqual(letters.length, 9);
});

test('train/test direction block only when $train_dir/$test_dir are used', () => {
  const withE = M.buildScript([key('detr_layout', 'fold_diplomatic'), key('detr_layout', 'train_test_cross_manuscript')], { timestamp: 'x' });
  assert.ok(withE.includes('if [ "$DATA_DIR" == "data/I-Ct_91" ]; then'));
  assert.ok(withE.includes('train_dir="data/I-Ct_91"'));
  assert.ok(withE.includes('test_dir="data/I-Ct_91"'));

  const withoutE = M.buildScript([key('kraken_ocr', 'fold_diplomatic')], { timestamp: 'x' });
  assert.ok(!withoutE.includes('train_dir="'), 'no train/test block expected');

  const onlyH = M.buildScript([key('vlt_omr', 'pretrain_train_test')], { timestamp: 'x' });
  assert.ok(onlyH.includes('train_dir="data/I-Fn_BR_18"'), 'H needs the direction block');
});

test('disabled cells are ignored by buildScript', () => {
  const script = M.buildScript([
    key('detr_layout', 'pretrain_synthetic_sequential'),
    key('bgk_omr', 'pretrain_catmus_train_test')
  ], { timestamp: 'x' });
  assert.strictEqual(commandLines(script).length, 0);
});

test('unknown / malformed selection entries are ignored', () => {
  const script = M.buildScript(['nope|fold_diplomatic', 'detr_layout|nope', 'garbage', null, { rowId: 'detr_layout', colId: 'fold_diplomatic' }], { timestamp: 'x' });
  assert.strictEqual(commandLines(script).length, 1);
});

test('empty selection yields a valid script with no run.sh lines', () => {
  const script = M.buildScript([], { timestamp: '2026-10-22T00:00:00.000Z' });
  assert.strictEqual(commandLines(script).length, 0);
  assert.ok(script.includes('# No cells selected'));
  assert.ok(script.includes('COMMON_ARGS="$DEBUG_FLAG $DEVICE_FLAG"'));
  bashSyntax(script, 'empty');
});

test('Set and object selection forms are accepted', () => {
  const a = M.buildScript(new Set([key('detr_layout', 'fold_diplomatic')]), { timestamp: 'x' });
  const b = M.buildScript([{ rowId: 'detr_layout', colId: 'fold_diplomatic' }], { timestamp: 'x' });
  assert.strictEqual(a, b);
  assert.strictEqual(commandLines(a).length, 1);
});

test('full selection: 139 commands, no duplicates, bash -n passes', () => {
  const all = [];
  M.ROWS.forEach((row) => {
    M.COLUMNS.forEach((column) => {
      const k = key(row.id, column.id);
      if (M.isCellEnabled(row.id, column.id)) all.push(k);
    });
  });
  assert.strictEqual(all.length, 139);
  const script = M.buildScript(all);
  const lines = commandLines(script);
  assert.strictEqual(lines.length, 139);
  assert.strictEqual(new Set(lines).size, 139);
  bashSyntax(script, 'full');
});

test('entities in generated commands are limited to the fixed configuration', () => {
  const script = M.buildScript(REPRESENTATIVE, { timestamp: 'x' });
  const lines = commandLines(script);
  const frameworks = new Set(M.ROWS.map((row) => row.framework));
  const tasks = new Set(M.ROWS.map((row) => row.task));
  lines.forEach((line) => {
    assert.ok(/^\.\/run\.sh --framework [a-z0-9_]+ --task [a-z]+ /.test(line), 'unexpected command shape: ' + line);
    const framework = line.split(' ')[2];
    const task = line.split(' ')[4];
    assert.ok(frameworks.has(framework), 'unknown framework ' + framework);
    assert.ok(tasks.has(task), 'unknown task ' + task);
    assert.ok(line.endsWith('$COMMON_ARGS'), 'command must end with $COMMON_ARGS');
  });
});

test('generated sample passes bash -n and is written for inspection', () => {
  const script = M.buildScript(REPRESENTATIVE);
  fs.writeFileSync(SAMPLE_PATH, script, 'utf8');
  execFileSync('bash', ['-n', SAMPLE_PATH], { stdio: 'pipe' });
  assert.ok(fs.statSync(SAMPLE_PATH).size > 0);
});

/* ---------------------------------------------------------------- report */

console.log('');
console.log('Sample script: ' + SAMPLE_PATH);
console.log('Passed: ' + passed + ', failed: ' + failures.length);
if (failures.length > 0) {
  process.exitCode = 1;
}
