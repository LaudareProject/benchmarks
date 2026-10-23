# Laudare experiment matrix (static GUI)

A dependency-free web page that generates a runnable replacement for the repository's
`experiments.sh`: pick framework/task rows and protocol columns, then download a script that runs
exactly the selected benchmarks.

No build step, no npm packages, no external assets, fonts, or CDNs.

## Open it

Either of these works:

```bash
# Straight from disk (works because the scripts are plain classic <script> tags)
xdg-open gui/index.html

# Or over HTTP
python3 -m http.server -d gui 8000
# then browse http://localhost:8000/
```

## How the matrix works

- **Rows** are framework · task combinations (model variants are separate rows), grouped by task:
  Layout, OCR, OMR. Row ids are stable and live in `matrix.js` (`ROWS`).
- **Columns** are experiment protocols A–I, grouped in the header:
  - `5-fold` — A diplomatic, B editorial (`--n-fold`)
  - `Sequential` — C diplomatic, D editorial (`--strategy random_sample`)
  - `Train/test` — E cross-manuscript (`--train-dir "$train_dir" --test-dir "$test_dir"`)
  - `Pretrain → sequential` — F synthetic (default `data/pretrain_data`), G CATMuS
  - `Pretrain → train/test` — H train manuscript, I CATMuS
- **Grey hatched cells are not applicable** and cannot be selected:
  - F (synthetic pretraining) is disabled for every layout row — `data/pretrain_data` contains only
    OCR/OMR data, so there is no synthetic layout corpus.
  - H is available for layout rows: it pretrains on the train manuscript's own annotations, which do
    include layout (this matches the layout commands in the checked-in `experiments.sh`).
  - G and I are disabled for every non-OCR row (all OMR and layout rows) — CATMuS is an OCR-only
    pretraining corpus with no music or layout annotations.
  - Nothing else is restricted.
- **Interaction**: click a cell checkbox to toggle it; click a column title checkbox to toggle all
  enabled cells in that column (indeterminate when partially selected); click a row label to toggle
  all enabled cells in that row. Keyboard users can reach every control with Tab and toggle with
  Space/Enter.
- **Selected count** updates live; `Generate experiments.sh` is disabled while nothing is selected.

## Generated file

`Generate experiments.sh` writes `experiments.sh` via a `Blob` download. It keeps the original
script's scaffolding (argument parsing, `ALLOWED_DATA_DIRS`, data-dir validation, experiment ID,
`models/` + `results/` creation, inject/pause message, `COMMON_ARGS`, annotation notes) and replaces
the benchmark command list with only the selected commands.

- Commands are emitted in column order A→I, rows in display order, one section comment per column.
- The train/test direction block (`train_dir`/`test_dir`) is included only when a selected column
  uses `$train_dir`/`$test_dir`.
- Commands look like:

  ```bash
  ./run.sh --framework kraken --task ocr --edition diplomatic --strategy random_sample --data-dir "$DATA_DIR" $COMMON_ARGS
  ./run.sh --framework detr --task layout --edition diplomatic --train-dir "$train_dir" --test-dir "$test_dir" $COMMON_ARGS
  ```

The download is not executable by default:

```bash
chmod +x experiments.sh
./experiments.sh data/I-Ct_91
```

Put the generated file at the repository root (that is where `./run.sh`, `data/`, `models/`, and
`results/` are expected) before running it.

## Files

- `index.html` — page shell: toolbar, matrix table, legend.
- `matrix.js` — `ROWS`, `COLUMNS`, disable rules, command/script generation. Pure logic: usable from
  Node via `require('./matrix.js')`.
- `app.js` — matrix DOM construction, checkbox state, master checkboxes, download.
- `styles.css` — compact matrix, sticky header/first column, horizontal scroll, light/dark.
- `test_generate.cjs` — generation and shell-syntax tests.

## Tests

```bash
node gui/test_generate.cjs
```

The test asserts the disable rules, exact command strings, command ordering/uniqueness, the empty
selection case, and runs `bash -n` on both an empty and a full (139-command) script. It also writes a
representative sample to `/tmp/laudare-experiments.generated.sh`.
