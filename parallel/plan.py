#!/usr/bin/env python3
"""Split the experiments.sh matrix into two GPU-balanced lanes.

Work decomposition:
  * Pretraining (synthetic and CATMuS) depends only on framework/task/model, not
    on the dataset, so each checkpoint is trained once and shared by both dataset
    runs through one LAUDARE_EXPERIMENT_ID.
  * Downstream commands (fine-tuning/eval) depend on the dataset and are
    distributed across the two lanes by measured cost.
  * Jobs that write the same output directory (C/F/G and E/H/I differ only by
    pretrain source) are kept on the same lane and ordered, so the later row
    still overwrites the earlier one exactly as in the serial script.

Writes parallel/lane0.sh, parallel/lane1.sh and parallel/manifest.tsv.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Production cost per fold, hours. Sources: queue-log timestamps + 7-min probes.
FOLD_H = {
    ("yolo", "layout", "yolo26m"): 25 / 60,
    ("yolo", "layout", "yolo26s"): 17 / 60,
    ("doclayout_yolo", "layout", "default"): 13 / 60,
    ("paddleocr_vl", "ocr", "default"): 36 / 60,
    ("vlt", "ocr", "default"): 98 / 60,
    ("paddleocr_vl", "omr", "default"): 25 / 60,
    ("vlt", "omr", "default"): 98 / 60,
    ("bgk", "omr", "default"): 12 / 60,
    ("kraken", "ocr", "default"): 13 / 60,
    ("calamari", "ocr", "default"): 7 / 60,
    ("trocr", "ocr", "small"): 250 / 60,
    ("trocr", "ocr", "large"): 340 / 60,
}

# Multipliers on the per-fold cost.
SECTION_FACTOR = {"A": 5.0, "B": 5.0, "C": 3.5, "D": 3.5, "E": 1.3,
                  "F": 3.5, "G": 3.5, "H": 2.4, "I": 1.3}
N_FOLD = {"A", "B"}

# Pretraining hours. Synthetic: 5000 samples. CATMuS: 152,816 train lines.
SYNTHETIC_H = {
    # Recognition loaders iterate annotations (line/staff crops), detection loaders
    # iterate page images. Synthetic set: 50,396 OCR lines / 59,997 OMR staves /
    # 3,899 + 4,000 page images.
    #   paddleocr_vl 0.91 lines/s, vlt 60 lines/s, bgk 42.7 images/s (68 epochs)
    ("paddleocr_vl", "ocr", "default"): 50396 * 2 / 0.91 / 3600,
    ("paddleocr_vl", "omr", "default"): 59997 * 2 / 0.91 / 3600,
    ("vlt", "ocr", "default"): 50396 * 210 / 60 / 3600,
    ("vlt", "omr", "default"): 59997 * 210 / 60 / 3600,
    ("bgk", "omr", "default"): 4000 * 68 / 42.7 / 3600,
}
CATMUS_H = {
    # 152,816 lines, ~20 epochs before early stopping (patience 15, max 50/100/2/210).
    # Rates: trocr small 8.4 lines/s, large 6.2, paddleocr_vl 0.91, vlt 86 (measured),
    # calamari ~397 lines/s from the Laudare fold; kraken aborts on the PageXML.
    ("kraken", "ocr", "default"): 0.02,
    ("calamari", "ocr", "default"): 152816 * 20 / 397 / 3600,
    ("trocr", "ocr", "small"): 152816 * 20 / 8.4 / 3600,
    ("trocr", "ocr", "large"): 152816 * 20 / 6.2 / 3600,
    ("paddleocr_vl", "ocr", "default"): 152816 * 2 / 0.91 / 3600,
    ("vlt", "ocr", "default"): 152816 * 210 / 86 / 3600,
}

OMR_SCALE_BR18 = 1.85


def parse_matrix(path: Path):
    section = None
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"^#@ ([A-I]) · (.*)$", line)
        if m:
            section = m.group(1)
            continue
        if not line.startswith("./run.sh"):
            continue
        raw = (line.replace('"$DATA_DIR"', "__DATA_DIR__")
                   .replace('"$train_dir"', "__TRAIN_DIR__")
                   .replace('"$test_dir"', "__TEST_DIR__")
                   .replace("$COMMON_ARGS", "__COMMON__"))
        toks = [t for t in shlex.split(raw) if t != "__COMMON__"]
        rows.append({"section": section, "tokens": toks})
    return rows


def opt(tokens, name, default=None):
    if name in tokens:
        i = tokens.index(name)
        if i + 1 < len(tokens):
            return tokens[i + 1]
    return default


def cost_hours(section, fw, task, model, dataset) -> float:
    key = (fw, task, model)
    if key not in FOLD_H:
        raise KeyError(f"no cost model for {key}")
    scale = OMR_SCALE_BR18 if (dataset == "I-Fn_BR_18" and task == "omr") else 1.0
    return FOLD_H[key] * scale * SECTION_FACTOR[section]


def dataset_paths(dataset: str):
    if dataset == "I-Ct_91":
        return "data/I-Ct_91", "data/I-Fn_BR_18"
    return "data/I-Fn_BR_18", "data/I-Ct_91"


def build_jobs(rows, datasets, run_ids, pretrain_id):
    downstream = []
    pretrains = OrderedDict()

    for dataset in datasets:
        data_dir, test_dir = dataset_paths(dataset)
        train_dir = data_dir
        ds_tag = "ct" if dataset == "I-Ct_91" else "fn"

        for row in rows:
            section = row["section"]
            toks = list(row["tokens"])
            fw = opt(toks, "--framework")
            task = opt(toks, "--task")
            model = opt(toks, "--model-name", "default")
            edition = opt(toks, "--edition")
            nfold = "--n-fold" in toks

            cmd = []
            for t in toks:
                cmd.append({"__DATA_DIR__": data_dir,
                            "__TRAIN_DIR__": train_dir,
                            "__TEST_DIR__": test_dir}.get(t, t))

            eff_test = test_dir if section in {"E", "H", "I"} else dataset
            run_id = run_ids[dataset]
            if section in {"C", "F", "G"}:
                key = f"{run_id}|seq|{dataset}|{edition}|{fw}|{task}|{model}"
            elif section in {"E", "H", "I"}:
                key = f"{run_id}|tt|{eff_test}|{edition}|{fw}|{task}|{model}"
            else:
                key = f"{run_id}|fold|{dataset}|{edition}|{fw}|{task}|{model}|{section}"

            pre_id = None
            if section in {"F", "G", "I"}:
                src = "syn" if section == "F" else "cat"
                pre_id = f"pre_{src}_{fw}_{task}_{model}"

            job = {
                "phase": "downstream",
                "pretrain_id": pre_id,
                "run_id": run_id,
                "id": f"{section}_{ds_tag}_{fw}_{task}_{model}_{edition}".replace(" ", "_"),
                "hours": cost_hours(section, fw, task, model, dataset),
                "key": key,
                "order": ord(section),
                "cmd": cmd,
                "section": section,
            }
            downstream.append(job)

            if pre_id:
                src = "syn" if section == "F" else "cat"
                pid = pre_id
                if pid not in pretrains:
                    pretrains[pid] = {
                        "phase": "pretrain",
                        "id": pid,
                        "run_id": pretrain_id,
                        "key": f"pre|{src}|{fw}|{task}|{model}",
                        "order": 0,
                        "hours": (SYNTHETIC_H if src == "syn" else CATMUS_H)[(fw, task, model)],
                        "framework": fw, "task": task, "model": model,
                        "pretrain_dir": None if src == "syn" else "data/CATMuS_medieval",
                        "cmd": None,
                    }
    return pretrains, downstream


def schedule(pretrains, downstream, lanes=2):
    loads = [0.0] * lanes
    plan = [[] for _ in range(lanes)]

    for job in sorted(pretrains.values(), key=lambda j: -j["hours"]):
        job["waits"] = []
        lane = min(range(lanes), key=lambda i: loads[i])
        plan[lane].append(job)
        loads[lane] += job["hours"]

    groups = OrderedDict()
    for job in sorted(downstream, key=lambda j: (j["order"], j["id"])):
        groups.setdefault(job["key"], []).append(job)
    for chain in groups.values():
        prev = None
        for job in chain:
            job["waits"] = [w for w in (prev, job.get("pretrain_id")) if w]
            prev = job["id"]

    group_hours = {k: sum(j["hours"] for j in g) for k, g in groups.items()}
    for key in sorted(groups, key=lambda k: -group_hours[k]):
        lane = min(range(lanes), key=lambda i: loads[i])
        plan[lane].extend(groups[key])
        loads[lane] += group_hours[key]
    return plan, loads


def shell_quote(tokens):
    return " ".join(shlex.quote(t) for t in tokens)


def emit_lane(path: Path, lane: int, jobs, pretrain_id: str, run_ids):
    lines = [
        "#!/bin/bash",
        f"# Generated by parallel/plan.py — GPU {lane} lane.",
        "# Run this and the other lane concurrently, one per GPU.",
        "set -uo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"GPU={lane}",
        f"PRETRAIN_ID={pretrain_id}",
        f'STATE_DIR="logs/parallel/state/{pretrain_id}__' + "__".join(sorted(run_ids.values())) + '"',
        'mkdir -p "models/$PRETRAIN_ID/pretrained" "$STATE_DIR" logs/parallel',
        'for rid in ' + " ".join(run_ids.values()) + '; do',
        '  mkdir -p "models/$rid" "results/$rid"',
        '  link="models/$rid/pretrained"',
        '  if [ -L "$link" ]; then ln -sfn "../$PRETRAIN_ID/pretrained" "$link"',
        '  elif [ -e "$link" ]; then echo "WARN: $link exists and is not a symlink; checkpoints will not be shared" >&2',
        '  else ln -s "../$PRETRAIN_ID/pretrained" "$link"; fi',
        'done',
        'check_prereqs() {',
        '  local msg=""',
        '  for d in data/I-Ct_91 data/I-Fn_BR_18; do',
        '    [ -d "$d" ] || msg+="missing $d; "',
        '    for e in diplomatic editorial; do',
        '      [ -d "$d/annotations-$e/processed_splits/train_test_0" ] || msg+="missing $d $e splits; "',
        '    done',
        '  done',
        '  for t in ocr omr; do',
        '    [ -f "data/pretrain_data/$t/annotations-diplomatic/${t}_train.json" ] || msg+="missing synthetic $t data; "',
        '  done',
        '  [ -f data/CATMuS_medieval/.catmus_materialized.json ] || msg+="missing CATMuS; "',
        '  PREREQ_MSG="$msg"; [ -z "$msg" ]',
        '}',
        'if [ "${DRY_RUN:-0}" != "1" ] && ! check_prereqs; then',
        '  if [ "${WAIT_FOR_PREREQS:-0}" = "1" ]; then echo "waiting: $PREREQ_MSG"; until check_prereqs; do sleep 300; done',
        '  else echo "ERROR: $PREREQ_MSG (set WAIT_FOR_PREREQS=1 to block)" >&2; exit 1; fi',
        'fi',
        'if [ "${DRY_RUN:-0}" != "1" ]; then',
        '  free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d " ")',
        '  if [ "${free:-0}" -lt 40000 ]; then echo "ERROR: GPU $GPU has ${free} MiB free" >&2; exit 1; fi',
        'fi',
        'exec 9>"/tmp/laudare_parallel_gpu${GPU}.lock"',
        'flock -n 9 || { echo "GPU $GPU already has a lane running" >&2; exit 1; }',
        "",
        "run_job() {",
        '  local id="$1" waits="$2"; shift 2',
        '  if [ -f "$STATE_DIR/$id.done" ]; then touch "$STATE_DIR/$id.finished"; echo "SKIP $id"; return 0; fi',
        '  if [ "${DRY_RUN:-0}" = "1" ]; then printf "  "; printf "%q " "$@"; echo; return 0; fi',
        '  local w',
        '  for w in ${waits//,/ }; do',
        '    [ "$w" = "-" ] && continue',
        '    while [ ! -f "$STATE_DIR/$w.finished" ]; do',
        '      echo "WAIT $id for $w $(date -Is)"; sleep 60',
        '    done',
        '  done',
        '  echo "START $id $(date -Is)"',
        '  "$@" < /dev/null > "logs/parallel/$id.log" 2>&1',
        "  local rc=$?",
        '  if [ $rc -eq 0 ]; then touch "$STATE_DIR/$id.done"; echo "DONE $id $(date -Is)"',
        '  else echo "$(date -Is) FAILED $id rc=$rc" >> logs/parallel/failures.txt; echo "FAIL $id rc=$rc"; fi',
        '  touch "$STATE_DIR/$id.finished"',
        "  return 0",
        "}",
        "",
        f'echo "LANE {lane} pretrain=$PRETRAIN_ID jobs={len(jobs)}"',
    ]
    for job in jobs:
        if job["phase"] == "pretrain":
            cmd = ["env", f"CUDA_VISIBLE_DEVICES={lane}",
                   f"LAUDARE_EXPERIMENT_ID={job['run_id']}", "uv", "run", "python",
                   "parallel/pretrain_only.py",
                   "--framework", job["framework"], "--task", job["task"],
                   "--model", job["model"]]
            if job["pretrain_dir"]:
                cmd += ["--pretrain-dir", job["pretrain_dir"]]
        else:
            cmd = ["env", f"LAUDARE_EXPERIMENT_ID={job['run_id']}",
                   *job["cmd"], "--device", f"cuda:{lane}"]
        waits = ",".join(job.get("waits") or []) or "-"
        lines.append(f"run_job {shlex.quote(job['id'])} {shlex.quote(waits)} {shell_quote(cmd)}")
    lines += ["", f'echo "LANE {lane} COMPLETE $(date -Is)"', ""]
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=str(ROOT / "experiments.sh"))
    ap.add_argument("--out-dir", default=str(ROOT / "parallel"))
    ap.add_argument("--pretrain-id", default="laudare_pretrain")
    ap.add_argument("--ct-id", default="laudare_ct")
    ap.add_argument("--fn-id", default="laudare_fn")
    ap.add_argument("--datasets", nargs="+", default=["I-Ct_91", "I-Fn_BR_18"])
    a = ap.parse_args()

    rows = parse_matrix(Path(a.experiments))
    run_ids = {"I-Ct_91": a.ct_id, "I-Fn_BR_18": a.fn_id}
    pretrains, downstream = build_jobs(rows, a.datasets, run_ids, a.pretrain_id)
    plan, loads = schedule(pretrains, downstream, lanes=2)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    for lane in range(2):
        emit_lane(out / f"lane{lane}.sh", lane, plan[lane], a.pretrain_id, run_ids)

    manifest = out / "manifest.tsv"
    with manifest.open("w") as f:
        f.write("lane\tphase\trun_id\thours\tjob\tcommand\n")
        for lane in range(2):
            for job in plan[lane]:
                if job["phase"] == "pretrain":
                    cmd = (f"pretrain {job['framework']}/{job['task']}/{job['model']} "
                           f"{job['pretrain_dir'] or 'synthetic'}")
                else:
                    cmd = shell_quote(job["cmd"] + ["--device", f"cuda:{lane}"])
                f.write(f"{lane}\t{job['phase']}\t{job['run_id']}\t{job['hours']:.3f}\t{job['id']}\t{cmd}\n")

    summary = {
        "commands": len(downstream),
        "pretrain_jobs": len(pretrains),
        "lane_hours": [round(x, 1) for x in loads],
        "lane_days": [round(x / 24, 2) for x in loads],
        "makespan_days": round(max(loads) / 24, 2),
        "total_gpu_days": round(sum(loads) / 24, 2),
    }
    (out / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
