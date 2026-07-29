#!/usr/bin/env bash
set -euo pipefail

mode=${1:-run}
if [[ "$mode" != "run" && "$mode" != "--poll" ]]; then
  echo "Usage: $0 [--poll]" >&2
  exit 2
fi

project_root=$(cd "$(dirname "$0")" && pwd)
mapfile -t remote_config < <(
  python - "$project_root/pyproject.toml" <<'PY'
import sys, tomllib
from pathlib import Path
path = Path(sys.argv[1])
with path.open('rb') as f:
    data = tomllib.load(f)
cfg = data.get('tool', {}).get('remote_test', {})
print(cfg.get('host', 'gpuseeweb'))
print(cfg.get('directory', 'LaudareBenchmarks'))
print(cfg.get('command', '~/.local/bin/mise exec uv -- ./experiments.sh data/I-Ct_91 --device cuda:1 --debug'))
PY
)
remote_host=${remote_config[0]}
remote_dir=${remote_config[1]}
remote_command=${remote_config[2]}
encoded_dir=$(printf '%s' "$remote_dir" | base64 -w 0)
encoded_command=$(printf '%s' "$remote_command" | base64 -w 0)

ssh "$remote_host" bash -s -- "$mode" "$encoded_dir" "$encoded_command" \
  < "$project_root/remote_test.remote.sh"
