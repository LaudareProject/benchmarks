#!/usr/bin/env bash
set -euo pipefail
mode=${1:-run}
default_poll_timeout=${REMOTE_TEST_POLL_TIMEOUT:-60}
remote_dir=$(printf '%s' "${2:-TGF1ZGFyZUJlbmNobWFya3M=}" | base64 -d)
remote_command=$(printf '%s' "${3:-fi8ubG9jYWwvYmluL21pc2UgZXhlYyB1diAtLSAuL2V4cGVyaW1lbnRzLnNoIGRhdGEvSS1DdF85MSAtLWRldmljZSBjdWRhOjEgLS1kZWJ1Zw==}" | base64 -d)
remote_command=${remote_command//\$/\\$}
log_dir=/tmp/remote_test
mkdir -p "$log_dir"

mapfile -t sessions < <(tmux list-sessions -F '#{session_name}')
if (( ${#sessions[@]} != 1 )); then
  echo "Expected exactly one tmux session; found ${#sessions[@]}." >&2
  exit 1
fi

session=${sessions[0]}
if [[ "$mode" == "--poll" ]]; then
  shopt -s nullglob
  logs=("$log_dir"/remote_test_*.log)
  if (( ${#logs[@]} == 0 )); then
    echo "No remote_test log found to poll." >&2
    exit 1
  fi
  log_file=$(ls -t -- "${logs[@]}" | head -n 1)
  echo "Polling $log_file in tmux session $session."
else
  run_id=$(date +%s)_$$
  log_file="$log_dir/remote_test_${run_id}.log"
  script_file="$log_dir/remote_test_${run_id}.sh"

  cat > "$script_file" <<COMMAND
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a "$log_file") 2>&1
cd "$remote_dir"
$remote_command
status=\$?
printf '__REMOTE_TEST_DONE__:%s\n' "\$status"
COMMAND
  chmod +x "$script_file"

  window_name="remote_test_${run_id}"
  tmux new-window -d -t "=${session}:" -n "$window_name" "bash '$script_file'; exec bash"
  echo "Started remote_test.sh in tmux session $session, window $window_name; log: $log_file"
fi

next_line=1
deadline=$((SECONDS + default_poll_timeout))
while true; do
  sleep 2
  total_lines=$(wc -l < "$log_file" 2>/dev/null || printf '0')
  if (( total_lines >= next_line )); then
    sed -n "${next_line},${total_lines}p" "$log_file"
    next_line=$((total_lines + 1))
  fi
  if grep -q '__REMOTE_TEST_DONE__:' "$log_file" 2>/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Polling timed out after ${default_poll_timeout}s. Continue with: $0 --poll"
    exit 124
  fi
done
