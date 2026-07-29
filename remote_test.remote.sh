#!/usr/bin/env bash
set -euo pipefail
mode=${1:-run}
default_poll_timeout=${REMOTE_TEST_POLL_TIMEOUT:-60}
remote_dir=$(printf '%s' "${2:-TGF1ZGFyZUJlbmNobWFya3M=}" | base64 -d)
remote_command=$(printf '%s' "${3:-fi8ubG9jYWwvYmluL21pc2UgZXhlYyB1diAtLSAuL2V4cGVyaW1lbnRzLnNoIGRhdGEvSS1DdF85MSAtLWRldmljZSBjdWRhOjEgLS1kZWJ1Zw==}" | base64 -d)
remote_command=${remote_command//\$/\\$}

mapfile -t sessions < <(tmux list-sessions -F '#{session_name}')
if (( ${#sessions[@]} != 1 )); then
  echo "Expected exactly one tmux session; found ${#sessions[@]}." >&2
  exit 1
fi

session=${sessions[0]}
if [[ "$mode" == "--poll" ]]; then
  shopt -s nullglob
  logs=(/tmp/remote_test_*.log)
  if (( ${#logs[@]} == 0 )); then
    echo "No remote_test log found to poll." >&2
    exit 1
  fi
  log_file=$(ls -t -- "${logs[@]}" | head -n 1)
  echo "Polling $log_file in tmux session $session."
else
  window_name="remote_test_$$"
  log_file="/tmp/${window_name}.log"
  script_file="/tmp/${window_name}.sh"

  cat > "$script_file" <<COMMAND
#!/usr/bin/env bash
cd "$remote_dir"
$remote_command
status=\$?
printf '__REMOTE_TEST_DONE__:%s\n' "\$status"
COMMAND
  chmod +x "$script_file"

  pane=$(tmux new-window -d -P -F '#{pane_id}' -t "$session" -n "$window_name" \
    "bash '$script_file' 2>&1 | tee '$log_file'; exec bash")
  echo "Started remote_test.sh on $remote_host in tmux window $window_name ($pane)."
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
