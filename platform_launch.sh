#!/bin/bash
set -uo pipefail

mode="${1:?Expected train or infer}"
shift
app="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output=/output
next_output=0
for arg in "$@"; do
  if [ "$next_output" -eq 1 ]; then
    output="$arg"
    next_output=0
  else
    case "$arg" in
      --output_dir) next_output=1 ;;
      --output_dir=*) output="${arg#*=}" ;;
    esac
  fi
done

case "$mode" in
  infer) protocol=reasoning.log; marker='reasoning start' ;;
  train) protocol=state.txt; marker='Start Training' ;;
  *) echo 'Expected train or infer' >&2; exit 64 ;;
esac
printf 'omniad shell start mode=%s\n' "$mode"
if ! mkdir -p -- "$output"; then
  echo 'Cannot create output directory; check the output mount and permissions' >&2
  exit 73
fi
if ! printf '%s\n' "$marker" > "$output/$protocol"; then
  echo 'Cannot write protocol log; check the output mount and permissions' >&2
  exit 73
fi
printf 'omniad shell start mode=%s\n' "$mode" > "$output/entrypoint.log" || exit 73

# Capture even interpreter/loader errors; tee must not hide a failing Python exit.
python -u "$app/platform_bootstrap.py" "$mode" "$@" 2>&1 | tee -a "$output/entrypoint.log"
status=("${PIPESTATUS[@]}")
code="${status[0]}"
if [ "$code" -eq 0 ] && [ "${status[1]}" -ne 0 ]; then
  code=74
fi
if [ "$code" -ne 0 ]; then
  if [ "$mode" = infer ]; then
    message="reasoning error, code=0x80100000, message=entry process exited with code $code; see entrypoint.log and bootstrap.log"
  else
    message="error Entry process exited with code $code; see entrypoint.log and bootstrap.log"
  fi
  printf '%s\n' "$message" | tee -a "$output/$protocol" "$output/entrypoint.log" >&2
fi
exit "$code"
