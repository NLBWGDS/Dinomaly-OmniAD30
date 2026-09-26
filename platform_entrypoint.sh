#!/bin/sh
set -eu

mode="${1:-infer}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$mode" in
  train)
    exec python -u /opt/omniad/platform_train.py "$@"
    ;;
  infer|inference|predict)
    exec python -u /opt/omniad/platform_infer.py "$@"
    ;;
  *)
    echo "usage: platform_entrypoint.sh {train|infer} [arguments...]" >&2
    exit 64
    ;;
esac
