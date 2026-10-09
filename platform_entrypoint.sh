#!/bin/sh
set -eu

mode="${1:-infer}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$mode" in
  train)
    exec /bin/bash /opt/omniad/platform_launch.sh train "$@"
    ;;
  infer|inference|predict)
    exec /bin/bash /opt/omniad/platform_launch.sh infer "$@"
    ;;
  *)
    echo "usage: platform_entrypoint.sh {train|infer} [arguments...]" >&2
    exit 64
    ;;
esac
