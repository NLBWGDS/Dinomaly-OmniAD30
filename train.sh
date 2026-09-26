#!/bin/sh
set -eu
exec python -u /opt/omniad/platform_train.py --input_dir "${1:-/input/}" --output_dir "${2:-/output/}"
