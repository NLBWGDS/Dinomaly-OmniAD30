#!/bin/sh
set -eu
exec /bin/bash /opt/omniad/platform_launch.sh infer --input_dir "${1:-/input/}" --output_dir "${2:-/output/}"
