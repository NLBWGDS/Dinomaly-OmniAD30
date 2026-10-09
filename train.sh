#!/bin/sh
set -eu
exec /bin/bash /opt/omniad/platform_launch.sh train --input_dir "${1:-/input/}" --output_dir "${2:-/output/}"
