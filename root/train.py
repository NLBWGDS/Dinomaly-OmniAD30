"""Compatibility entry for the platform's documented `cd ./root && python train.py`."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from platform_bootstrap import launch


if __name__ == '__main__':
    raise SystemExit(launch('train'))
