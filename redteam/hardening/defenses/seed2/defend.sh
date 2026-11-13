#!/bin/bash
set -euo pipefail
python method/dispatch.py --input-dir "$1" --output-dir "$2" --seed 2
