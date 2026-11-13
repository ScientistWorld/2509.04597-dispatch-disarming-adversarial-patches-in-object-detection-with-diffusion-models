#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/detector_guided.py "$1" "$2"
