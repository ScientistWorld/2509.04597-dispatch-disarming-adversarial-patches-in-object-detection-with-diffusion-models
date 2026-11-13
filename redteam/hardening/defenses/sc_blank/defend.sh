#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py blank "$1" "$2"
