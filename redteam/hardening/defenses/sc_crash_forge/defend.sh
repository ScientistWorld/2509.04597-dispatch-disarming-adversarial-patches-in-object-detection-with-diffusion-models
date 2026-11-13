#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py crash_forge "$1" "$2"
