#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py read_answers "$1" "$2"
