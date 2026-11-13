#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py resize "$1" "$2"
