#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py partial "$1" "$2"
