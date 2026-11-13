#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py template_visible "$1" "$2"
