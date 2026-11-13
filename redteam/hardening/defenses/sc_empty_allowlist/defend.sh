#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py empty_allowlist "$1" "$2"
