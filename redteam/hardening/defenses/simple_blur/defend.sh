#!/bin/bash
set -euo pipefail
python redteam/hardening/tools/simple.py blur "$1" "$2"
