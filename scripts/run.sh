#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python redteam/run_bundle.py
python redteam/rank_audit.py
