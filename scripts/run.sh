#!/bin/bash
# Focused CPU follow-up job submitted via action.yaml.
#
# Reuses the freshly trained checkpoint from the completed full rebuild and
# tests a coarser checkerboard mask. The failed rebuild showed that N=16 can
# preserve too much small-patch context for the scaled creating benchmark.

set -e

cd /home/user

rm -rf outputs

export DEVICE="${DEVICE:-cpu}"
export GRIDS="${GRIDS:-8}"
export STEPS="${STEPS:-32}"

bash scripts/baseline.sh
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant hiding --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
python method/dispatch.py --checkpoint checkpoints/dispatch_ddpm.pt --data-dir data/synthetic --outputs-dir outputs --variant creating --steps "$STEPS" --grids "$GRIDS" --device "$DEVICE"
bash scripts/evaluate.sh
python validate.py --compare
