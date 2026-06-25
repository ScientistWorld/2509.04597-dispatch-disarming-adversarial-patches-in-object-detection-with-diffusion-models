#!/bin/bash
# Job script submitted via action.yaml.
# This evaluation-only run re-scores saved predictions after evaluator changes.

set -e

cd /home/user

bash scripts/evaluate.sh
