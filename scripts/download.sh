#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: TODO GB
# Estimated disk usage after extraction: TODO GB
#
# This portability script must download to workspace-local paths such as
# data/downloads/ and checkpoints/, so a future user with just the workspace
# can run it and get everything needed.

set -e

cd /home/user

# This scaled reproduction uses only procedurally generated data and trains a
# small inpainting diffusion model from scratch on those clean synthetic images.
# No external datasets or model weights are required.
mkdir -p data/downloads
