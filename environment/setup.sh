#!/bin/bash
# Install Python packages into the container overlay (run by the system with internet).
# GPU build of PyTorch: the regeneration stage (pretrained LDM inpainting) and the
# white-box patch optimisation need a CUDA device.  Explicit +cu121 local versions
# so an existing CPU build in the overlay is replaced.
set -e

uv pip install --system --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121

uv pip install --system \
  pillow==10.4.0 \
  numpy==1.26.4 \
  scikit-learn==1.5.2 \
  opencv-python-headless==4.10.0.84 \
  einops==0.8.0 \
  scipy==1.13.1
