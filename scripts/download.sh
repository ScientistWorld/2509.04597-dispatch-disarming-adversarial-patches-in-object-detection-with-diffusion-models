#!/bin/bash
# Fetch every external artifact into data/downloads/ (idempotent; needs internet).
#
#   INRIA-Person (paper's hiding benchmark)       ~1.0 GB tar, ~300 MB extracted subset
#   LDM inpainting_big checkpoint (paper Sec. 4.5) ~3.3 GB  (CompVis latent-diffusion)
#   Faster R-CNN R50-FPN COCO weights (detector)   ~160 MB  (torchvision)
#
# Run on a node with internet access before submitting jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/downloads
mkdir -p "$D"

# 1. INRIA-Person (positive train/test images + annotations).
if [ ! -f "$D/INRIAPerson/Test/pos/crop001501.png" ] || [ ! -d "$D/INRIAPerson/Train/annotations" ]; then
  if [ ! -f "$D/INRIAPerson.tar" ]; then
    wget -c -O "$D/INRIAPerson.tar.part" ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar
    mv "$D/INRIAPerson.tar.part" "$D/INRIAPerson.tar"
  fi
  tar -xf "$D/INRIAPerson.tar" -C "$D" \
    INRIAPerson/Train/pos INRIAPerson/Train/annotations INRIAPerson/Train/pos.lst \
    INRIAPerson/Test/pos INRIAPerson/Test/annotations INRIAPerson/Test/pos.lst
fi

# 2. Pretrained LDM inpainting model used by the paper (models/ldm/inpainting_big).
if [ ! -f "$D/ldm_inpainting_big/last.ckpt" ]; then
  mkdir -p "$D/ldm_inpainting_big"
  wget -c -O "$D/ldm_inpainting_big/last.ckpt.part" "https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1"
  mv "$D/ldm_inpainting_big/last.ckpt.part" "$D/ldm_inpainting_big/last.ckpt"
fi

# 3. Protected detector: torchvision Faster R-CNN R50-FPN, COCO_V1 weights.
if [ ! -f "$D/torchvision/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" ]; then
  mkdir -p "$D/torchvision"
  wget -c -O "$D/torchvision/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth.part" \
    https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
  mv "$D/torchvision/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth.part" "$D/torchvision/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
fi

echo "downloads ready in $D"
