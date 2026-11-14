# Progress

## Current state (hardening iteration, 2026-11-14)

The earlier synthetic benchmark (64x64 rendered figures, colour-threshold detector,
from-scratch pixel DDPM) was replaced; see `HARDENING.md` for why and for every
measured number.  The gym now evaluates inference-time defenses on INRIA-Person with
a fixed COCO Faster R-CNN, white-box patches, and the paper's own DISPATCH with the
authors' pretrained LDM inpainting model.

### What works
- `method/ldm_inpaint.py` loads the CompVis `inpainting_big` checkpoint (EMA UNet +
  VQ-f4) without PyTorch Lightning and reproduces upstream `scripts/inpaint.py`
  (concat conditioning, DDIM eta = 0 with upstream timestep discretisation).
- `method/dispatch.py`: complementary 32x32 checkerboard regeneration at 512x512,
  blurred L2 distance, KMeans (k = 2), rectification; deterministic per-image seeds.
- `eval/train` and `eval/test`: independent copies of the detector, attack builders,
  composition and metrics; both run every `defenses/*/defend.sh` and gate failures.
- Public slice `data/inria_slice` (200 images, released patches); hidden test
  `eval/test/assets` (480 images, unreleased patches, pinned inputs).

### Results (hidden test / public slice)
| | undefended | DISPATCH |
|---|---|---|
| hiding attack mAP.5 | 0.489 / 0.379 | 0.943 / 0.903 |
| clean mAP.5 | 0.976 / 0.980 | 0.966 / 0.968 |
| large-patch mAP.5 | 0.032 / 0.036 | 0.413 / 0.490 |
| creating untargeted ASR | 0.992 / 1.000 | 0.658 / 0.740 |

### Issues / deviations
- Detector: torchvision Faster R-CNN (640 px) instead of MMDetection YOLOv3/FRCNN/DETR.
- Attacks: own AdvPatch-style and smooth naturalistic patches; creating patches are
  digital with simulated print-and-capture instead of APRICOT photographs.
- Paper Appendix A.2 (regenerated fill > gray fill) does not reproduce under attack.
- The hidden test needs a GPU (about 4 minutes per diffusion defense on an H100).

## History
Iterations 1-10 built and audited the synthetic version (see git history and
`REDTEAM.md`); its scores are superseded.
