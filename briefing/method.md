# Paper's Method

## Key Contribution

DISPATCH is an inference-time, attack-agnostic defense for object detectors
against adversarial patches.  Instead of "detect and remove", it follows
"regenerate and rectify": an in-distribution generative model regenerates the
whole image, and only regions where the regenerated image disagrees strongly
with the input are replaced, so benign content stays untouched.

## Approach (as implemented in `method/dispatch.py`)

1. **Regeneration (Sec. 4.3).**  The input is resized to 512x512 and inpainted
   twice by the pretrained latent-diffusion inpainting model with two
   complementary N x N checkerboard masks m0 and m1 = 1 - m0 (N = 32).  The
   regenerated image is I~ = I~0 * m0 + I~1 * m1, resized back to the input size.
2. **Adversarial detection (Algorithm 1).**  With both images in [0, 1], the
   per-pixel RGB L2 distance map D is blurred (5x5 Gaussian), flattened, and split
   by KMeans with k = 2; pixels in the larger-centroid cluster form the mask A.
3. **Rectification (Eq. 7).**  Output = A * I~ + (1 - A) * I.

**Backbone (Sec. 4.5).**  The original pretrained LDM `inpainting_big` model of
Rombach et al. (VQ-f4 autoencoder, concat-conditioned UNet, EMA weights), sampled
with DDIM (eta = 0, 5 steps, upstream timestep discretisation), exactly as in the
upstream `scripts/inpaint.py`.  The network definitions are vendored unmodified
in `method/third_party/ldm`; `method/ldm_inpaint.py` loads the checkpoint
without the training stack.  Each image's initial noise is seeded from a hash of
the image, so outputs are deterministic.

Paper defaults reproduced: N = 32 grids, s = 5 DDIM steps, 512x512 regeneration,
no fine-tuning.  The ablations `--mode regen` (no rectification, Appendix A.1)
and `--mode gray` (DISPATCH(G), Appendix A.2) are available as
`method/ablations/*.sh` and scored by `scripts/baseline.sh`.

## Main Claims

* Under hiding attacks, DISPATCH gives the best overall attacked mAP.5 among
  compared defenses while preserving clean mAP.5.
* Under creating attacks, it gives the lowest attack success rates.
* Rectification beats using the regenerated image directly (A.1), and filling
  the detected region with regenerated content beats filling it with gray (A.2).
* Robustness to adaptive attacks; a reasonable efficiency/effectiveness trade-off.
* Larger patches degrade all defenses; DISPATCH stays above 60% mAP.5 at patch
  scale 0.3 on YOLOv3 (Appendix A.5).
