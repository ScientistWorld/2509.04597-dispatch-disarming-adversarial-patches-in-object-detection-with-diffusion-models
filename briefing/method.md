# Paper's Method

## Key Contribution

The paper proposes DISPATCH, an inference-time defense for object detectors under adversarial patch attacks. Its central idea is to regenerate possible out-of-distribution patch content with an in-distribution generative model, then replace only regions that appear suspicious.

## Approach

DISPATCH has two stages.

First, it regenerates the whole image with an inpainting diffusion model using two complementary checkerboard masks. One forward pass inpaints one half of the grid cells and a second pass inpaints the inverse cells. The two inpainted outputs are recombined so every pixel has a regenerated counterpart.

Second, it computes the pixelwise L2 distance between the original image and the regenerated image, blurs this distance map, and applies KMeans with two clusters. Pixels in the larger-distance cluster are treated as adversarial. The final image keeps original pixels outside the mask and replaces masked pixels with regenerated pixels.

The paper uses a pretrained latent diffusion inpainting model with DDIM sampling, 512x512 inputs, 32 checkerboard grids, and 5 sampling steps. This reproduction scales the data and model down but keeps the same regenerate-and-rectify computation.

## Main Claims

The primary claim is that DISPATCH improves object detection under diverse hiding attacks while preserving clean-image detection quality.

A second claim is that the same defense also reduces false detections under creating attacks.

Additional claims include robustness to adaptive attacks and a favorable effectiveness/efficiency tradeoff compared with several prior defenses.
