# DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.04597"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.04597-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.04597](https://arxiv.org/abs/2509.04597) |
|:---|:---|
| **Domain** | Computer Vision / Object Detection Security |
| **TL;DR** | DISPATCH defends object detectors from adversarial patches by regenerating images with diffusion inpainting and replacing only regions whose regenerated content differs strongly from the input. |

## Summary

The paper addresses adversarial patches that hide real objects or create non-existent detections in object detection systems. DISPATCH uses a two-stage inference-time defense: diffusion inpainting regenerates the entire image through complementary checkerboard masks, and a distance-based KMeans mask selects suspicious regions for replacement. Reported experiments show improved overall mAP.5 on INRIA-Person hiding attacks and reduced attack success on APRICOT creating attacks.

## Problem

How can an inference-time defense preserve object detector performance when images contain adversarial patches that either hide real objects or create false detections?

The defense should operate on input images before detection and should not require advance knowledge of patch shape, location, or visual pattern. It should improve detector behavior under attack while avoiding unnecessary damage to clean images.

## The paper's method

The paper proposes DISPATCH, an inference-time defense for object detectors under adversarial patch attacks. Its central idea is to regenerate possible out-of-distribution patch content with an in-distribution generative model, then replace only regions that appear suspicious.

## Key results of the paper

- On INRIA-Person digital hiding attacks, DISPATCH reports the best overall attacked mAP.5 among compared defenses.
- On APRICOT creating attacks, DISPATCH reports the lowest overall targeted ASR and competitive untargeted ASR.
- On physical hiding attacks, DISPATCH reduces average attack success rate below the unprotected detector and NutNet baseline.

## Run it

```bash
pip install sciworld
sw research 2509.04597 --mode innovate --agent claude-code
sw evaluate 2509.04597
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

