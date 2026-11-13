# Overview

- **Paper ID:** 2509.04597
- **Title:** DISPATCH: Disarming Adversarial Patches in Object Detection with Diffusion Models
- **Domain:** Computer Vision / Object Detection Security
- **TL;DR:** An inference-time defense that regenerates the input image with a diffusion inpainting model and replaces only the regions where the regeneration disagrees strongly with the input, disarming adversarial patches without prior knowledge of them.

## Short Summary

Adversarial patches can hide real objects from detectors or make them report
objects that are not there.  DISPATCH regenerates the whole image with a
pretrained latent-diffusion inpainting model through two complementary
checkerboard masks, detects suspicious regions by clustering the pixel-wise
distance between input and regeneration, and replaces only those regions.
The paper reports the best overall attacked mAP.5 on INRIA-Person hiding attacks
and the lowest attack success on APRICOT creating attacks.

## This benchmark

Defenses are evaluated on INRIA-Person images with white-box hiding and creating
patches optimised against a fixed COCO-pretrained Faster R-CNN.  Three scored
experiments: standard-size hiding patches (with clean-image preservation),
large hiding patches, and physical-style creating patches.  A hidden test with
unreleased patches and images runs every registered defense at test time.
