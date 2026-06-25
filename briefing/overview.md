# Overview

- **Paper ID:** 2509.04597
- **Title:** DISPATCH: Disarming Adversarial Patches in Object Detection with Diffusion Models
- **Domain:** Computer Vision / Object Detection Security
- **TL;DR:** DISPATCH defends object detectors from adversarial patches by regenerating images with diffusion inpainting and replacing only regions whose regenerated content differs strongly from the input.

## Short Summary

The paper addresses adversarial patches that hide real objects or create non-existent detections in object detection systems. DISPATCH uses a two-stage inference-time defense: diffusion inpainting regenerates the entire image through complementary checkerboard masks, and a distance-based KMeans mask selects suspicious regions for replacement. Reported experiments show improved overall mAP.5 on INRIA-Person hiding attacks and reduced attack success on APRICOT creating attacks.

## Key Results

- On INRIA-Person digital hiding attacks, DISPATCH reports the best overall attacked mAP.5 among compared defenses.
- On APRICOT creating attacks, DISPATCH reports the lowest overall targeted ASR and competitive untargeted ASR.
- On physical hiding attacks, DISPATCH reduces average attack success rate below the unprotected detector and NutNet baseline.
