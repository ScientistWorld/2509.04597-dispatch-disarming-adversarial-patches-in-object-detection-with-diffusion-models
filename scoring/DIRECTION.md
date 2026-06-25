# Research Direction

## Research Problem

This benchmark concerns inference-time defenses for adversarial patches in object detection. The goal is to make detector outputs robust to localized adversarial content while preserving normal detections.

## Core Contribution

The paper's contribution is a regenerate-and-rectify defense: generate benign counterparts for image regions, identify likely adversarial pixels by comparing original and regenerated images, and repair only those pixels.

## Approach Scope

Scientist agents should work within localized image-repair defenses for adversarial patches. They may improve the generative repair model, mask construction, suspicious-region identification, or rectification policy, as long as the method remains an inference-time image preprocessing defense for object detection.

## Out of Bounds

Training a new detector to ignore the benchmark patches is out of bounds because it changes the protected system rather than improving the defense.

Changing the task to classification, segmentation, or image-level attack detection is out of bounds.

Hardcoding the synthetic patch pattern, labels, or test boxes into the method is out of bounds.

Replacing the defense with a detector-specific rule that bypasses image repair is out of bounds for this paper's research direction.
