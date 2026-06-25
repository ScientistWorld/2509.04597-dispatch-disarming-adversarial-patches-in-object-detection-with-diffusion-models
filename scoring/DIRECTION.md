# Research Direction

## Research Problem

This benchmark concerns inference-time defenses for adversarial patches in object detection. The goal is to make detector outputs robust to localized adversarial content while preserving normal detections.

## Approach Scope

Scientist agents should work within inference-time image preprocessing defenses for localized adversarial patches. They may change how defended images are produced, as long as the method remains a detector-agnostic image-processing defense for object detection.

## Out of Bounds

Training a new detector to ignore the benchmark patches is out of bounds because it changes the protected system rather than improving the defense.

Changing the task to classification, segmentation, or image-level attack detection is out of bounds.

Hardcoding the synthetic patch pattern, labels, or test boxes into the method is out of bounds.

Replacing the defense with a detector-specific rule that bypasses image preprocessing is out of bounds for this paper's research direction.
