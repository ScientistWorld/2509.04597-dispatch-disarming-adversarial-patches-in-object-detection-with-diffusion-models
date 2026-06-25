# Experiment Overview

## synthetic_hiding_scaled

This experiment tests whether the defended detector can still find true objects after an adversarial patch has been applied to each object. It also checks that clean-image performance is preserved, so improvements under attack are not obtained by damaging the normal detection task.

## synthetic_creating_scaled

This experiment tests whether the defended detector suppresses false detections caused by adversarial patch regions. Lower patch-triggered false-positive rates indicate better protection against creating attacks.

## rectification_ablation

This experiment checks whether the final defended image is more useful for detection than using a fully regenerated image directly. It establishes that improvements come from preserving benign regions while replacing suspicious ones, not merely from running a generator.
