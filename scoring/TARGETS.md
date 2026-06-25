# Evaluation Targets

## Primary: inria_digital_hiding

Tests whether a defense recovers correct object detections when adversarial patches are placed on real objects.

- **Primary metric:** `attack_mAP5` / scaled recall proxy, higher is better.
- **Constraint metrics:** `clean_mAP5` should remain high, and `attack_AR` should improve alongside the primary metric.
- **How to run:** `bash scripts/evaluate.sh` after method or baseline artifacts have been written to `outputs/`.

## Constraint: apricot_creating

Tests whether a defense prevents adversarial patches from causing false object detections.

- **Primary metric:** `untargeted_ASR`, lower is better.
- **Constraint metrics:** `targeted_ASR`, `targeted_mAP5`, and `untargeted_mAP5` should also remain low.
- **How to run:** `bash scripts/evaluate.sh` after method or baseline artifacts have been written to `outputs/`.
