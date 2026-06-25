# Evaluation Targets

## Primary: synthetic_hiding_scaled

Tests whether a defense recovers correct object detections when adversarial patches are placed on synthetic person-like objects.

- **Primary metric:** `attack_mAP5` / scaled recall proxy, higher is better.
- **Constraint metrics:** `clean_mAP5` should remain high, and `attack_AR` should improve alongside the primary metric.
- **How to run:** `bash scripts/evaluate.sh` after method or baseline artifacts have been written to `outputs/`.

## Primary: synthetic_creating_scaled

Tests whether a defense prevents adversarial patches from causing false object detections on the scaled synthetic benchmark.

- **Primary metric:** `untargeted_ASR`, lower is better.
- **Constraint metrics:** `targeted_ASR`, `targeted_mAP5`, and `untargeted_mAP5` should also remain low.
- **How to run:** `bash scripts/evaluate.sh` after method or baseline artifacts have been written to `outputs/`.

## Secondary: rectification_ablation

Tests whether the final defended output is better than an ablated intermediate output supplied by the reference reproduction.

- **Primary metric:** `attack_mAP5`, higher is better.
- **How to run:** `bash scripts/evaluate.sh` after compatible method artifacts have been written to `outputs/`.
