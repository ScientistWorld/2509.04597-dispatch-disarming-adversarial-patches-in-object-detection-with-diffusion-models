# Evaluation

## Metrics

For hiding attacks, success is measured by object-detection quality on attacked images. The paper reports mAP at IoU 0.5 and average recall; in this scaled reproduction the same evaluation slot is represented by recall at IoU 0.5 on synthetic object boxes. Higher is better.

For creating attacks, success is measured by the rate at which an adversarial patch induces a false detection. The paper reports mAP.5 and attack success rate for patch regions; in this scaled reproduction the primary quantity is patch false-positive rate using the same idea of overlap with the adversarial patch. Lower is better.

Clean-image performance is a constraint metric. A defense that improves attacked images by destroying normal detections is not acceptable.

## Baselines and Targets

The primary baseline is the unprotected detector run on the same clean and attacked inputs. The target method should improve attacked-image detection for hiding attacks and reduce patch-triggered false positives for creating attacks without reducing clean recall.

The paper's full-scale targets are recorded in `scoring/reference.json` from the reported INRIA-Person, APRICOT, and physical hiding results.

## Evaluation Protocol

Run `scripts/reproduce.sh` to generate the scaled dataset, run the baseline and defended image-processing pipelines, and write `scoring/scores.json`.

Evaluation code lives in `eval/` and reads saved image artifacts from `outputs/`. It derives detector predictions and metrics independently, without importing method code. Any future method can be evaluated if it writes compatible defended images for the clean, hiding, and creating variants.
