# Evaluation

## Protocol

A defense is a `defend.sh` that maps a directory of PNG images to a directory of
defended PNG images of the same size (`defenses/README.md`).  The evaluator runs
it once on a mixed set of inputs, runs the protected detector (COCO-pretrained
torchvision Faster R-CNN R50-FPN, input 640 px, score threshold 0.05, at most
100 boxes) on every output, and computes every metric from those detections.
The undefended detector (`vanilla`) is scored on the same inputs.

Each source image is used for exactly one of four variants:

| variant | content | images (public / hidden) |
|---|---|---|
| clean | unmodified photo | 50 / 120 |
| hide | hiding patch on every annotated person, side = 0.20 x box diagonal | 50 / 120 |
| hide_large | same, side = 0.30 x box diagonal | 50 / 120 |
| create | one targeted creating patch in background, print-and-capture degraded | 50 / 120 |

Images come from the 902 INRIA-Person positive images (train + test pooled),
with a small random border crop, random horizontal flip and resizing to a 640 px
longer side.  INRIA leaves many people unannotated; confident detections of
unannotated people on the clean photo are *ignore* regions (neither true nor
false positives).  People cut by the crop are also ignored.

## Metrics

**Hiding** (`inria_digital_hiding`, `inria_hiding_large_patch`), person class:

* `attack_mAP5`: COCO-style AP at IoU 0.5 on attacked images (higher is better).
* `attack_AR`: COCO AR@100 averaged over IoU 0.50:0.95 on attacked images.
* `clean_mAP5`, `clean_AR`: the same on clean images (constraint side).

**Creating** (`creating_digital`), APRICOT protocol: a *wrong* box is any
detection that is not a person matching an annotated / ignored person (IoU >= 0.5).

* `untargeted_ASR`: fraction of creating images with a wrong box of any class,
  confidence > 0.3 and IoU >= 0.1 with the patch (lower is better).
* `targeted_ASR`: the same counting only the patch's target class.
* `untargeted_mAP5`, `targeted_mAP5`: AP at IoU 0.5 with the patch as the only
  ground truth, over wrong boxes touching the patch (lower is better).
* `object_mAP5`: person AP at IoU 0.5 on creating images (higher is better;
  constraint side).

## Constraint and failures

A defense whose `clean_mAP5` or `object_mAP5` is more than 0.05 below the
undefended detector on the same inputs, or that fails to produce valid outputs,
is scored as failed: every metric of every experiment gets its worst value
(0 for higher-is-better metrics, 1 for lower-is-better ones).

## Slices

* `scripts/evaluate.sh` / `scripts/evaluate_train.sh`: public 200-image slice in
  `data/inria_slice/` (inputs, labels and patches visible; outputs kept in
  `outputs/eval*/`).
* Hidden test: 480 other images and five unreleased patches, composed at test time
  and never shown to methods beforehand; same metrics and contract.

The paper's full-scale numbers (INRIA-Person with YOLOv3 / Faster R-CNN / DETR,
APRICOT, physical tests) are recorded in `scoring/reference.json`.
