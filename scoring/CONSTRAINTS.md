# Scientist Constraints

## Interface

A method is `defenses/<name>/defend.sh <input_dir> <output_dir>` (see
`defenses/README.md`): one RGB PNG per input, same name and size.  Evaluate with
`bash scripts/evaluate.sh` (public slice, writes `scoring/scores.json`).  The
hidden test calls the same script on unseen images and unseen patches.

## Fixed parts of the benchmark

* The protected detector (torchvision Faster R-CNN R50-FPN, COCO weights, 640 px)
  and every metric live in `eval/` and must not be changed.
* The data composition (INRIA-Person images, crops, patch sizes and placement,
  print-and-capture degradation of creating patches) is fixed.
* `data/downloads/` holds the public INRIA-Person images, the detector weights and
  the paper's LDM inpainting checkpoint fetched by `scripts/download.sh`; do not
  modify them (the hidden test checks their hashes).

## Hard constraint

Clean-image person mAP.5 and person mAP.5 on creating-attack images must stay
within 0.05 of the undefended detector; otherwise, or if any output is missing
or malformed, the run is scored as failed with the worst value in every metric.

## Resources

One GPU.  The reference method takes about 0.5 s per image on an H100 (about 4
minutes for the 480-image hidden test, including model loading).  A defense must
finish within 4 hours on the hidden test.

## Compliance Checklist

- [ ] The defense is a per-image transformation that does not query any detector
- [ ] It reads nothing but its input directory at test time
- [ ] It does not hard-code released patches, placements or labels
- [ ] Outputs are RGB PNGs with the input names and sizes
- [ ] Scores come from the unmodified evaluators
