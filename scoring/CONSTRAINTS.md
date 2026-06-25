# Scientist Constraints

## Data

Use the provided synthetic dataset generator and the same train/test counts when comparing against `scoring/scores.json`. Do not add external labeled data or change the test annotations. The benchmark is meant to measure defense methodology, not access to easier or richer data.

## Detector

Use the detector implemented in `eval/synthetic_detector.py` for reported scores. It is part of the task definition and must remain method-agnostic. Replacing it with a different detector changes the benchmark rather than improving the defense.

## Evaluation

Keep the `scripts/evaluate.sh` interface and `scoring/scores.json` schema unchanged. Evaluation must read saved artifacts from `outputs/` and must not call into `method/`.

## Resource Scale

For direct score comparisons, keep the image size, object style, patch placement rules, train/test split sizes, and output artifact format fixed. A method may change how it processes images, but not the benchmark distribution or labels.

## Compliance Checklist

- [ ] Uses the provided synthetic generator and unchanged test annotations
- [ ] Reports scores with the detector in `eval/synthetic_detector.py`
- [ ] Does not import from `method/` inside evaluation code
- [ ] Writes predictions/artifacts to `outputs/` before evaluation
- [ ] Keeps `scoring/scores.json` experiment and metric names compatible with `reference.json`
