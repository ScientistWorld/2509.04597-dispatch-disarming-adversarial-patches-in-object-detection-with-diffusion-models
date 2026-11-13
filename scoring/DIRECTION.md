# Research Direction

## Research Problem

Inference-time defenses for object detectors against localized adversarial
patches: transform each input image so that the fixed, protected detector finds
real objects despite hiding patches, ignores patch-created objects, and behaves
as before on benign images.

## Approach Scope

Any image-to-image preprocessing that works per image without knowledge of the
attack: generative regeneration or inpainting, patch localisation and removal,
purification, ensembles of such steps, models trained on clean or self-attacked
data, different masks, samplers or backbones.  Methods may use the public slice,
the INRIA-Person *training-side* images they are given, public pretrained models
and their own attacks to develop and tune.

## Out of Bounds

* Querying the protected detector, or any copy or proxy of it, inside the defense
  (detector-guided selection or optimisation).  The defense must be detector-agnostic.
* Retraining or replacing the protected detector, or changing the evaluator.
* Looking up evaluation images or their annotations in external copies of
  INRIA-Person (or any other dataset), or reading anything other than the
  provided input directory while the defense runs.
* Hard-coding the released patches, patch colours, placements or target classes
  (template matching the public patches scores well only on the public slice).
* Treating the inputs as a batch to exploit cross-image structure (e.g. finding
  a pattern repeated across images); each output must depend only on its own input.
* Changing the task (classification, segmentation, attack detection with rejection).
