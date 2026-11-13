# Problem Definition

## Research Question

How can an inference-time defense keep an object detector working when its input
images carry adversarial patches that either hide real objects (hiding attacks)
or make the detector report objects that are not there (creating attacks)?

The defense is an image-to-image transformation applied before a fixed,
pretrained detector.  It does not know whether an image is attacked, which
attack was used, or the patch's shape, location or pattern, and it must not
retrain or query the protected detector.  It should restore detections under
attack while leaving benign images intact.

## Why It Matters

Adversarial patches are printable and can be digitally inserted, so they are a
practical threat to surveillance, robotics and other systems built on object
detection.  A defense that removes patches by damaging benign content trades one
failure for another; a useful defense must do both jobs at once.

## Setting in this benchmark

* Images: INRIA-Person street and scene photographs (the paper's hiding
  benchmark), with person annotations.
* Protected detector: a COCO-pretrained Faster R-CNN (ResNet-50 FPN), fixed.
* Hiding attacks: universal white-box patches optimised against that detector,
  pasted on every annotated person at a size proportional to the person (0.2
  and 0.3 of the box diagonal).
* Creating attacks: targeted white-box patches that make the detector report a
  chosen object class, placed in empty background and degraded the way a printed,
  photographed patch would be (the paper's creating benchmark uses physical
  patches).
* The patches used for the hidden test are never released; the public slice
  uses patches built the same way with different random seeds.

## Success Criteria

A successful defense recovers person detections under hiding patches (including
large ones), suppresses patch-created false detections, and keeps clean-image
and non-attacked-object detection close to the undefended detector.
