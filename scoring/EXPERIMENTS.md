# Experiment Overview

Every experiment runs the defense once on a mixed set of clean and attacked
INRIA-Person images and scores the protected detector's outputs on the defended
images.  The patches are optimised white-box against that detector; the hidden
test uses patches and images that are never released.

## inria_digital_hiding

Establishes whether a defense restores person detection when every person
carries an adversarial patch of the standard size (0.2 of the person's box
diagonal), and whether it does so without degrading detection on clean images.
Benefit: detection quality on attacked images.  Constraint: detection quality on
clean images, which must stay within 0.05 mAP.5 of the undefended detector and is
also scored.

## inria_hiding_large_patch

Establishes how well the defense holds up when the hiding patches are larger
(0.3 of the box diagonal), where the undefended detector loses most people and
known defenses degrade sharply.  This is the benchmark's main source of headroom.

## creating_digital

Establishes whether the defense suppresses false detections caused by a
patch designed to make the detector report a specific object class, placed in
empty background and degraded like a printed, photographed patch.  Benefit:
lower attack success and lower patch-detection AP (targeted and any class).
Constraint: person detection on the same images, which must stay within 0.05
mAP.5 of the undefended detector and is also scored.
