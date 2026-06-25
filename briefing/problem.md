# Problem Definition

## Research Question

How can an inference-time defense preserve object detector performance when images contain adversarial patches that either hide real objects or create false detections?

The defense should operate on input images before detection and should not require advance knowledge of patch shape, location, or visual pattern. It should improve detector behavior under attack while avoiding unnecessary damage to clean images.

## Why It Matters

Adversarial patches can be physically printed or digitally inserted, making them practical threats for surveillance, robotics, and other vision systems that rely on object detection.

## Success Criteria

A successful solution should recover detections of real objects under hiding attacks, suppress false detections on adversarial patches under creating attacks, and keep clean-image detection performance close to the undefended detector.
