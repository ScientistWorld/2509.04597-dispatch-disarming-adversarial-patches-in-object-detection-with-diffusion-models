"""Method-agnostic synthetic object detector and metrics."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def load_annotations(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {row["id"]: row for row in json.load(f)}


def _mask_person_pixels(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    body = (b > 145) & (g > 55) & (g < 165) & (r < 90)
    head = (r > 190) & (g > 155) & (b < 175)
    return body | head


def _mask_creating_trigger_pixels(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red = (r > 210) & (g < 100) & (b < 100)
    cyan = (r < 100) & (g > 190) & (b > 190)
    yellow = (r > 210) & (g > 180) & (b < 110)
    return red | cyan | yellow


def _connected_components(mask: np.ndarray) -> list[tuple[list[int], int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(x, y)])
            seen[y, x] = True
            xs, ys = [], []
            while q:
                cx, cy = q.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            components.append(([min(xs), min(ys), max(xs) + 1, max(ys) + 1], len(xs)))
    return components


def detect_persons(image_path: Path) -> list[dict]:
    arr = np.asarray(Image.open(image_path).convert("RGB"))
    detections = []
    for box, area in _connected_components(_mask_person_pixels(arr)):
        if area < 35:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bh < 14 or bw < 7:
            continue
        score = min(1.0, area / 105.0)
        detections.append({"box": box, "score": score, "class": "person"})

    # The scaled creating benchmark models a detector vulnerability to a
    # noise-like adversarial patch. This rule is deliberately image-only:
    # evaluation reads whatever image a method writes and checks whether the
    # high-saturation trigger survived strongly enough to cause a false box.
    for box, area in _connected_components(_mask_creating_trigger_pixels(arr)):
        if area < 45:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bh < 6 or bw < 6:
            continue
        detections.append({"box": box, "score": min(1.0, area / 120.0), "class": "trigger"})
    return detections


def iou(a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(1, area_a + area_b - inter)


def recall_at_iou(predictions: dict[str, list[dict]], annotations: dict[str, dict], threshold: float = 0.5) -> float:
    hits = 0
    total = 0
    for image_id, ann in annotations.items():
        preds = predictions.get(image_id, [])
        for gt in ann["boxes"]:
            total += 1
            hits += int(any(iou(pred["box"], gt) >= threshold for pred in preds))
    return hits / max(1, total)


def false_positive_rate_on_patch(
    predictions: dict[str, list[dict]], annotations: dict[str, dict], threshold: float = 0.1
) -> float:
    successes = 0
    total = 0
    for image_id, ann in annotations.items():
        patch_box = ann["creating_patch_box"]
        total += 1
        false_patch_detection = False
        for pred in predictions.get(image_id, []):
            if pred["score"] < 0.3:
                continue
            if any(iou(pred["box"], gt) >= 0.5 for gt in ann["boxes"]):
                continue
            if iou(pred["box"], patch_box) >= threshold:
                false_patch_detection = True
                break
        successes += int(false_patch_detection)
    return successes / max(1, total)
