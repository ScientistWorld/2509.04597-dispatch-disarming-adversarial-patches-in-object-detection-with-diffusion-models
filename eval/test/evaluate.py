"""Held-out test-slice evaluator for the synthetic DISPATCH benchmark.

This evaluator scores the same method entries produced by the train slice
whenever scoring/scores_train.json is present. It scores saved image artifacts
only and never imports or executes method code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


SLICE = "test"
SPLIT_SEED = "dispatch-synthetic-eval-split-v1"


def load_annotations(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {row["id"]: row for row in json.load(f)}


def split_annotations(annotations: dict[str, dict], slice_name: str) -> dict[str, dict]:
    groups: dict[tuple[int, bool], list[dict]] = {}
    for ann in annotations.values():
        key = (len(ann["boxes"]), bool(ann.get("creating_active", False)))
        groups.setdefault(key, []).append(ann)

    selected: dict[str, dict] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: stable_digest(row["id"]))
        for index, ann in enumerate(ordered):
            if (slice_name == "train" and index % 2 == 0) or (slice_name == "test" and index % 2 == 1):
                selected[ann["id"]] = ann
    return dict(sorted(selected.items()))


def stable_digest(image_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{image_id}".encode("utf-8")).hexdigest()


def mask_person_pixels(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    body = (b > 145) & (g > 55) & (g < 165) & (r < 90)
    head = (r > 190) & (g > 155) & (b < 175)
    return body | head


def mask_creating_trigger_pixels(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red = (r > 210) & (g < 100) & (b < 100)
    cyan = (r < 100) & (g > 190) & (b > 190)
    magenta = (r > 200) & (g < 100) & (b > 200)
    return red | cyan | magenta


def connected_components(mask: np.ndarray) -> list[tuple[list[int], int]]:
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
    for box, area in connected_components(mask_person_pixels(arr)):
        if area < 35:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bh < 14 or bw < 7:
            continue
        detections.append({"box": box, "score": min(1.0, area / 105.0), "class": "person"})

    for box, area in connected_components(mask_creating_trigger_pixels(arr)):
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


def detect_folder(folder: Path, image_ids: list[str]) -> dict[str, list[dict]]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing image folder for evaluation: {folder}")
    predictions = {}
    for image_id in image_ids:
        image_path = folder / f"{image_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for evaluation: {image_path}")
        predictions[image_id] = detect_persons(image_path)
    return predictions


def write_prediction_cache(
    cache_dir: Path, method_name: str, variant: str, image_folder: Path, image_ids: list[str]
) -> dict[str, list[dict]]:
    predictions = detect_folder(image_folder, image_ids)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{method_name}_{variant}_predictions.json"
    payload = {
        "slice": SLICE,
        "method": method_name,
        "variant": variant,
        "image_folder": str(image_folder),
        "image_ids": image_ids,
        "predictions": predictions,
    }
    with cache_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return predictions


def existing_dir(path: Path) -> Path | None:
    return path if path.exists() else None


def discover_method_sources(outputs_dir: Path, data_dir: Path) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {
        "vanilla": {
            "clean": data_dir / "test" / "clean",
            "hiding": data_dir / "test" / "hiding",
            "creating": data_dir / "test" / "creating",
        }
    }

    dispatch_hiding = existing_dir(outputs_dir / "dispatch_images" / "hiding")
    dispatch_creating = existing_dir(outputs_dir / "dispatch_images" / "creating")
    if dispatch_hiding or dispatch_creating:
        sources["dispatch"] = {"clean": data_dir / "test" / "clean"}
        if dispatch_hiding:
            sources["dispatch"]["hiding"] = dispatch_hiding
        if dispatch_creating:
            sources["dispatch"]["creating"] = dispatch_creating

    regenerated_hiding = existing_dir(outputs_dir / "regenerated_images" / "hiding")
    if regenerated_hiding:
        sources["regenerated_only"] = {"hiding": regenerated_hiding}

    method_root = outputs_dir / "methods"
    if method_root.exists():
        for method_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
            method_sources = {}
            for variant in ("clean", "hiding", "creating"):
                variant_dir = existing_dir(method_dir / variant)
                if variant_dir:
                    method_sources[variant] = variant_dir
            if method_sources:
                method_sources.setdefault("clean", data_dir / "test" / "clean")
                sources[method_dir.name] = method_sources

    for image_dir in sorted(outputs_dir.glob("*_images")):
        if image_dir.name in {"dispatch_images", "regenerated_images"}:
            continue
        method_name = image_dir.name.removesuffix("_images")
        method_sources = {}
        for variant in ("clean", "hiding", "creating"):
            variant_dir = existing_dir(image_dir / variant)
            if variant_dir:
                method_sources[variant] = variant_dir
        if method_sources:
            method_sources.setdefault("clean", data_dir / "test" / "clean")
            sources[method_name] = method_sources

    return sources


def load_requested_methods(train_score_path: Path) -> dict[str, set[str]] | None:
    if not train_score_path.exists():
        return None
    with train_score_path.open() as f:
        train_scores = json.load(f)
    requested: dict[str, set[str]] = {}
    for experiment, payload in train_scores.get("experiments", {}).items():
        requested[experiment] = set(payload.get("results", {}))
    return requested


def score_methods(
    sources: dict[str, dict[str, Path]],
    annotations: dict[str, dict],
    cache_dir: Path,
    requested: dict[str, set[str]] | None,
) -> dict[str, dict]:
    image_ids = list(annotations)
    scores = {
        "experiments": {
            "synthetic_hiding_scaled": {"results": {}},
            "synthetic_creating_scaled": {"results": {}},
            "rectification_ablation": {"results": {}},
        }
    }

    if requested is None:
        requested = {
            "synthetic_hiding_scaled": {
                name for name, method_sources in sources.items() if name != "regenerated_only" and "hiding" in method_sources
            },
            "synthetic_creating_scaled": {
                name for name, method_sources in sources.items() if name != "regenerated_only" and "creating" in method_sources
            },
            "rectification_ablation": {
                name for name, method_sources in sources.items() if name in {"dispatch", "regenerated_only"} and "hiding" in method_sources
            },
        }

    for method_name in sorted(requested.get("synthetic_hiding_scaled", set())):
        if method_name not in sources or "hiding" not in sources[method_name]:
            raise FileNotFoundError(f"Train slice scored {method_name}, but no hiding images were found for test scoring")
        if "clean" in sources[method_name]:
            clean_predictions = write_prediction_cache(
                cache_dir, method_name, "clean", sources[method_name]["clean"], image_ids
            )
            clean_recall = recall_at_iou(clean_predictions, annotations)
        else:
            clean_recall = 0.0
        hiding_predictions = write_prediction_cache(
            cache_dir, method_name, "hiding", sources[method_name]["hiding"], image_ids
        )
        hiding_recall = recall_at_iou(hiding_predictions, annotations)
        scores["experiments"]["synthetic_hiding_scaled"]["results"][method_name] = {
            "clean_mAP5": clean_recall,
            "attack_mAP5": hiding_recall,
            "attack_AR": hiding_recall,
        }

    for method_name in sorted(requested.get("synthetic_creating_scaled", set())):
        if method_name not in sources or "creating" not in sources[method_name]:
            raise FileNotFoundError(f"Train slice scored {method_name}, but no creating images were found for test scoring")
        creating_predictions = write_prediction_cache(
            cache_dir, method_name, "creating", sources[method_name]["creating"], image_ids
        )
        creating_asr = false_positive_rate_on_patch(creating_predictions, annotations)
        scores["experiments"]["synthetic_creating_scaled"]["results"][method_name] = {
            "targeted_mAP5": creating_asr,
            "targeted_ASR": creating_asr,
            "untargeted_mAP5": creating_asr,
            "untargeted_ASR": creating_asr,
        }

    for method_name in sorted(requested.get("rectification_ablation", set())):
        if method_name not in sources or "hiding" not in sources[method_name]:
            raise FileNotFoundError(
                f"Train slice scored {method_name}, but no rectification hiding images were found for test scoring"
            )
        hiding_predictions = write_prediction_cache(
            cache_dir, method_name, "rectification_hiding", sources[method_name]["hiding"], image_ids
        )
        scores["experiments"]["rectification_ablation"]["results"][method_name] = {
            "attack_mAP5": recall_at_iou(hiding_predictions, annotations)
        }

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--score-path", default="scoring/scores_test.json")
    parser.add_argument("--train-score-path", default="scoring/scores_train.json")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    data_dir = Path(args.data_dir)
    annotations = split_annotations(load_annotations(data_dir / "test" / "annotations.json"), SLICE)
    scores = score_methods(
        discover_method_sources(outputs_dir, data_dir),
        annotations,
        outputs_dir / f"eval_{SLICE}",
        load_requested_methods(Path(args.train_score_path)),
    )

    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
