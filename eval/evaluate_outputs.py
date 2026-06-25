"""Evaluate saved detector predictions for the synthetic DISPATCH benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_detector import detect_persons, false_positive_rate_on_patch, iou, load_annotations, recall_at_iou


def detect_folder(folder: Path) -> dict[str, list[dict]]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing image folder for evaluation: {folder}")
    return {path.stem: detect_persons(path) for path in sorted(folder.glob("*.png"))}


def load_or_detect(path: Path, image_folder: Path) -> dict[str, list[dict]]:
    predictions = detect_folder(image_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(predictions, f, indent=2)
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

    dispatch_sources = {}
    for variant in ("clean", "hiding", "creating"):
        variant_dir = existing_dir(outputs_dir / "dispatch_images" / variant)
        if variant_dir:
            dispatch_sources[variant] = variant_dir
    if "hiding" in dispatch_sources or "creating" in dispatch_sources:
        sources["dispatch"] = dispatch_sources

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
            sources[method_name] = method_sources

    return sources


def person_detections(preds: list[dict]) -> list[dict]:
    return [pred for pred in preds if pred.get("class") == "person"]


def average_recall(predictions: dict[str, list[dict]], annotations: dict[str, dict]) -> float:
    thresholds = (0.5, 0.6, 0.7)
    return sum(recall_at_iou(predictions, annotations, threshold) for threshold in thresholds) / len(thresholds)


def precision_at_iou(predictions: dict[str, list[dict]], annotations: dict[str, dict], threshold: float = 0.5) -> float:
    true_positive = 0
    false_positive = 0
    for image_id, ann in annotations.items():
        matched_gt: set[int] = set()
        for pred in person_detections(predictions.get(image_id, [])):
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(ann["boxes"]):
                if idx in matched_gt:
                    continue
                overlap = iou(pred["box"], gt)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
            if best_iou >= threshold:
                matched_gt.add(best_idx)
                true_positive += 1
            else:
                false_positive += 1
    return true_positive / max(1, true_positive + false_positive)


def score_methods(
    sources: dict[str, dict[str, Path]], annotations: dict[str, dict], outputs_dir: Path
) -> dict[str, dict]:
    scores = {
        "experiments": {
            "synthetic_hiding_scaled": {"results": {}},
            "synthetic_creating_scaled": {"results": {}},
            "rectification_ablation": {"results": {}},
        }
    }

    for method_name, method_sources in sources.items():
        if method_name != "regenerated_only" and "hiding" in method_sources:
            if "clean" in method_sources:
                clean_predictions = load_or_detect(
                    outputs_dir / f"{method_name}_clean_predictions.json", method_sources["clean"]
                )
                clean_recall = recall_at_iou(clean_predictions, annotations)
                clean_precision = precision_at_iou(clean_predictions, annotations)
            else:
                clean_recall = 0.0
                clean_precision = 0.0

            hiding_predictions = load_or_detect(
                outputs_dir / f"{method_name}_hiding_predictions.json", method_sources["hiding"]
            )
            scores["experiments"]["synthetic_hiding_scaled"]["results"][method_name] = {
                "clean_mAP5": clean_recall,
                "clean_precision": clean_precision,
                "attack_mAP5": recall_at_iou(hiding_predictions, annotations),
                "attack_AR": average_recall(hiding_predictions, annotations),
                "attack_precision": precision_at_iou(hiding_predictions, annotations),
            }

        if method_name != "regenerated_only" and "creating" in method_sources:
            creating_predictions = load_or_detect(
                outputs_dir / f"{method_name}_creating_predictions.json", method_sources["creating"]
            )
            creating_asr = false_positive_rate_on_patch(creating_predictions, annotations)
            scores["experiments"]["synthetic_creating_scaled"]["results"][method_name] = {
                "targeted_mAP5": creating_asr,
                "targeted_ASR": creating_asr,
                "untargeted_mAP5": creating_asr,
                "untargeted_ASR": creating_asr,
                "object_mAP5": recall_at_iou(creating_predictions, annotations),
            }

        if method_name in {"dispatch", "regenerated_only"} and "hiding" in method_sources:
            hiding_predictions = load_or_detect(
                outputs_dir / f"{method_name}_rectification_hiding_predictions.json", method_sources["hiding"]
            )
            scores["experiments"]["rectification_ablation"]["results"][method_name] = {
                "attack_mAP5": recall_at_iou(hiding_predictions, annotations),
                "attack_AR": average_recall(hiding_predictions, annotations),
                "attack_precision": precision_at_iou(hiding_predictions, annotations),
            }

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--score-path", default="scoring/scores.json")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    annotations = load_annotations(Path(args.data_dir) / "test" / "annotations.json")
    scores = score_methods(discover_method_sources(outputs_dir, Path(args.data_dir)), annotations, outputs_dir)
    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
