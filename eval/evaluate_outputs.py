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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--score-path", default="scoring/scores.json")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    annotations = load_annotations(Path(args.data_dir) / "test" / "annotations.json")

    clean = load_or_detect(
        outputs_dir / "clean_predictions.json",
        Path(args.data_dir) / "test" / "clean",
    )
    hiding_vanilla = load_or_detect(
        outputs_dir / "hiding_vanilla_predictions.json",
        Path(args.data_dir) / "test" / "hiding",
    )
    hiding_dispatch = load_or_detect(
        outputs_dir / "hiding_dispatch_predictions.json",
        outputs_dir / "dispatch_images" / "hiding",
    )
    hiding_regenerated = load_or_detect(
        outputs_dir / "hiding_regenerated_predictions.json",
        outputs_dir / "regenerated_images" / "hiding",
    )
    creating_vanilla = load_or_detect(
        outputs_dir / "creating_vanilla_predictions.json",
        Path(args.data_dir) / "test" / "creating",
    )
    creating_dispatch = load_or_detect(
        outputs_dir / "creating_dispatch_predictions.json",
        outputs_dir / "dispatch_images" / "creating",
    )

    clean_recall = recall_at_iou(clean, annotations)
    clean_precision = precision_at_iou(clean, annotations)
    hiding_vanilla_recall = recall_at_iou(hiding_vanilla, annotations)
    hiding_vanilla_ar = average_recall(hiding_vanilla, annotations)
    hiding_vanilla_precision = precision_at_iou(hiding_vanilla, annotations)
    hiding_dispatch_recall = recall_at_iou(hiding_dispatch, annotations)
    hiding_dispatch_ar = average_recall(hiding_dispatch, annotations)
    hiding_dispatch_precision = precision_at_iou(hiding_dispatch, annotations)
    hiding_regenerated_recall = recall_at_iou(hiding_regenerated, annotations)
    hiding_regenerated_ar = average_recall(hiding_regenerated, annotations)
    hiding_regenerated_precision = precision_at_iou(hiding_regenerated, annotations)
    creating_vanilla_asr = false_positive_rate_on_patch(creating_vanilla, annotations)
    creating_dispatch_asr = false_positive_rate_on_patch(creating_dispatch, annotations)
    creating_vanilla_object_recall = recall_at_iou(creating_vanilla, annotations)
    creating_dispatch_object_recall = recall_at_iou(creating_dispatch, annotations)

    scores = {
        "experiments": {
            "synthetic_hiding_scaled": {
                "results": {
                    "vanilla": {
                        "clean_mAP5": clean_recall,
                        "clean_precision": clean_precision,
                        "attack_mAP5": hiding_vanilla_recall,
                        "attack_AR": hiding_vanilla_ar,
                        "attack_precision": hiding_vanilla_precision,
                    },
                    "dispatch": {
                        "clean_mAP5": clean_recall,
                        "clean_precision": clean_precision,
                        "attack_mAP5": hiding_dispatch_recall,
                        "attack_AR": hiding_dispatch_ar,
                        "attack_precision": hiding_dispatch_precision,
                    },
                }
            },
            "synthetic_creating_scaled": {
                "results": {
                    "vanilla": {
                        "targeted_mAP5": creating_vanilla_asr,
                        "targeted_ASR": creating_vanilla_asr,
                        "untargeted_mAP5": creating_vanilla_asr,
                        "untargeted_ASR": creating_vanilla_asr,
                        "object_mAP5": creating_vanilla_object_recall,
                    },
                    "dispatch": {
                        "targeted_mAP5": creating_dispatch_asr,
                        "targeted_ASR": creating_dispatch_asr,
                        "untargeted_mAP5": creating_dispatch_asr,
                        "untargeted_ASR": creating_dispatch_asr,
                        "object_mAP5": creating_dispatch_object_recall,
                    },
                }
            },
            "rectification_ablation": {
                "results": {
                    "regenerated_only": {
                        "attack_mAP5": hiding_regenerated_recall,
                        "attack_AR": hiding_regenerated_ar,
                        "attack_precision": hiding_regenerated_precision,
                    },
                    "dispatch": {
                        "attack_mAP5": hiding_dispatch_recall,
                        "attack_AR": hiding_dispatch_ar,
                        "attack_precision": hiding_dispatch_precision,
                    },
                }
            },
        }
    }
    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
