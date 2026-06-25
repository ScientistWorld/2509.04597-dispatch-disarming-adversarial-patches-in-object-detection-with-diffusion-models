"""Evaluate saved detector predictions for the synthetic DISPATCH benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_detector import detect_persons, false_positive_rate_on_patch, load_annotations, recall_at_iou


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
    hiding_vanilla_recall = recall_at_iou(hiding_vanilla, annotations)
    hiding_dispatch_recall = recall_at_iou(hiding_dispatch, annotations)
    hiding_regenerated_recall = recall_at_iou(hiding_regenerated, annotations)
    creating_vanilla_asr = false_positive_rate_on_patch(creating_vanilla, annotations)
    creating_dispatch_asr = false_positive_rate_on_patch(creating_dispatch, annotations)

    scores = {
        "experiments": {
            "synthetic_hiding_scaled": {
                "results": {
                    "vanilla": {
                        "clean_mAP5": clean_recall,
                        "attack_mAP5": hiding_vanilla_recall,
                        "attack_AR": hiding_vanilla_recall,
                    },
                    "dispatch": {
                        "clean_mAP5": clean_recall,
                        "attack_mAP5": hiding_dispatch_recall,
                        "attack_AR": hiding_dispatch_recall,
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
                    },
                    "dispatch": {
                        "targeted_mAP5": creating_dispatch_asr,
                        "targeted_ASR": creating_dispatch_asr,
                        "untargeted_mAP5": creating_dispatch_asr,
                        "untargeted_ASR": creating_dispatch_asr,
                    },
                }
            },
            "rectification_ablation": {
                "results": {
                    "regenerated_only": {
                        "attack_mAP5": hiding_regenerated_recall,
                    },
                    "dispatch": {
                        "attack_mAP5": hiding_dispatch_recall,
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
