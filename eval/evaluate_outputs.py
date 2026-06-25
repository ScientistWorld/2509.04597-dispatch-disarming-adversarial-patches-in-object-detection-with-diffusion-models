"""Evaluate saved detector predictions for the synthetic DISPATCH benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_detector import false_positive_rate_on_patch, load_annotations, recall_at_iou


def load_predictions(path: Path) -> dict[str, list[dict]]:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--score-path", default="scoring/scores.json")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    annotations = load_annotations(Path(args.data_dir) / "test" / "annotations.json")

    clean = load_predictions(outputs_dir / "clean_predictions.json")
    hiding_vanilla = load_predictions(outputs_dir / "hiding_vanilla_predictions.json")
    hiding_dispatch = load_predictions(outputs_dir / "hiding_dispatch_predictions.json")
    creating_vanilla = load_predictions(outputs_dir / "creating_vanilla_predictions.json")
    creating_dispatch = load_predictions(outputs_dir / "creating_dispatch_predictions.json")

    clean_recall = recall_at_iou(clean, annotations)
    hiding_vanilla_recall = recall_at_iou(hiding_vanilla, annotations)
    hiding_dispatch_recall = recall_at_iou(hiding_dispatch, annotations)
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
        }
    }
    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
