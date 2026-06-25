"""Run the unprotected detector on clean and attacked images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from synthetic_detector import detect_persons  # noqa: E402


def detect_folder(folder: Path) -> dict[str, list[dict]]:
    return {path.stem: detect_persons(path) for path in sorted(folder.glob("*.png"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()
    out = Path(args.outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for variant, name in [
        ("clean", "clean_predictions.json"),
        ("hiding", "hiding_vanilla_predictions.json"),
        ("creating", "creating_vanilla_predictions.json"),
    ]:
        with (out / name).open("w") as f:
            json.dump(detect_folder(Path(args.data_dir) / "test" / variant), f, indent=2)


if __name__ == "__main__":
    main()
