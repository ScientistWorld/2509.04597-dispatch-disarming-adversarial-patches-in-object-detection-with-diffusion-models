"""Synthetic object-detection data for the DISPATCH reproduction.

The original paper evaluates on INRIA-Person and APRICOT. This small dataset
keeps the same problem structure while avoiding proprietary/large assets: clean
images contain one or two simple "person" objects, hiding attacks occlude each
person with a high-frequency patch, and creating attacks add a decoy patch that
resembles object colors in empty space.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_SIZE = 64
PERSON_COLOR = (35, 115, 220)
HEAD_COLOR = (240, 200, 130)
PATCH_COLORS = [(35, 115, 220), (240, 200, 130), (20, 20, 20), (245, 245, 245)]


def _draw_person(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    head_r = max(3, w // 5)
    cx = (x0 + x1) // 2
    head_y = y0 + head_r + 1
    draw.ellipse((cx - head_r, head_y - head_r, cx + head_r, head_y + head_r), fill=HEAD_COLOR)
    torso_top = y0 + 2 * head_r + 2
    torso_bot = y0 + int(0.68 * h)
    draw.rectangle((x0 + w // 4, torso_top, x1 - w // 4, torso_bot), fill=PERSON_COLOR)
    draw.line((x0 + w // 3, torso_bot, x0 + w // 5, y1), fill=PERSON_COLOR, width=3)
    draw.line((x1 - w // 3, torso_bot, x1 - w // 5, y1), fill=PERSON_COLOR, width=3)
    arm_y = torso_top + (torso_bot - torso_top) // 3
    draw.line((x0 + w // 4, arm_y, x0, torso_bot - 2), fill=PERSON_COLOR, width=3)
    draw.line((x1 - w // 4, arm_y, x1, torso_bot - 2), fill=PERSON_COLOR, width=3)


def _patch_pattern(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    patch = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(patch)
    cell = 4
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=rng.choice(PATCH_COLORS))
    return patch


def _non_overlapping_person_boxes(rng: random.Random) -> list[tuple[int, int, int, int]]:
    boxes = []
    count = 1 if rng.random() < 0.55 else 2
    for _ in range(count):
        for _attempt in range(50):
            w = rng.randint(17, 22)
            h = rng.randint(35, 45)
            x0 = rng.randint(4, IMAGE_SIZE - w - 4)
            y0 = rng.randint(8, IMAGE_SIZE - h - 3)
            box = (x0, y0, x0 + w, y0 + h)
            if all(_iou(box, other) < 0.05 for other in boxes):
                boxes.append(box)
                break
    return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(1, area_a + area_b - inter)


def make_example(index: int, split: str) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
    rng = random.Random(index + (0 if split == "train" else 100_000))
    bg = tuple(rng.randint(178, 235) for _ in range(3))
    clean = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(clean)
    boxes = _non_overlapping_person_boxes(rng)
    for box in boxes:
        _draw_person(draw, box)

    hiding = clean.copy()
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        w = x1 - x0
        h = y1 - y0
        p = max(12, int(((w * w + h * h) ** 0.5) * 0.28))
        px = (x0 + x1 - p) // 2
        py = y0 + int(0.38 * h)
        hiding.paste(_patch_pattern(p, index * 13 + k), (px, py))

    creating = clean.copy()
    patch_size = 15
    for _attempt in range(100):
        px = rng.randint(2, IMAGE_SIZE - patch_size - 2)
        py = rng.randint(2, IMAGE_SIZE - patch_size - 2)
        patch_box = (px, py, px + patch_size, py + patch_size)
        if all(_iou(patch_box, box) < 0.01 for box in boxes):
            break
    creating.paste(_patch_pattern(patch_size, index * 17 + 3), (px, py))

    meta = {
        "id": f"{split}_{index:04d}",
        "boxes": [list(b) for b in boxes],
        "creating_patch_box": [px, py, px + patch_size, py + patch_size],
    }
    return clean, hiding, creating, meta


def generate_dataset(output_dir: Path, train_count: int, test_count: int) -> None:
    for split, count in [("train", train_count), ("test", test_count)]:
        for variant in ["clean", "hiding", "creating"]:
            (output_dir / split / variant).mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(count):
            clean, hiding, creating, meta = make_example(i, split)
            for variant, image in [("clean", clean), ("hiding", hiding), ("creating", creating)]:
                image.save(output_dir / split / variant / f"{meta['id']}.png")
            records.append(meta)
        with (output_dir / split / "annotations.json").open("w") as f:
            json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--test-count", type=int, default=120)
    args = parser.parse_args()
    generate_dataset(Path(args.output_dir), args.train_count, args.test_count)


if __name__ == "__main__":
    main()
