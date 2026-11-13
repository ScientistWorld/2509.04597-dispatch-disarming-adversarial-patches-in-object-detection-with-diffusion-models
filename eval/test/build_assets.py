"""Build the hidden test assets: eval/test/assets/.

Same construction as the visible slice, but
* evaluation images are the 288 INRIA-Person *test* images (72 per variant),
* the four white-box patches are optimised with seeds that are never
  released (on all 614 INRIA-Person training images),
* crops, flips, variant assignment, patch placement and image ids come from a
  salt that only this directory holds,
* the SHA-256 of every fixed input (source images, detector weights) is
  pinned so the evaluator refuses tampered inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack import background_box, hiding_center, optimise_creating_patch, optimise_hiding_patch  # noqa: E402
from common import DETECTOR_WEIGHTS, INRIA_DIR, ROOT, detect, inria_images, load_detector, pick_device, prepare, sha256_file  # noqa: E402

SALT = "dispatch-inria-hidden-test-v2-44556860a99a6aa7"
SPLIT = "Test"
ATTACK_SPLIT = "Train"
HIDE_SCALE = {"hide": 0.20, "hide_large": 0.30}
VARIANTS = ["clean", "hide", "hide_large", "create"]
PATCHES = {
    "p_h1": ("hide", "adv", 9101),
    "p_h2": ("hide", "natural", 9102),
    "p_c1": ("create", 13, 9103),
    "p_c2": ("create", 28, 9104),
}


def h(text: str) -> str:
    return hashlib.sha256(f"{SALT}:{text}".encode()).hexdigest()


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "test" / "assets"))
    ap.add_argument("--hide-iters", type=int, default=3000)
    ap.add_argument("--create-iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8, help="images per patch-optimisation step")
    ap.add_argument("--per-variant", type=int, default=0, help="limit items per variant (smoke tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device {device}", flush=True)
    model = load_detector(device)
    for p in model.parameters():
        p.requires_grad_(False)

    attack_data = []
    for name in inria_images(ATTACK_SPLIT):
        img, persons, _ = prepare(ATTACK_SPLIT, name, [0, 0, 0, 0], False)
        attack_data.append((to_tensor(img), persons))
    for pname, (role, arg, seed) in PATCHES.items():
        path = out / "patches" / f"{pname}.png"
        if path.exists():
            print(f"patch {pname}: exists, skipping", flush=True)
            continue
        print(f"optimising {pname}", flush=True)
        if role == "hide":
            arr = optimise_hiding_patch(model, attack_data, arg, seed, args.hide_iters, device, batch=args.batch)
        else:
            arr = optimise_creating_patch(model, attack_data, arg, seed, args.create_iters, device, batch=args.batch)
        Image.fromarray(arr).save(path)

    eval_names = inria_images(SPLIT)
    counts, prepared = {}, {}
    for name in eval_names:
        rng = random.Random(h("crop:" + name))
        crop = [round(rng.uniform(0.0, 0.04), 4) for _ in range(4)]
        flip = rng.random() < 0.5
        img, persons, ignore = prepare(SPLIT, name, crop, flip)
        prepared[name] = (crop, flip, img, persons, ignore)
        counts[name] = len(persons)
    ordered = sorted(eval_names, key=lambda n: (counts[n], h("order:" + n)))
    assign = {name: VARIANTS[i % len(VARIANTS)] for i, name in enumerate(ordered)}

    items = []
    per_variant = {v: 0 for v in VARIANTS}
    hide_kind = {"hide": 0, "hide_large": 0}
    create_k = 0
    for name in ordered:
        variant = assign[name]
        if args.per_variant and per_variant[variant] >= args.per_variant:
            continue
        per_variant[variant] += 1
        crop, flip, img, persons, ignore = prepared[name]
        item = {
            "id": h("id:" + name)[:12],
            "split": SPLIT,
            "source": name,
            "crop": crop,
            "flip": flip,
            "size": list(img.size),
            "variant": variant,
            "persons": persons,
            "ignore": ignore,
            "patches": [],
        }
        if variant in HIDE_SCALE:
            pname = ["p_h1", "p_h2"][hide_kind[variant] % 2]
            hide_kind[variant] += 1
            for box in persons:
                side = HIDE_SCALE[variant] * math.hypot(box[2] - box[0], box[3] - box[1])
                cx, cy = hiding_center(box)
                item["patches"].append(
                    {"patch": pname, "box": [round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)]}
                )
        elif variant == "create":
            pname, target = [("p_c1", 13), ("p_c2", 28)][create_k % 2]
            create_k += 1
            w, hh = img.size
            dets = detect(model, [np.asarray(img)], device)[0]
            avoid = [b for b, s in zip(dets["boxes"], dets["scores"]) if s >= 0.3]
            rng = random.Random(h("place:" + name))
            base = rng.uniform(0.14, 0.20) * min(w, hh)
            place = None
            for frac in (1.0, 0.8, 0.65, 0.5):
                place = background_box(persons + ignore, w, hh, base * frac, rng, avoid=avoid)
                if place:
                    break
            if place is None:
                place = background_box(persons + ignore, w, hh, base * 0.5, rng) or [0, 0, base * 0.5, base * 0.5]
            box = [round(v) for v in place]
            item["patches"].append({"patch": pname, "box": box})
            item["create"] = {"box": box, "target": target}
        items.append(item)

    items.sort(key=lambda it: it["id"])
    pins = {
        "detector": sha256_file(DETECTOR_WEIGHTS),
        "sources": {it["source"]: sha256_file(INRIA_DIR / SPLIT / "pos" / it["source"]) for it in items},
        "patches": {p.stem: sha256_file(p) for p in sorted((out / "patches").glob("*.png"))},
    }
    (out / "spec.json").write_text(json.dumps({"split": SPLIT, "items": items}, indent=1))
    (out / "pins.json").write_text(json.dumps(pins, indent=1))
    print(f"wrote {len(items)} hidden items to {out} ({per_variant})", flush=True)


if __name__ == "__main__":
    main()
