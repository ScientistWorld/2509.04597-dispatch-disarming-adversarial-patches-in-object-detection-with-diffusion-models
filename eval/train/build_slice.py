"""Build the visible (train) evaluation slice: data/inria_slice/.

* 200 INRIA-Person *train* images are held out for evaluation; the other 414
  are used to optimise the visible white-box patches against the protected
  detector (two hiding patches, two targeted creating patches).
* Every evaluation image is used for exactly one variant, so no attacked
  image has a clean counterpart among the inputs:
    clean       - unmodified image (clean-performance constraint)
    hide        - hiding patch on every person, scale 0.20 x box diagonal
    hide_large  - hiding patch on every person, scale 0.30 x box diagonal
    create      - one targeted creating patch in a background region
* Each image gets a small random border crop and a random horizontal flip.

This slice, its patches and its labels are public.  The hidden test is built
the same way from INRIA-Person *test* images with patches that are never
released.
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
from common import ROOT, detect, inria_images, load_detector, prepare  # noqa: E402

SALT = "dispatch-inria-visible-slice-v2"
SPLIT = "Train"
N_EVAL = 200
HIDE_SCALE = {"hide": 0.20, "hide_large": 0.30}
VARIANTS = ["clean", "hide", "hide_large", "create"]
PATCHES = {
    "hide_adv": ("hide", "adv", 11),
    "hide_nat": ("hide", "natural", 12),
    "create_stopsign": ("create", 13, 13),
    "create_umbrella": ("create", 28, 14),
}


def h(text: str) -> str:
    return hashlib.sha256(f"{SALT}:{text}".encode()).hexdigest()


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "inria_slice"))
    ap.add_argument("--hide-iters", type=int, default=3000)
    ap.add_argument("--create-iters", type=int, default=600)
    ap.add_argument("--per-variant", type=int, default=0, help="limit items per variant (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_detector(device)
    for p in model.parameters():
        p.requires_grad_(False)

    names = sorted(inria_images(SPLIT), key=h)
    eval_names, attack_names = sorted(names[:N_EVAL]), sorted(names[N_EVAL:])

    # 1. Visible patches, optimised on images that are not evaluated.
    attack_data = []
    for name in attack_names:
        img, persons, _ = prepare(SPLIT, name, [0, 0, 0, 0], False)
        attack_data.append((to_tensor(img), persons))
    for pname, (role, arg, seed) in PATCHES.items():
        path = out / "patches" / f"{pname}.png"
        if path.exists():
            print(f"patch {pname}: exists, skipping", flush=True)
            continue
        print(f"optimising {pname}", flush=True)
        if role == "hide":
            arr = optimise_hiding_patch(model, attack_data, arg, seed, args.hide_iters, device)
        else:
            arr = optimise_creating_patch(model, attack_data, arg, seed, args.create_iters, device)
        Image.fromarray(arr).save(path)

    # 2. Evaluation items.
    counts = {}
    prepared = {}
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
            pname = ["hide_adv", "hide_nat"][hide_kind[variant] % 2]
            hide_kind[variant] += 1
            for box in persons:
                side = HIDE_SCALE[variant] * math.hypot(box[2] - box[0], box[3] - box[1])
                cx, cy = hiding_center(box)
                item["patches"].append(
                    {"patch": pname, "box": [round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)]}
                )
        elif variant == "create":
            pname, target = [("create_stopsign", 13), ("create_umbrella", 28)][create_k % 2]
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
    spec = {"split": SPLIT, "salt": SALT, "items": items}
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {len(items)} items to {out} ({per_variant})", flush=True)


if __name__ == "__main__":
    main()
