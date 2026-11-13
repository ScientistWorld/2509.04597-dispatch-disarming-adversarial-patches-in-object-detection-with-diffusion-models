"""Build the hidden test assets: eval/test/assets/.

Same construction as the visible slice (see the public slice builder), but
* the 480 evaluation images (120 per variant) are drawn, with a salt only this
  directory holds, from the pooled INRIA-Person positives that are *not* in the
  public 200-image slice,
* the five white-box patches (three hiding, two creating) are optimised with
  seeds that are never released, on the pooled images that are not evaluated
  here,
* crops, flips, variant assignment, placement, capture conditions and image
  ids come from the hidden salt,
* the SHA-256 of every fixed input (source images, detector weights, patches)
  is pinned so the evaluator refuses tampered inputs.
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
from common import (  # noqa: E402
    DETECTOR_WEIGHTS,
    INRIA_DIR,
    ROOT,
    detect,
    inria_images,
    iou,
    load_detector,
    pick_device,
    prepare,
    sha256_file,
)

SALT = "dispatch-inria-hidden-test-v3-ee5cfb13d4015a56"
PUBLIC_SLICE_SALT = "dispatch-inria-visible-slice-v3"  # the public slice takes the first 200 by this salt
N_PUBLIC = 200
N_HIDDEN = 480
HIDE_SCALE = {"hide": 0.20, "hide_large": 0.30}
VARIANTS = ["clean", "hide", "hide_large", "create"]
HIDE_PATCHES = {"q_h1": ("adv", 9201), "q_h2": ("adv", 9205), "q_h3": ("natural", 9202)}
CREATE_PATCHES = {"q_c1": (13, 9203), "q_c2": (28, 9204)}


def h(text: str) -> str:
    return hashlib.sha256(f"{SALT}:{text}".encode()).hexdigest()


def h_public(text: str) -> str:
    return hashlib.sha256(f"{PUBLIC_SLICE_SALT}:{text}".encode()).hexdigest()


def pool() -> list[tuple[str, str]]:
    return [(split, name) for split in ("Train", "Test") for name in inria_images(split)]


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def add_ignore_regions(persons: list, ignore: list, dets: dict) -> list:
    out = list(ignore)
    for box, label, score in zip(dets["boxes"], dets["labels"], dets["scores"]):
        if label != 1 or score < 0.5:
            continue
        if any(iou(box, g) >= 0.3 for g in persons) or any(iou(box, g) >= 0.5 for g in out):
            continue
        out.append([round(v, 2) for v in box])
    return out


def capture_params(rng: random.Random) -> dict:
    return {
        "contrast": round(rng.uniform(0.70, 0.95), 3),
        "brightness": round(rng.uniform(-0.05, 0.08), 3),
        "gain": [round(rng.uniform(0.90, 1.10), 3) for _ in range(3)],
        "blur": round(rng.uniform(0.6, 1.3), 3),
        "noise": 0.02,
        "noise_seed": rng.randrange(1 << 30),
    }


def build(out: Path, eval_set: list, attack_set: list, hide_patches: dict, create_patches: dict, salt_fn,
          hide_iters: int, create_iters: int, per_variant: int, batch: int, device) -> tuple[list, dict]:
    (out / "patches").mkdir(parents=True, exist_ok=True)
    model = load_detector(device)
    for p in model.parameters():
        p.requires_grad_(False)

    attack_data = []
    for split, name in attack_set:
        img, persons, _ = prepare(split, name, [0, 0, 0, 0], False)
        attack_data.append((to_tensor(img), persons))
    for pname, (kind, seed) in hide_patches.items():
        path = out / "patches" / f"{pname}.png"
        if not path.exists():
            print(f"optimising {pname}", flush=True)
            Image.fromarray(optimise_hiding_patch(model, attack_data, kind, seed, hide_iters, device, batch=batch)).save(path)
    for pname, (target, seed) in create_patches.items():
        path = out / "patches" / f"{pname}.png"
        if not path.exists():
            print(f"optimising {pname}", flush=True)
            Image.fromarray(optimise_creating_patch(model, attack_data, target, seed, create_iters, device, batch=batch)).save(path)
    del attack_data

    prepared = {}
    for split, name in eval_set:
        key = f"{split}/{name}"
        rng = random.Random(salt_fn("crop:" + key))
        crop = [round(rng.uniform(0.0, 0.04), 4) for _ in range(4)]
        flip = rng.random() < 0.5
        img, persons, ignore = prepare(split, name, crop, flip)
        dets = detect(model, [np.asarray(img)], device)[0]
        ignore = add_ignore_regions(persons, ignore, dets)
        prepared[key] = (split, name, crop, flip, img, persons, ignore, dets)
    ordered = sorted(prepared, key=lambda k: (len(prepared[k][5]), salt_fn("order:" + k)))
    assign = {k: VARIANTS[i % len(VARIANTS)] for i, k in enumerate(ordered)}

    items = []
    counts = {v: 0 for v in VARIANTS}
    hide_names = list(hide_patches)
    create_names = list(create_patches)
    for key in ordered:
        variant = assign[key]
        if per_variant and counts[variant] >= per_variant:
            continue
        split, name, crop, flip, img, persons, ignore, dets = prepared[key]
        item = {
            "id": salt_fn("id:" + key)[:12],
            "split": split,
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
            pname = hide_names[counts[variant] % len(hide_names)]
            for box in persons:
                side = HIDE_SCALE[variant] * math.hypot(box[2] - box[0], box[3] - box[1])
                cx, cy = hiding_center(box)
                item["patches"].append(
                    {"patch": pname, "box": [round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)]}
                )
        elif variant == "create":
            pname = create_names[counts[variant] % len(create_names)]
            target = create_patches[pname][0]
            w, hh = img.size
            avoid = [b for b, s in zip(dets["boxes"], dets["scores"]) if s >= 0.3]
            rng = random.Random(salt_fn("place:" + key))
            base = rng.uniform(0.14, 0.20) * min(w, hh)
            place = None
            for frac in (1.0, 0.8, 0.65, 0.5):
                place = background_box(persons + ignore, w, hh, base * frac, rng, avoid=avoid)
                if place:
                    break
            if place is None:
                place = background_box(persons + ignore, w, hh, base * 0.5, rng) or [0, 0, base * 0.5, base * 0.5]
            box = [round(v) for v in place]
            item["patches"].append({"patch": pname, "box": box, "capture": capture_params(rng)})
            item["create"] = {"box": box, "target": target}
        counts[variant] += 1
        items.append(item)
    items.sort(key=lambda it: it["id"])
    return items, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "test" / "assets"))
    ap.add_argument("--hide-iters", type=int, default=3000)
    ap.add_argument("--create-iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8, help="images per patch-optimisation step")
    ap.add_argument("--per-variant", type=int, default=0, help="limit items per variant (smoke tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device {device}", flush=True)
    images = pool()
    public = set(sorted(images, key=lambda sn: h_public(f"{sn[0]}/{sn[1]}"))[:N_PUBLIC])
    rest = sorted((sn for sn in images if sn not in public), key=lambda sn: h(f"{sn[0]}/{sn[1]}"))
    eval_set = sorted(rest[:N_HIDDEN])
    attack_set = sorted(sn for sn in images if sn not in set(eval_set))
    out = Path(args.out_dir)
    items, counts = build(out, eval_set, attack_set, HIDE_PATCHES, CREATE_PATCHES, h, args.hide_iters,
                          args.create_iters, args.per_variant, args.batch, device)
    pins = {
        "detector": sha256_file(DETECTOR_WEIGHTS),
        "sources": {f"{it['split']}/{it['source']}": sha256_file(INRIA_DIR / it["split"] / "pos" / it["source"]) for it in items},
        "patches": {p.stem: sha256_file(p) for p in sorted((out / "patches").glob("*.png"))},
    }
    (out / "spec.json").write_text(json.dumps({"items": items}, indent=1))
    (out / "pins.json").write_text(json.dumps(pins, indent=1))
    print(f"wrote {len(items)} hidden items to {out} ({counts})", flush=True)


if __name__ == "__main__":
    main()
