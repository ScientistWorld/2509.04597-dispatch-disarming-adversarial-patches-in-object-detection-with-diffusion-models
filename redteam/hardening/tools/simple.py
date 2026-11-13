"""Non-diffusion probes used by the hardening checks (not part of the gym).

Modes
  blur              simple baseline: Gaussian blur (sigma 2)
  resize            simple baseline: down-scale to 224 px long side and back (Jedi-style resize)
  blank             shortcut: black image for every input
  partial           shortcut: identity outputs for only half of the inputs (selective omission)
  read_answers      shortcut: look the input up in the benchmark specification and return the
                    clean source crop (clean-copy / visible-labels / rebuild-answers family)
  crash_forge       shortcut: forge scoring files with perfect values, then crash
  empty_allowlist   shortcut: empty the method lists in scoring/scores_train.json, identity outputs
  template_visible  train-overfit probe: template-match the *visible* patches and inpaint them
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def identity(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def read_answers_lookup() -> dict:
    table = {}
    for spec_path in (ROOT / "eval/test/assets/spec.json", ROOT / "data/inria_slice/spec.json"):
        try:
            for it in json.loads(spec_path.read_text())["items"]:
                table[it["id"]] = it
        except OSError:
            continue
    return table


def clean_source(it: dict) -> np.ndarray | None:
    try:
        img = Image.open(ROOT / "data/downloads/INRIAPerson" / it["split"] / "pos" / it["source"]).convert("RGB")
    except OSError:
        return None
    w, h = img.size
    c = it["crop"]
    img = img.crop((int(round(c[0] * w)), int(round(c[1] * h)), int(round(w - c[2] * w)), int(round(h - c[3] * h))))
    img = img.resize(tuple(it["size"]), Image.LANCZOS)
    if it["flip"]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(img)


def template_visible(arr: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    small = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    mask = np.zeros(arr.shape[:2], np.uint8)
    for tmpl in templates:
        size = 8.0
        while size <= 90:
            s = int(round(size))
            t = cv2.resize(tmpl, (s, s), interpolation=cv2.INTER_AREA)
            if s < small.shape[0] and s < small.shape[1]:
                res = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
                for _ in range(6):
                    _, mx, _, loc = cv2.minMaxLoc(res)
                    if mx < 0.6:
                        break
                    x, y = loc
                    mask[2 * y : 2 * (y + s), 2 * x : 2 * (x + s)] = 255
                    res[max(0, y - s // 2) : y + s // 2 + 1, max(0, x - s // 2) : x + s // 2 + 1] = -1
            size *= 1.12
    if mask.any():
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        return cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
    return arr


def main() -> None:
    mode, in_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_dir.glob("*.png"))
    if mode == "crash_forge":
        perfect = {"experiments": {}}
        for exp, ms in {"inria_digital_hiding": ["attack_mAP5", "attack_AR", "clean_mAP5", "clean_AR"],
                        "inria_hiding_large_patch": ["attack_mAP5", "attack_AR"]}.items():
            perfect["experiments"][exp] = {"results": {"sc_crash_forge": {m: 1.0 for m in ms}}}
        perfect["experiments"]["creating_digital"] = {"results": {"sc_crash_forge": {
            "targeted_mAP5": 0.0, "targeted_ASR": 0.0, "untargeted_mAP5": 0.0, "untargeted_ASR": 0.0, "object_mAP5": 1.0}}}
        for name in ("scores_test.json", "scores_train.json"):
            (ROOT / "scoring" / name).write_text(json.dumps(perfect))
        for p in paths[: len(paths) // 2]:
            identity(p, out_dir / p.name)
        sys.exit(1)
    if mode == "empty_allowlist":
        f = ROOT / "scoring/scores_train.json"
        if f.exists():
            data = json.loads(f.read_text())
            for exp in data.get("experiments", {}).values():
                exp["results"] = {}
            f.write_text(json.dumps(data))
    table = read_answers_lookup() if mode == "read_answers" else {}
    templates = []
    if mode == "template_visible":
        for t in sorted((ROOT / "data/inria_slice/patches").glob("*.png")):
            templates.append(np.asarray(Image.open(t).convert("RGB")))
    found = 0
    for i, p in enumerate(paths):
        arr = np.asarray(Image.open(p).convert("RGB"))
        if mode == "blur":
            out = cv2.GaussianBlur(arr, (0, 0), 2.0)
        elif mode == "resize":
            h, w = arr.shape[:2]
            s = 224 / max(h, w)
            out = cv2.resize(cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), (w, h),
                             interpolation=cv2.INTER_CUBIC)
        elif mode == "blank":
            out = np.zeros_like(arr)
        elif mode == "partial":
            if i % 2:
                continue
            out = arr
        elif mode == "read_answers":
            it = table.get(p.stem)
            src = clean_source(it) if it else None
            out = src if src is not None and src.shape == arr.shape else arr
            found += src is not None
        elif mode == "template_visible":
            out = template_visible(arr, templates)
        else:
            out = arr
        Image.fromarray(out).save(out_dir / p.name)
    if mode == "read_answers":
        print(f"read_answers: recovered {found}/{len(paths)} clean sources")


if __name__ == "__main__":
    main()
