"""Visible-slice evaluator: runs every defense on disk and scores it.

Contract (see defenses/README.md): a defense is a directory
``defenses/<name>/`` containing ``defend.sh``.  The evaluator calls
``bash defenses/<name>/defend.sh <input_dir> <output_dir>`` from the
workspace root.  <input_dir> holds the benchmark images as ``<id>.png``
(clean, hiding-attacked and creating-attacked images, mixed, with opaque
names).  The defense must write one RGB PNG with the same name and size per
input to <output_dir>.  The evaluator then runs the protected detector on
the outputs and computes every metric itself.

A defense that fails (non-zero exit, timeout, missing / unreadable /
wrongly sized outputs) or violates the clean-preservation constraint
receives the worst value of every metric in every experiment.  The
undefended detector ("vanilla") is always scored on the same inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT,
    compose,
    constraint_failures,
    detect,
    load_detector,
    pick_device,
    score_predictions,
    worst_scores,
)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def discover(defenses_dir: Path) -> list[Path]:
    if not defenses_dir.is_dir():
        return []
    return sorted(p for p in defenses_dir.iterdir() if p.is_dir() and (p / "defend.sh").is_file())


def run_defense(script: Path, in_dir: Path, out_dir: Path, timeout: float, log_path: Path) -> str | None:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log_path.open("w") as log:
            proc = subprocess.run(
                ["bash", str(script), str(in_dir), str(out_dir)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout:.0f}s"
    print(f"  {script.parent.name}: exit {proc.returncode} in {time.time() - start:.0f}s", flush=True)
    if proc.returncode != 0:
        return f"exit code {proc.returncode}"
    return None


def load_outputs(out_dir: Path, items: list[dict], inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray] | None, str | None]:
    arrays = []
    for it in items:
        path = out_dir / f"{it['id']}.png"
        if not path.is_file():
            return None, f"missing output {path.name}"
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            return None, f"unreadable output {path.name}: {exc}"
        if arr.shape != inputs[it["id"]].shape:
            return None, f"output {path.name} has shape {arr.shape}, expected {inputs[it['id']].shape}"
        arrays.append(arr)
    return arrays, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default=str(ROOT / "data" / "inria_slice"))
    ap.add_argument("--defenses-dir", default=str(ROOT / "defenses"))
    ap.add_argument("--score-path", default=str(ROOT / "scoring" / "scores_train.json"))
    ap.add_argument("--save-dir", default=str(ROOT / "outputs" / "eval_train"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("GYM_DEFENSE_TIMEOUT", 4 * 3600)))
    args = ap.parse_args()

    slice_dir = Path(args.slice_dir)
    spec = json.loads((slice_dir / "spec.json").read_text())
    items = spec["items"]
    device = pick_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(os.environ.get("GYM_TMPDIR", ROOT / "outputs"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".eval_train_", dir=tmp_root))
    try:
        inputs = {}
        in_dir = tmp / "inputs"
        in_dir.mkdir()
        (save_dir / "inputs").mkdir(exist_ok=True)
        for it in items:
            img = compose(it, slice_dir / "patches")
            inputs[it["id"]] = np.asarray(img)
            img.save(in_dir / f"{it['id']}.png")
            img.save(save_dir / "inputs" / f"{it['id']}.png")
        detector = load_detector(device)

        def evaluate(arrays: list[np.ndarray]) -> tuple[dict, list]:
            preds = detect(detector, arrays, device)
            return score_predictions(items, preds), preds

        results, status, all_preds = {}, {}, {}
        vanilla, all_preds["vanilla"] = evaluate([inputs[it["id"]] for it in items])
        results["vanilla"], status["vanilla"] = vanilla, "ok (undefended detector, computed by the evaluator)"

        for ddir in discover(Path(args.defenses_dir)):
            name = ddir.name
            if name == "vanilla" or not NAME_RE.match(name):
                status[name] = "skipped: reserved or invalid name"
                continue
            print(f"defense {name}", flush=True)
            results[name], status[name] = worst_scores(), "failed: evaluation did not complete"
            try:
                run_in, run_out = tmp / name / "in", tmp / name / "out"
                shutil.copytree(in_dir, run_in)
                run_out.mkdir(parents=True)
                err = run_defense(ddir / "defend.sh", run_in, run_out, args.timeout, save_dir / "logs" / f"{name}.log")
                arrays = None
                if err is None:
                    arrays, err = load_outputs(run_out, items, inputs)
                if err is not None:
                    status[name] = f"failed: {err}"
                    continue
                scores, all_preds[name] = evaluate(arrays)
                fails = constraint_failures(scores, vanilla)
                if fails:
                    status[name] = "failed constraint: " + "; ".join(fails)
                else:
                    results[name], status[name] = scores, "ok"
                keep = save_dir / name
                shutil.rmtree(keep, ignore_errors=True)
                shutil.copytree(run_out, keep)
            except Exception as exc:  # noqa: BLE001  - any evaluation error fails only this defense
                results[name], status[name] = worst_scores(), f"failed: {type(exc).__name__}: {str(exc)[:300]}"
            finally:
                shutil.rmtree(tmp / name, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    scores = {"experiments": {}}
    for exp in ("inria_digital_hiding", "inria_hiding_large_patch", "creating_digital"):
        scores["experiments"][exp] = {"results": {m: r[exp] for m, r in results.items()}}
    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    (save_dir / "status.json").write_text(json.dumps(status, indent=2))
    (save_dir / "predictions.json").write_text(json.dumps({"items": items, "predictions": all_preds}))
    print(json.dumps(status, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
