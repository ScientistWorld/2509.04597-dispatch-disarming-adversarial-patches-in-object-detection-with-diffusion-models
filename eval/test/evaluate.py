"""Hidden-test evaluator: runs every defense on disk on hidden inputs and scores it.

Same contract and metrics as the visible slice (defenses/README.md):
``bash defenses/<name>/defend.sh <input_dir> <output_dir>`` is called for
every directory under defenses/ that has a defend.sh.  The inputs are
composed here, at test time, from INRIA-Person test images and patches that
only this directory holds; they are written with opaque names to a fresh
temporary directory.  While a defense runs, this directory (answers, patches,
specification) and the fixed inputs it is built from are made unreadable.

Every number is recomputed from the defended images: the evaluator runs the
protected detector itself.  A defense that fails, writes missing or
malformed outputs, or drops clean / object mAP.5 more than 0.05 below the
undefended detector receives the worst value of every metric in every
experiment.  Fixed inputs are checked against pinned SHA-256 digests; if
they were altered, every defense is scored as failed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    DETECTOR_WEIGHTS,
    INRIA_DIR,
    ROOT,
    compose,
    constraint_failures,
    detect,
    load_detector,
    score_predictions,
    sha256_file,
    worst_scores,
)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
EXPERIMENT_ORDER = ("inria_digital_hiding", "inria_hiding_large_patch", "creating_digital")


class Locks:
    """Make answer-bearing paths unreadable while a defense runs."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [p for p in paths if p.exists() and not p.is_symlink()]
        self.saved: dict[Path, int] = {}

    def __enter__(self):
        for p in self.paths:
            try:
                self.saved[p] = stat.S_IMODE(p.stat().st_mode)
                p.chmod(0)
            except OSError:
                pass
        return self

    def __exit__(self, *exc):
        for p, mode in self.saved.items():
            try:
                p.chmod(mode)
            except OSError:
                pass
        self.saved = {}


def discover(defenses_dir: Path) -> list[Path]:
    if not defenses_dir.is_dir():
        return []
    return sorted(p for p in defenses_dir.iterdir() if p.is_dir() and (p / "defend.sh").is_file())


def run_defense(script: Path, in_dir: Path, out_dir: Path, timeout: float, log_path: Path, locks: Locks) -> str | None:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log_path.open("w") as log, locks:
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


def check_pins(assets: Path, items: list[dict]) -> list[str]:
    pins = json.loads((assets / "pins.json").read_text())
    bad = []
    if sha256_file(DETECTOR_WEIGHTS) != pins["detector"]:
        bad.append("detector weights")
    for src in sorted({it["source"] for it in items}):
        if sha256_file(INRIA_DIR / "Test" / "pos" / src) != pins["sources"].get(src):
            bad.append(f"source {src}")
    for name, digest in pins["patches"].items():
        if sha256_file(assets / "patches" / f"{name}.png") != digest:
            bad.append(f"patch {name}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", default=str(HERE / "assets"))
    ap.add_argument("--defenses-dir", default=str(ROOT / "defenses"))
    ap.add_argument("--score-path", default=str(ROOT / "scoring" / "scores_test.json"))
    ap.add_argument("--log-dir", default=str(ROOT / "outputs" / "eval_test"))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("GYM_DEFENSE_TIMEOUT", 4 * 3600)))
    args = ap.parse_args()

    assets = Path(args.assets_dir)
    spec = json.loads((assets / "spec.json").read_text())
    items = spec["items"]
    tampered = check_pins(assets, items)
    if tampered:
        print(f"FIXED INPUTS ALTERED: {tampered[:5]} -> every defense is scored as failed", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    locks = Locks([HERE, INRIA_DIR / "Test", DETECTOR_WEIGHTS.parent])

    def restore(signum, frame):  # restore permissions if the job is interrupted
        locks.__exit__(None, None, None)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, restore)

    tmp = Path(tempfile.mkdtemp(prefix="dispatch_eval_"))
    results, status = {}, {}
    try:
        inputs = {}
        in_dir = tmp / "inputs"
        in_dir.mkdir()
        for it in items:
            img = compose(it, assets / "patches")
            inputs[it["id"]] = np.asarray(img)
            img.save(in_dir / f"{it['id']}.png")
        detector = load_detector(device)

        def evaluate(arrays: list[np.ndarray]) -> dict:
            return score_predictions(items, detect(detector, arrays, device))

        vanilla = evaluate([inputs[it["id"]] for it in items])
        results["vanilla"], status["vanilla"] = vanilla, "ok (undefended detector, computed by the evaluator)"

        for ddir in discover(Path(args.defenses_dir)):
            name = ddir.name
            if name == "vanilla" or not NAME_RE.match(name):
                status[name] = "skipped: reserved or invalid name"
                continue
            print(f"defense {name}", flush=True)
            run_in, run_out = tmp / name / "in", tmp / name / "out"
            shutil.copytree(in_dir, run_in)
            run_out.mkdir(parents=True)
            err = run_defense(ddir / "defend.sh", run_in, run_out, args.timeout, log_dir / "logs" / f"{name}.log", locks)
            arrays = None
            if err is None and tampered:
                err = "fixed inputs altered: " + ", ".join(tampered[:3])
            if err is None:
                arrays, err = load_outputs(run_out, items, inputs)
            if err is not None:
                results[name], status[name] = worst_scores(), f"failed: {err}"
            else:
                scores = evaluate(arrays)
                fails = constraint_failures(scores, vanilla)
                if fails:
                    results[name], status[name] = worst_scores(), "failed constraint: " + "; ".join(fails)
                else:
                    results[name], status[name] = scores, "ok"
            shutil.rmtree(tmp / name, ignore_errors=True)
    finally:
        locks.__exit__(None, None, None)
        shutil.rmtree(tmp, ignore_errors=True)

    scores = {"experiments": {exp: {"results": {m: r[exp] for m, r in results.items()}} for exp in EXPERIMENT_ORDER}}
    score_path = Path(args.score_path)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scores, indent=2))
    (log_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
