"""DISPATCH: regenerate-and-rectify defense against adversarial patches.

Implements Sec. 4.3-4.4 of the paper on top of the pretrained LDM inpainting
model (Sec. 4.5):

1. Regeneration.  The input I is resized to 512x512 and inpainted twice with
   complementary N x N checkerboard masks m0 and m1 = 1 - m0 (Eq. 3); the
   regenerated image is I~ = I~0 * m0 + I~1 * m1 (Eq. 4).
2. Rectification (Algorithm 1).  Both images are in [0, 1]; D[i,j] is the
   per-pixel RGB L2 distance (Eq. 5), D is blurred with a smoothing filter,
   flattened and split by KMeans with k = 2; pixels in the cluster with the
   larger centroid form the adversarial mask A (Eq. 6).
3. Output  I^ = A * I~ + (1 - A) * I  (Eq. 7).

Paper defaults: N = 32 grids, s = 5 DDIM steps, 512x512 regeneration.

The CLI processes every PNG in --input-dir and writes a PNG with the same
name and size to --output-dir (the defense contract in defenses/README.md).
Ablations used as baselines in the paper are exposed through --mode:
``regen`` returns I~ directly (Appendix A.1) and ``gray`` replaces masked
pixels by gray instead of regenerated content (DISPATCH(G), Appendix A.2).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ldm_inpaint import DEFAULT_CKPT, LDMInpainter  # noqa: E402

REGEN_SIZE = 512


def checkerboard_masks(size: int, grids: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Complementary checkerboard masks m0 and m1 (1 = region to regenerate)."""
    idx = (torch.arange(size) * grids) // size
    board = ((idx[:, None] + idx[None, :]) % 2 == 0).float()[None, None]
    return board, 1.0 - board


def image_seed(arr: np.ndarray, seed: int) -> int:
    digest = hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + f":{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def kmeans2_1d(values: np.ndarray, iters: int = 100) -> np.ndarray:
    """Exact Lloyd iterations of KMeans (k = 2) on a 1-D vector.

    Returns a boolean array: True for members of the larger-centroid cluster.
    """
    v = values.astype(np.float64).ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros(values.shape, dtype=bool)
    c0, c1 = np.percentile(v, 10), np.percentile(v, 90)
    if c1 <= c0:
        c0, c1 = lo, hi
    for _ in range(iters):
        thr = 0.5 * (c0 + c1)
        high = v > thr
        if high.all() or (~high).all():
            break
        n0, n1 = v[~high].mean(), v[high].mean()
        if abs(n0 - c0) < 1e-10 and abs(n1 - c1) < 1e-10:
            break
        c0, c1 = n0, n1
    return (values > 0.5 * (c0 + c1))


def adversarial_mask(original: np.ndarray, regenerated: np.ndarray, blur: int) -> np.ndarray:
    """Algorithm 1.  Inputs are HxWx3 float arrays in [0, 1]."""
    dist = np.linalg.norm(original - regenerated, axis=2).astype(np.float32)
    if blur > 1:
        dist = cv2.GaussianBlur(dist, (blur, blur), 0)
    return kmeans2_1d(dist).astype(np.float32)


class Dispatch:
    def __init__(self, ckpt: Path, device: str, grids: int, steps: int, seed: int, blur: int) -> None:
        self.ldm = LDMInpainter(ckpt, device=device)
        self.grids = grids
        self.steps = steps
        self.seed = seed
        self.blur = blur
        self.m0, self.m1 = checkerboard_masks(REGEN_SIZE, grids)

    def regenerate(self, images: list[np.ndarray], sample: int = 0) -> list[np.ndarray]:
        """Full-image regeneration (Eq. 3-4).  images: HxWx3 uint8 arrays."""
        batch, noises = [], []
        for arr in images:
            small = cv2.resize(arr, (REGEN_SIZE, REGEN_SIZE), interpolation=cv2.INTER_CUBIC)
            batch.append(torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1))
            gen = torch.Generator().manual_seed(image_seed(arr, self.seed * 1000 + sample))
            lat = REGEN_SIZE // 4
            noises.append(torch.randn((2, 3, lat, lat), generator=gen))
        x = torch.stack(batch)
        b = x.shape[0]
        # One UNet batch holds the m0 pass and the m1 pass for every image.
        xx = torch.cat([x, x], dim=0)
        masks = torch.cat([self.m0.expand(b, -1, -1, -1), self.m1.expand(b, -1, -1, -1)], dim=0)
        noise = torch.cat([torch.stack([n[0] for n in noises]), torch.stack([n[1] for n in noises])], dim=0)
        inpainted = self.ldm.inpaint(xx, masks, self.steps, noise).cpu()
        i0, i1 = inpainted[:b], inpainted[b:]
        regen = i0 * self.m0 + i1 * self.m1
        out = []
        for arr, r in zip(images, regen):
            h, w = arr.shape[:2]
            r_np = (r.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.float32)
            out.append(cv2.resize(r_np, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 255) / 255.0)
        return out

    def defend(self, images: list[np.ndarray], mode: str = "rectify", ensemble: int = 1, dilate: int = 0,
               debug: list | None = None) -> list[np.ndarray]:
        regens = [self.regenerate(images, sample=k) for k in range(ensemble)]
        regen = [np.mean([r[i] for r in regens], axis=0) for i in range(len(images))]
        out = []
        for arr, rg in zip(images, regen):
            mask = None
            orig = arr.astype(np.float32) / 255.0
            if mode == "regen":
                final = rg
            else:
                mask = adversarial_mask(orig, rg, self.blur)
                if dilate > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
                    mask = cv2.dilate(mask, kernel)
                mask = mask[..., None]
                fill = rg if mode == "rectify" else np.full_like(orig, 0.5)
                final = mask * fill + (1.0 - mask) * orig
            out.append((final * 255.0).round().clip(0, 255).astype(np.uint8))
            if debug is not None:
                debug.append((rg, mask))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--grids", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["rectify", "regen", "gray"], default="rectify")
    parser.add_argument("--iters", type=int, default=1, help="apply the defense this many times in sequence")
    parser.add_argument("--ensemble", type=int, default=1, help="average this many regenerations")
    parser.add_argument("--dilate", type=int, default=0, help="dilate the adversarial mask by this many pixels")
    parser.add_argument("--batch", type=int, default=int(os.environ.get("DISPATCH_BATCH", "4")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug-dir", default=None, help="also save regenerated images and masks here")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_dir.glob("*.png"))
    model = Dispatch(Path(args.checkpoint), args.device, args.grids, args.steps, args.seed, args.blur)
    start = time.time()
    for i in range(0, len(paths), args.batch):
        chunk = paths[i : i + args.batch]
        arrays = [np.asarray(Image.open(p).convert("RGB")) for p in chunk]
        debug = [] if args.debug_dir else None
        for _ in range(args.iters):
            arrays = model.defend(arrays, mode=args.mode, ensemble=args.ensemble, dilate=args.dilate, debug=debug)
        for p, arr in zip(chunk, arrays):
            Image.fromarray(arr).save(out_dir / p.name)
        if debug is not None:
            dbg = Path(args.debug_dir)
            dbg.mkdir(parents=True, exist_ok=True)
            for p, (rg, mask) in zip(chunk, debug[-len(chunk):]):
                Image.fromarray((rg * 255).round().clip(0, 255).astype(np.uint8)).save(dbg / f"{p.stem}_regen.png")
                if mask is not None:
                    Image.fromarray((mask[..., 0] * 255).astype(np.uint8)).save(dbg / f"{p.stem}_mask.png")
    elapsed = time.time() - start
    print(f"dispatch: {len(paths)} images in {elapsed:.1f}s ({elapsed / max(1, len(paths)):.3f}s/img) on {args.device}", flush=True)


if __name__ == "__main__":
    main()
