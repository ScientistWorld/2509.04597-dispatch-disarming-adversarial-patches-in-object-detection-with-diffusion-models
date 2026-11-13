"""White-box adversarial patch optimisation against the protected detector.

Hiding patches follow AdvPatch (Thys et al. 2019, the paper's ref. [12]): one
universal patch is pasted on every person, at a size proportional to the
person's box diagonal (paper: scale 0.2), with expectation over
transformations (contrast, brightness, noise, rotation, scale, position), and
optimised to minimise the detector's maximum person confidence near every
patched person plus a total-variation term.  ``kind="natural"`` restricts the
patch to a smooth low-resolution texture (a stand-in for the paper's
naturalistic NatPatch family, ref. [13]).

Creating patches are targeted: pasted in background regions, they are
optimised to make the detector output a box of a chosen COCO class on the
patch (the APRICOT threat model, ref. [30]).
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from common import PERSON

PATCH_RES = 200


def tv_loss(p: torch.Tensor) -> torch.Tensor:
    return (p[:, :, 1:] - p[:, :, :-1]).abs().mean() + (p[:, 1:, :] - p[:, :-1, :]).abs().mean()


def paste(img: torch.Tensor, patch: torch.Tensor, cx: float, cy: float, side: float, angle: float) -> torch.Tensor:
    """Differentiably paste a square patch (side px, rotated) centred at (cx, cy)."""
    _, h, w = img.shape
    xc, yc = 2.0 * cx / w - 1.0, 2.0 * cy / h - 1.0
    ca, sa = math.cos(angle), math.sin(angle)
    a = torch.tensor([[ca, sa], [-sa, ca]], dtype=img.dtype, device=img.device) @ torch.diag(
        torch.tensor([w / side, h / side], dtype=img.dtype, device=img.device)
    )
    t = -a @ torch.tensor([xc, yc], dtype=img.dtype, device=img.device)
    theta = torch.cat([a, t[:, None]], dim=1)[None]
    grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
    warped = F.grid_sample(patch[None], grid, align_corners=False, padding_mode="zeros")[0]
    mask = F.grid_sample(torch.ones_like(patch[:1])[None], grid, align_corners=False, padding_mode="zeros")[0]
    return img * (1.0 - mask) + warped * mask


def eot(patch: torch.Tensor, rng: random.Random) -> torch.Tensor:
    contrast = rng.uniform(0.8, 1.2)
    brightness = rng.uniform(-0.1, 0.1)
    noise = (torch.rand_like(patch) * 2.0 - 1.0) * 0.06
    return (patch * contrast + brightness + noise).clamp(0.0, 1.0)


def class_probs(model, images: list[torch.Tensor], extra: list[torch.Tensor] | None = None):
    """Softmax class probabilities of every second-stage proposal (+ optional extra boxes)."""
    imgs, _ = model.transform(images)
    feats = model.backbone(imgs.tensors)
    proposals, _ = model.rpn(imgs, feats)
    scaled_extra = None
    if extra is not None:
        scaled_extra = []
        for e, (oh, ow), (nh, nw) in zip(extra, [im.shape[-2:] for im in images], imgs.image_sizes):
            s = torch.tensor([nw / ow, nh / oh, nw / ow, nh / oh], device=e.device, dtype=e.dtype)
            scaled_extra.append(e * s)
        proposals = [torch.cat([p, e]) for p, e in zip(proposals, scaled_extra)]
    pooled = model.roi_heads.box_roi_pool(feats, proposals, imgs.image_sizes)
    logits, _ = model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
    probs = F.softmax(logits, dim=-1).split([len(p) for p in proposals])
    # Map proposals back to input coordinates.
    back = []
    for p, (oh, ow), (nh, nw) in zip(proposals, [im.shape[-2:] for im in images], imgs.image_sizes):
        s = torch.tensor([ow / nw, oh / nh, ow / nw, oh / nh], device=p.device, dtype=p.dtype)
        back.append(p * s)
    return probs, back


def hiding_center(box: list[float]) -> tuple[float, float]:
    """Patch centre on a person box: horizontal centre, slightly above the box centre (torso)."""
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1) - 0.08 * (y1 - y0)


def init_patch(kind: str, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    res = 20 if kind == "natural" else PATCH_RES
    return torch.rand((3, res, res), generator=g).to(device).requires_grad_(True)


def render(param: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "natural":
        return F.interpolate(param[None], size=(PATCH_RES, PATCH_RES), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    return param.clamp(0, 1)


def optimise_hiding_patch(model, data: list[tuple[torch.Tensor, list]], kind: str, seed: int, iters: int,
                          device, batch: int = 8, scale_range=(0.18, 0.32), log=lambda m: print(m, flush=True)) -> np.ndarray:
    """data: list of (image tensor [3,H,W] in [0,1] on CPU, person boxes)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    param = init_patch(kind, seed, device)
    opt = torch.optim.Adam([param], lr=0.03 if kind != "natural" else 0.05)
    pool = [d for d in data if d[1]]
    t0 = time.time()
    for it in range(iters):
        samples = rng.sample(pool, min(batch, len(pool)))
        patch = render(param, kind)
        imgs, gts = [], []
        for img, boxes in samples:
            x = img.to(device)
            for box in boxes:
                w, h = box[2] - box[0], box[3] - box[1]
                side = rng.uniform(*scale_range) * math.hypot(w, h) * rng.uniform(0.9, 1.1)
                cx, cy = hiding_center(box)
                cx += rng.uniform(-0.05, 0.05) * w
                cy += rng.uniform(-0.05, 0.05) * h
                x = paste(x, eot(patch, rng), cx, cy, max(side, 4.0), math.radians(rng.uniform(-20, 20)))
            imgs.append(x)
            gts.append(torch.tensor(boxes, dtype=torch.float32, device=device))
        probs, props = class_probs(model, imgs)
        det = []
        for pr, pp, gt in zip(probs, props, gts):
            ov = box_iou(pp, gt)  # [P, G]
            person = pr[:, PERSON][:, None].expand_as(ov)
            near = torch.where(ov >= 0.3, person, torch.zeros_like(person))
            top = near.topk(min(10, near.shape[0]), dim=0).values  # [k, G]
            det.append((0.5 * top[0] + 0.5 * top.mean(dim=0)).mean())
        det_loss = torch.stack(det).mean()
        loss = det_loss + torch.clamp(2.5 * tv_loss(patch), min=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            param.clamp_(0, 1)
        if it % 250 == 0 or it == iters - 1:
            log(f"  hiding[{kind},{seed}] it={it} det={det_loss.item():.4f} tv={tv_loss(patch).item():.4f} t={time.time() - t0:.0f}s")
    return (render(param, kind).detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def background_box(boxes: list, w: int, h: int, side: int, rng: random.Random, avoid: list | None = None,
                   tries: int = 300):
    """A square of the given side that does not touch any (expanded) person / avoid box."""
    blocked = [[b[0] - 8, b[1] - 8, b[2] + 8, b[3] + 8] for b in boxes] + list(avoid or [])
    for _ in range(tries):
        x0 = rng.uniform(0, max(0, w - side))
        y0 = rng.uniform(0, max(0, h - side))
        cand = [x0, y0, x0 + side, y0 + side]
        if all(cand[2] <= b[0] or cand[0] >= b[2] or cand[3] <= b[1] or cand[1] >= b[3] for b in blocked):
            return cand
    return None


def optimise_creating_patch(model, data: list[tuple[torch.Tensor, list]], target: int, seed: int, iters: int,
                            device, batch: int = 8, size_range=(0.13, 0.20), log=lambda m: print(m, flush=True)) -> np.ndarray:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    param = init_patch("adv", seed, device)
    opt = torch.optim.Adam([param], lr=0.03)
    t0 = time.time()
    for it in range(iters):
        samples = rng.sample(data, min(batch, len(data)))
        patch = param.clamp(0, 1)
        imgs, extra, places = [], [], []
        for img, boxes in samples:
            _, h, w = img.shape
            side = rng.uniform(*size_range) * min(w, h)
            place = background_box(boxes, w, h, side, rng) or [0.0, 0.0, side, side]
            cx, cy = 0.5 * (place[0] + place[2]), 0.5 * (place[1] + place[3])
            x = paste(img.to(device), eot(patch, rng), cx, cy, side, math.radians(rng.uniform(-15, 15)))
            imgs.append(x)
            pb = torch.tensor(place, dtype=torch.float32, device=device)
            jit = [pb]
            for _ in range(4):
                d = (torch.rand(4, device=device) - 0.5) * 0.2 * side
                jit.append(pb + d)
            extra.append(torch.stack(jit))
            places.append(pb)
        probs, props = class_probs(model, imgs, extra)
        terms = []
        for pr, pp, pb in zip(probs, props, places):
            ov = box_iou(pp, pb[None])[:, 0]
            near = pr[ov >= 0.3, target]
            terms.append(-torch.log(near.topk(min(5, len(near))).values + 1e-6).mean())
        det_loss = torch.stack(terms).mean()
        loss = det_loss + torch.clamp(2.5 * tv_loss(patch), min=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            param.clamp_(0, 1)
        if it % 100 == 0 or it == iters - 1:
            log(f"  creating[{target},{seed}] it={it} nll={det_loss.item():.4f} t={time.time() - t0:.0f}s")
    return (param.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
