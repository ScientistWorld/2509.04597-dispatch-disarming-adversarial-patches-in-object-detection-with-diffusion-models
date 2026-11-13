"""Detector, data composition and metrics for the hidden test.

Everything on the scoring side of the comparison lives here and is fixed:
the protected detector (COCO-pretrained Faster R-CNN R50-FPN, torchvision
weights), how attacked inputs are composed from INRIA-Person images and
optimised patches, and the metrics (COCO-style person AP at IoU 0.5 and AR,
APRICOT-style creating-attack ASR / mAP with the patch as ground truth).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INRIA_DIR = ROOT / "data" / "downloads" / "INRIAPerson"
DETECTOR_WEIGHTS = ROOT / "data" / "downloads" / "torchvision" / "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

LONG_SIDE = 640  # every benchmark image is resized so its longer side is 640 px
PERSON = 1  # torchvision COCO label id
SCORE_ASR = 0.3  # APRICOT / paper creating-attack confidence threshold
PATCH_IOU_ASR = 0.1  # paper: IoU of 0.1 with the adversarial patch
CLEAN_TOLERANCE = 0.05  # constraint: clean / object mAP.5 may drop at most this much vs. no defense

COCO_LABELS = {13: "stop sign", 28: "umbrella"}

EXPERIMENTS = {
    "inria_digital_hiding": {
        "attack_mAP5": True,
        "attack_AR": True,
        "clean_mAP5": True,
        "clean_AR": True,
    },
    "inria_hiding_large_patch": {
        "attack_mAP5": True,
        "attack_AR": True,
    },
    "creating_digital": {
        "targeted_mAP5": False,
        "targeted_ASR": False,
        "untargeted_mAP5": False,
        "untargeted_ASR": False,
        "object_mAP5": True,
    },
}


# ── INRIA ────────────────────────────────────────────────────────────────────

_BOX_RE = re.compile(r"\(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)")


def inria_annotation(split: str, image_name: str) -> list[list[float]]:
    path = INRIA_DIR / split / "annotations" / (Path(image_name).stem + ".txt")
    text = path.read_bytes().decode("latin-1")
    return [[float(v) for v in m.groups()] for m in _BOX_RE.finditer(text)]


def inria_images(split: str) -> list[str]:
    return sorted(p.name for p in (INRIA_DIR / split / "pos").glob("*.png"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(split: str, image_name: str, crop: list[float], flip: bool) -> tuple[Image.Image, list, list]:
    """Crop (fractions of width/height), optionally flip, resize longer side to LONG_SIDE.

    Returns the image, person boxes and ignore boxes (persons cut by the crop).
    """
    img = Image.open(INRIA_DIR / split / "pos" / image_name).convert("RGB")
    w, h = img.size
    x0, y0 = int(round(crop[0] * w)), int(round(crop[1] * h))
    x1, y1 = int(round(w - crop[2] * w)), int(round(h - crop[3] * h))
    img = img.crop((x0, y0, x1, y1))
    cw, ch = img.size
    scale = LONG_SIDE / max(cw, ch)
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    persons, ignore = [], []
    for bx0, by0, bx1, by1 in inria_annotation(split, image_name):
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        vis = (cx1 - cx0) * (cy1 - cy0) / area
        box = [(cx0 - x0) * scale, (cy0 - y0) * scale, (cx1 - x0) * scale, (cy1 - y0) * scale]
        if flip:
            box = [nw - box[2], box[1], nw - box[0], box[3]]
        box = [round(v, 2) for v in box]
        if vis >= 0.5 and box[3] - box[1] >= 20:
            persons.append(box)
        else:
            ignore.append(box)
    return img, persons, ignore


def compose(item: dict, patch_dir: Path) -> Image.Image:
    """Build the input image of one benchmark item."""
    img, _, _ = prepare(item["split"], item["source"], item["crop"], item["flip"])
    for p in item.get("patches", []):
        patch = Image.open(patch_dir / f"{p['patch']}.png").convert("RGB")
        x0, y0, x1, y1 = [int(v) for v in p["box"]]
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        img.paste(patch.resize((x1 - x0, y1 - y0), Image.BILINEAR), (x0, y0))
    return img


# ── Detector ────────────────────────────────────────────────────────────────


def load_detector(device: torch.device):
    """torchvision fasterrcnn_resnet50_fpn with the COCO_V1 weights, built offline."""
    from torchvision.models import resnet50
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection._utils import overwrite_eps
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops import misc as misc_nn_ops

    backbone = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, 3)
    model = FasterRCNN(
        backbone,
        num_classes=91,
        min_size=LONG_SIDE,
        max_size=LONG_SIDE,
        box_score_thresh=0.05,
        box_detections_per_img=100,
    )
    model.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location="cpu"))
    overwrite_eps(model, 0.0)
    return model.eval().to(device)


@torch.no_grad()
def detect(model, images: list[np.ndarray], device: torch.device, batch: int = 8) -> list[dict]:
    out = []
    for i in range(0, len(images), batch):
        tensors = [torch.from_numpy(np.array(a)).permute(2, 0, 1).float().div(255.0).to(device) for a in images[i : i + batch]]
        for r in model(tensors):
            out.append(
                {
                    "boxes": r["boxes"].cpu().numpy().round(2).tolist(),
                    "labels": r["labels"].cpu().numpy().tolist(),
                    "scores": r["scores"].cpu().numpy().round(5).tolist(),
                }
            )
    return out


# ── Metrics ─────────────────────────────────────────────────────────────────


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _match(dets: list[tuple[float, list]], gts: list, ignore: list, thr: float):
    """Greedy COCO-style matching in one image. Returns (scores, is_tp) of counted dets and #matched gts."""
    order = sorted(range(len(dets)), key=lambda k: -dets[k][0])[:100]
    used = [False] * len(gts)
    scores, tps = [], []
    for k in order:
        s, box = dets[k]
        best, best_j = thr, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            o = iou(box, g)
            if o >= best:
                best, best_j = o, j
        if best_j >= 0:
            used[best_j] = True
            scores.append(s)
            tps.append(1)
        elif any(iou(box, g) >= thr for g in ignore):
            continue
        else:
            scores.append(s)
            tps.append(0)
    return scores, tps, sum(used)


def average_precision(per_image: list[tuple[list, list, list]], thr: float = 0.5) -> float:
    """COCO 101-point interpolated AP.  per_image: (dets[(score, box)], gts, ignore)."""
    all_scores, all_tps, n_gt = [], [], 0
    for dets, gts, ign in per_image:
        s, t, _ = _match(dets, gts, ign, thr)
        all_scores += s
        all_tps += t
        n_gt += len(gts)
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(all_scores, dtype=np.float64), kind="mergesort")
    tps = np.asarray(all_tps, dtype=np.float64)[order]
    if len(tps) == 0:
        return 0.0
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1.0 - tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    total = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        idx = np.searchsorted(recall, r, side="left")
        total += precision[idx] if idx < len(precision) else 0.0
    return float(total / 101.0)


def average_recall(per_image: list[tuple[list, list, list]]) -> float:
    """COCO AR@100 averaged over IoU 0.50:0.05:0.95."""
    n_gt = sum(len(g) for _, g, _ in per_image)
    if n_gt == 0:
        return 0.0
    recalls = []
    for thr in np.arange(0.5, 0.951, 0.05):
        matched = sum(_match(d, g, i, float(thr))[2] for d, g, i in per_image)
        recalls.append(matched / n_gt)
    return float(np.mean(recalls))


def person_dets(pred: dict) -> list[tuple[float, list]]:
    return [(s, b) for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]) if l == PERSON]


def hiding_metrics(items: list[dict], preds: list[dict]) -> tuple[float, float]:
    per_image = [(person_dets(p), it["persons"], it["ignore"]) for it, p in zip(items, preds)]
    return average_precision(per_image, 0.5), average_recall(per_image)


def creating_metrics(items: list[dict], preds: list[dict]) -> dict:
    """APRICOT protocol: a wrong box (not a correct person detection) with confidence
    over 0.3 and IoU >= 0.1 with the patch is a successful attack; mAP.5 treats the
    patch as the only ground truth among wrong boxes that touch it."""
    succ_t, succ_u = 0, 0
    near_t, near_u = [], []
    for it, p in zip(items, preds):
        patch = it["create"]["box"]
        target = it["create"]["target"]
        wrong_t, wrong_u = [], []
        hit_t = hit_u = False
        for box, label, score in zip(p["boxes"], p["labels"], p["scores"]):
            if label == PERSON and any(iou(box, g) >= 0.5 for g in it["persons"] + it["ignore"]):
                continue
            o = iou(box, patch)
            if o < PATCH_IOU_ASR:
                continue
            wrong_u.append((score, box))
            hit_u |= score > SCORE_ASR
            if label == target:
                wrong_t.append((score, box))
                hit_t |= score > SCORE_ASR
        succ_t += hit_t
        succ_u += hit_u
        near_t.append((wrong_t, [patch], []))
        near_u.append((wrong_u, [patch], []))
    n = max(1, len(items))
    obj = [(person_dets(p), it["persons"], it["ignore"]) for it, p in zip(items, preds)]
    return {
        "targeted_mAP5": average_precision(near_t, 0.5),
        "targeted_ASR": succ_t / n,
        "untargeted_mAP5": average_precision(near_u, 0.5),
        "untargeted_ASR": succ_u / n,
        "object_mAP5": average_precision(obj, 0.5),
    }


def score_predictions(items: list[dict], preds: list[dict]) -> dict:
    by_variant: dict[str, tuple[list, list]] = {}
    for it, p in zip(items, preds):
        by_variant.setdefault(it["variant"], ([], []))
        by_variant[it["variant"]][0].append(it)
        by_variant[it["variant"]][1].append(p)
    clean_ap, clean_ar = hiding_metrics(*by_variant["clean"])
    hide_ap, hide_ar = hiding_metrics(*by_variant["hide"])
    large_ap, large_ar = hiding_metrics(*by_variant["hide_large"])
    return {
        "inria_digital_hiding": {
            "attack_mAP5": hide_ap,
            "attack_AR": hide_ar,
            "clean_mAP5": clean_ap,
            "clean_AR": clean_ar,
        },
        "inria_hiding_large_patch": {"attack_mAP5": large_ap, "attack_AR": large_ar},
        "creating_digital": creating_metrics(*by_variant["create"]),
    }


def worst_scores() -> dict:
    return {exp: {m: (0.0 if hib else 1.0) for m, hib in metrics.items()} for exp, metrics in EXPERIMENTS.items()}


def constraint_failures(method: dict, vanilla: dict) -> list[str]:
    fails = []
    for exp, metric in (("inria_digital_hiding", "clean_mAP5"), ("creating_digital", "object_mAP5")):
        if method[exp][metric] < vanilla[exp][metric] - CLEAN_TOLERANCE:
            fails.append(
                f"{exp}.{metric}={method[exp][metric]:.4f} is more than {CLEAN_TOLERANCE} below "
                f"the undefended detector ({vanilla[exp][metric]:.4f})"
            )
    return fails


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)
