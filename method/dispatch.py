"""DISPATCH regeneration and rectification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from diffusion import DDPM, TinyInpaintUNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from synthetic_detector import detect_persons  # noqa: E402


def checkerboard_mask(size: int, grids: int, invert: bool = False) -> torch.Tensor:
    cell = size // grids
    mask = torch.zeros((1, size, size), dtype=torch.float32)
    for gy in range(grids):
        for gx in range(grids):
            value = (gx + gy) % 2
            if invert:
                value = 1 - value
            y0, x0 = gy * cell, gx * cell
            y1 = size if gy == grids - 1 else (gy + 1) * cell
            x1 = size if gx == grids - 1 else (gx + 1) * cell
            mask[:, y0:y1, x0:x1] = float(value)
    return mask


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0).squeeze(0)
    arr = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def adversarial_mask(original: Image.Image, regenerated: Image.Image, blur: int = 5) -> np.ndarray:
    orig = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    regen = np.asarray(regenerated.convert("RGB"), dtype=np.float32) / 255.0
    dist = np.linalg.norm(orig - regen, axis=2)
    dist = cv2.GaussianBlur(dist, (blur, blur), 0)
    flat = dist.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(flat)
    high = int(np.argmax(km.cluster_centers_.reshape(-1)))
    mask = (km.labels_.reshape(dist.shape) == high).astype(np.float32)
    return mask[..., None]


@torch.no_grad()
def dispatch_image(
    image: Image.Image,
    model: TinyInpaintUNet,
    ddpm: DDPM,
    grids: int,
    steps: int,
    device: torch.device,
    prediction_type: str = "eps",
) -> tuple[Image.Image, Image.Image, Image.Image]:
    size = image.size[0]
    x = image_to_tensor(image).to(device)
    m0 = checkerboard_mask(size, grids, invert=False).to(device)
    m1 = checkerboard_mask(size, grids, invert=True).to(device)
    r0 = ddpm.inpaint(model, x, m0, steps=steps, prediction_type=prediction_type)
    r1 = ddpm.inpaint(model, x, m1, steps=steps, prediction_type=prediction_type)
    regen_tensor = r0 * m0 + r1 * m1
    regenerated = tensor_to_image(regen_tensor)
    mask = adversarial_mask(image, regenerated)
    orig = np.asarray(image.convert("RGB"), dtype=np.float32)
    regen = np.asarray(regenerated.convert("RGB"), dtype=np.float32)
    rectified = (mask * regen + (1.0 - mask) * orig).clip(0, 255).astype(np.uint8)
    mask_img = Image.fromarray((mask.squeeze(-1) * 255).astype(np.uint8))
    return Image.fromarray(rectified), regenerated, mask_img


def run_folder(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device)
    model = TinyInpaintUNet(base=state.get("base_channels", 48)).to(device)
    model.load_state_dict(state["model"])
    ddpm = DDPM(timesteps=state["timesteps"], device=device)
    prediction_type = state.get("prediction_type", "eps")

    out_root = Path(args.outputs_dir)
    image_out = out_root / "dispatch_images" / args.variant
    regen_out = out_root / "regenerated_images" / args.variant
    mask_out = out_root / "masks" / args.variant
    for folder in [image_out, regen_out, mask_out]:
        folder.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for path in sorted((Path(args.data_dir) / "test" / args.variant).glob("*.png")):
        image = Image.open(path).convert("RGB")
        rectified, regenerated, mask = dispatch_image(
            image, model, ddpm, args.grids, args.steps, device, prediction_type=prediction_type
        )
        rectified_path = image_out / path.name
        rectified.save(rectified_path)
        regenerated.save(regen_out / path.name)
        mask.save(mask_out / path.name)
        predictions[path.stem] = detect_persons(rectified_path)

    pred_name = f"{args.variant}_dispatch_predictions.json"
    with (out_root / pred_name).open("w") as f:
        json.dump(predictions, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dispatch_ddpm.pt")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--variant", choices=["hiding", "creating"], required=True)
    parser.add_argument("--grids", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_folder(args)


if __name__ == "__main__":
    main()
