"""Train the clean-data inpainting diffusion backbone used by DISPATCH."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CleanImageDataset, random_dispatch_mask
from diffusion import DDPM, TinyInpaintUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic/train/clean")
    parser.add_argument("--checkpoint", default="checkpoints/dispatch_ddpm.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--grids", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = CleanImageDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = TinyInpaintUNet(base=args.base_channels).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for x0 in loader:
            x0 = x0.to(device)
            mask = random_dispatch_mask(x0.shape[0], x0.shape[2], x0.shape[3], device, grids=args.grids)
            masked = x0 * (1.0 - mask)
            t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = ddpm.q_sample(x0, t, noise)
            pred = model(x_t, masked, mask, t)
            loss = ((pred - noise) ** 2 * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            step += 1
        print(f"epoch={epoch + 1} loss={running / max(1, len(loader)):.6f}", flush=True)

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "timesteps": args.timesteps,
            "base_channels": args.base_channels,
            "steps": step,
        },
        ckpt,
    )


if __name__ == "__main__":
    main()
