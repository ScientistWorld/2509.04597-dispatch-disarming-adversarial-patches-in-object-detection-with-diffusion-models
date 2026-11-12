"""Seeded original training loop, with intermediate checkpoints; no test data."""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path('/home/user/method')))
from dataset import CleanImageDataset, random_dispatch_mask
from diffusion import DDPM, TinyInpaintUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, required=True)
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    dataset = CleanImageDataset('/home/user/data/synthetic/train/clean')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model = TinyInpaintUNet(base=32)
    ddpm = DDPM(timesteps=80, device='cpu')
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for epoch in range(1, a.epochs + 1):
        total = 0
        for x0 in loader:
            mask = random_dispatch_mask(x0.shape[0], x0.shape[2], x0.shape[3], torch.device('cpu'), grids=16)
            masked = x0 * (1 - mask)
            t = torch.randint(0, 80, (x0.shape[0],))
            noise = torch.randn_like(x0)
            pred = model(ddpm.q_sample(x0, t, noise), masked, mask, t)
            loss = ((pred - x0) ** 2 * mask).sum() / (mask.sum() * 3).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total += loss.item()
        print(f'seed={a.seed} epoch={epoch} loss={total / len(loader):.7f}', flush=True)
        if epoch in {1, 60, 120, a.epochs}:
            path = Path(f'/home/user/redteam/artifacts/seed{a.seed}_epoch{epoch}.pt')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'timesteps': 80, 'base_channels': 32,
                        'prediction_type': 'x0', 'steps': epoch * len(loader)}, path)

if __name__ == '__main__':
    main()
