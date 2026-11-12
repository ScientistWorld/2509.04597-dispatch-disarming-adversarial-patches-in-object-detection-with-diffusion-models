"""Seeded original training loop, with intermediate checkpoints; no test data."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
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
    artifact_dir = Path('/home/user/redteam/artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resume_path = artifact_dir / f'training_seed{a.seed}.pt'
    history_path = Path(f'/home/user/redteam/training_history_seed{a.seed}.json')
    first_epoch = 1
    history = []
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        first_epoch = saved['epoch'] + 1
        history = saved['history']
        print(f'resume seed={a.seed} at epoch={first_epoch}', flush=True)
    model.train()
    for epoch in range(first_epoch, a.epochs + 1):
        started = time.monotonic()
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
        history.append({'epoch': epoch, 'loss': total / len(loader),
                        'seconds': time.monotonic() - started})
        # Atomic epoch checkpoints survive SLURM drain/restart and preserve the
        # original training trajectory, including data order and diffusion noise.
        resume_tmp = resume_path.with_suffix('.tmp.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'epoch': epoch, 'history': history,
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state()}, resume_tmp)
        os.replace(resume_tmp, resume_path)
        history_path.write_text(json.dumps(history, indent=2) + '\n')
        if epoch in {1, 60, 120, a.epochs}:
            path = artifact_dir / f'seed{a.seed}_epoch{epoch}.pt'
            temporary = path.with_suffix('.tmp.pt')
            torch.save({'model': model.state_dict(), 'timesteps': 80, 'base_channels': 32,
                        'prediction_type': 'x0', 'steps': epoch * len(loader)}, temporary)
            os.replace(temporary, path)

if __name__ == '__main__':
    main()
