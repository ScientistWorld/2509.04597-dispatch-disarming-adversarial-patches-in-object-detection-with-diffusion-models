"""Torch dataset helpers for clean synthetic images."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CleanImageDataset(Dataset):
    def __init__(self, root: str | Path) -> None:
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No png files found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        return tensor


def random_block_mask(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch, 1, height, width), device=device)
    for i in range(batch):
        blocks = int(torch.randint(1, 4, (1,), device=device).item())
        for _ in range(blocks):
            size = int(torch.randint(8, 19, (1,), device=device).item())
            x = int(torch.randint(0, max(1, width - size), (1,), device=device).item())
            y = int(torch.randint(0, max(1, height - size), (1,), device=device).item())
            mask[i, :, y : y + size, x : x + size] = 1.0
    return mask
