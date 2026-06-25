"""Small conditional DDPM used as the inpainting diffusion model."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / max(1, half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TinyInpaintUNet(nn.Module):
    """A compact denoiser conditioned on masked image, mask, and timestep."""

    def __init__(self, base: int = 48, time_dim: int = 64) -> None:
        super().__init__()
        self.time = nn.Sequential(SinusoidalTime(time_dim), nn.Linear(time_dim, base * 4), nn.SiLU())
        in_ch = 3 + 3 + 1
        self.c1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.c2 = nn.Conv2d(base, base, 3, padding=1)
        self.d1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.d2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.d3 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid1 = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.mid2 = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.u1 = nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1)
        self.u2 = nn.Conv2d(base * 2 + base, base, 3, padding=1)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x_t: torch.Tensor, masked: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time(t)[:, :, None, None]
        h0 = F.silu(self.c1(torch.cat([x_t, masked, mask], dim=1)))
        h0 = F.silu(self.c2(h0))
        h1 = F.silu(self.d1(h0))
        h1 = F.silu(self.d2(h1))
        h2 = F.silu(self.d3(h1))
        h2 = F.silu(self.mid1(h2) + emb)
        h2 = F.silu(self.mid2(h2))
        h = F.interpolate(h2, size=h1.shape[-2:], mode="nearest")
        h = F.silu(self.u1(torch.cat([h, h1], dim=1)))
        h = F.interpolate(h, size=h0.shape[-2:], mode="nearest")
        h = F.silu(self.u2(torch.cat([h, h0], dim=1)))
        return self.out(h)


class DDPM:
    def __init__(self, timesteps: int = 80, device: str | torch.device = "cpu") -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)
        betas = torch.linspace(1e-4, 0.035, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    @torch.no_grad()
    def inpaint(
        self,
        model: nn.Module,
        image: torch.Tensor,
        mask: torch.Tensor,
        steps: int = 20,
    ) -> torch.Tensor:
        """RePaint-style DDPM inpainting.

        mask is 1 where pixels are regenerated and 0 where context is kept.
        """
        model.eval()
        if image.dim() == 3:
            image = image[None]
        if mask.dim() == 3:
            mask = mask[None]
        image = image.to(self.device)
        mask = mask.to(self.device)
        masked = image * (1.0 - mask)
        x = torch.randn_like(image)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        for idx, t_scalar in enumerate(schedule):
            t = t_scalar.repeat(image.shape[0])
            eps = model(x, masked, mask, t)
            beta = self.betas[t_scalar]
            alpha = self.alphas[t_scalar]
            alpha_bar = self.alpha_bars[t_scalar]
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
            if idx < len(schedule) - 1:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
                keep_t = schedule[idx + 1].repeat(image.shape[0])
                keep = self.q_sample(image, keep_t, torch.randn_like(image))
                x = mask * x + (1.0 - mask) * keep
            else:
                x = mean
                x = mask * x + (1.0 - mask) * image
        return x.clamp(-1.0, 1.0)
