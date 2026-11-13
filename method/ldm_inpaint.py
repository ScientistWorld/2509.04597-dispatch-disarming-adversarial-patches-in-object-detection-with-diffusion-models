"""Pretrained LDM inpainting backbone used by DISPATCH (paper Sec. 4.5).

The paper regenerates images with "the original pretrained inpainting LDM
weights provided by the authors" of latent diffusion, sampled with DDIM at
512x512.  This module loads exactly that checkpoint (CompVis latent-diffusion
``inpainting_big``) and reproduces the computation of the upstream
``scripts/inpaint.py``:

* the masked image is encoded by the VQ-f4 first stage and concatenated with
  the down-sampled mask (mask and image both mapped to [-1, 1]);
* a DDIM sampler (eta = 0, uniform discretisation, upstream "+1" timestep
  offset) denoises a 3x128x128 latent with the concat-conditioned UNet;
* the latent is decoded through the VQ quantiser and decoder and the known
  pixels are pasted back.

Only the pure-PyTorch network definitions are taken from the vendored
upstream code in ``method/third_party/ldm`` (unmodified, MIT licence); the
PyTorch-Lightning wrappers are replaced by the small classes below so the
checkpoint can be used without the original training stack.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from ldm.modules.diffusionmodules.model import Decoder, Encoder  # noqa: E402
from ldm.modules.diffusionmodules.openaimodel import UNetModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "data" / "downloads" / "ldm_inpainting_big" / "last.ckpt"

# models/ldm/inpainting_big/config.yaml (upstream)
UNET_CONFIG = dict(
    image_size=64,
    in_channels=7,
    out_channels=3,
    model_channels=256,
    attention_resolutions=[8, 4, 2],
    num_res_blocks=2,
    channel_mult=[1, 2, 3, 4],
    num_heads=8,
    resblock_updown=True,
)
DDCONFIG = dict(
    attn_type="none",
    double_z=False,
    z_channels=3,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
)
EMBED_DIM = 3
N_EMBED = 8192
LINEAR_START = 0.0015
LINEAR_END = 0.0205
NUM_TIMESTEPS = 1000


class VectorQuantizer(nn.Module):
    """Nearest-codebook quantiser (taming VectorQuantizer2, legacy, inference only)."""

    def __init__(self, n_e: int, e_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        weight = self.embedding.weight
        d = (flat**2).sum(1, keepdim=True) + (weight**2).sum(1) - 2.0 * flat @ weight.t()
        idx = torch.argmin(d, dim=1)
        zq = self.embedding(idx).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return zq


class VQFirstStage(nn.Module):
    """VQModelInterface: encode without quantisation, decode with quantisation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(**DDCONFIG)
        self.decoder = Decoder(**DDCONFIG)
        self.quantize = VectorQuantizer(N_EMBED, EMBED_DIM)
        self.quant_conv = nn.Conv2d(DDCONFIG["z_channels"], EMBED_DIM, 1)
        self.post_quant_conv = nn.Conv2d(EMBED_DIM, DDCONFIG["z_channels"], 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(self.quantize(h)))


class _Dummy:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _TolerantUnpickler(pickle.Unpickler):
    """Load Lightning checkpoints without installing Lightning/OmegaConf."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Dummy


class _TolerantPickle:
    Unpickler = _TolerantUnpickler
    load = pickle.load


def _load_state_dict(path: Path) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu")
    except (ModuleNotFoundError, AttributeError):
        ckpt = torch.load(path, map_location="cpu", pickle_module=_TolerantPickle)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


class LDMInpainter:
    """Upstream ``inpainting_big`` + DDIM, used as the DISPATCH regenerator."""

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str | torch.device = "cuda", use_ema: bool = True) -> None:
        self.device = torch.device(device)
        sd = _load_state_dict(Path(ckpt))
        self.unet = UNetModel(**UNET_CONFIG)
        self.first_stage = VQFirstStage()

        unet_sd = {k[len("model.diffusion_model."):]: v for k, v in sd.items() if k.startswith("model.diffusion_model.")}
        self.used_ema = False
        if use_ema and any(k.startswith("model_ema.") for k in sd):
            # LitEma stores "model_ema." + ("diffusion_model." + name).replace(".", "")
            ema_sd = {}
            for name, _ in self.unet.named_parameters():
                key = "model_ema." + ("diffusion_model." + name).replace(".", "")
                if key not in sd:
                    ema_sd = {}
                    break
                ema_sd[name] = sd[key]
            if ema_sd:
                unet_sd.update(ema_sd)
                self.used_ema = True
        missing, unexpected = self.unet.load_state_dict(unet_sd, strict=False)
        if missing:
            raise RuntimeError(f"LDM UNet weights missing: {missing[:5]} ...")
        fs_sd = {k[len("first_stage_model."):]: v for k, v in sd.items() if k.startswith("first_stage_model.")}
        fs_sd = {k: v for k, v in fs_sd.items() if not k.startswith("loss.")}
        missing, unexpected = self.first_stage.load_state_dict(fs_sd, strict=False)
        if missing:
            raise RuntimeError(f"LDM first-stage weights missing: {missing[:5]} ...")
        del sd

        self.unet.eval().to(self.device)
        self.first_stage.eval().to(self.device)
        for p in list(self.unet.parameters()) + list(self.first_stage.parameters()):
            p.requires_grad_(False)

        betas = torch.linspace(LINEAR_START**0.5, LINEAR_END**0.5, NUM_TIMESTEPS, dtype=torch.float64) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def _ddim_schedule(self, steps: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        c = NUM_TIMESTEPS // steps
        timesteps = np.asarray(list(range(0, NUM_TIMESTEPS, c))) + 1  # upstream make_ddim_timesteps
        ac = self.alphas_cumprod
        alphas = ac[timesteps].float()
        alphas_prev = torch.tensor([float(ac[0])] + ac[timesteps[:-1]].tolist(), dtype=torch.float32)
        return timesteps, alphas, alphas_prev

    @torch.no_grad()
    def inpaint(self, image: torch.Tensor, mask: torch.Tensor, steps: int, noise: torch.Tensor) -> torch.Tensor:
        """Inpaint masked pixels.

        image: [B,3,H,W] in [0,1]; mask: [B,1,H,W] in {0,1} (1 = regenerate);
        noise: [B,3,H/4,W/4] standard normal initial latent.
        Returns the inpainted image in [0,1] (known pixels copied from input).
        """
        image = image.to(self.device, torch.float32)
        mask = mask.to(self.device, torch.float32)
        masked = (1.0 - mask) * image
        c = self.first_stage.encode(masked * 2.0 - 1.0)
        cc = F.interpolate(mask * 2.0 - 1.0, size=c.shape[-2:])
        cond = torch.cat([c, cc], dim=1)

        timesteps, alphas, alphas_prev = self._ddim_schedule(steps)
        x = noise.to(self.device, torch.float32)
        b = x.shape[0]
        for i, step in enumerate(np.flip(timesteps)):
            index = len(timesteps) - i - 1
            t = torch.full((b,), int(step), device=self.device, dtype=torch.long)
            e_t = self.unet(torch.cat([x, cond], dim=1), t)
            a_t = alphas[index].to(self.device)
            a_prev = alphas_prev[index].to(self.device)
            pred_x0 = (x - (1.0 - a_t).sqrt() * e_t) / a_t.sqrt()
            x = a_prev.sqrt() * pred_x0 + (1.0 - a_prev).sqrt() * e_t  # eta = 0
        decoded = self.first_stage.decode(x)
        predicted = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        return (1.0 - mask) * image + mask * predicted
