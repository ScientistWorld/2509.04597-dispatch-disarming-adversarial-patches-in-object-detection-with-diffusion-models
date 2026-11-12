"""Run the unchanged DISPATCH computation at a fixed inference seed."""
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, '/home/user/method')
from dispatch import dispatch_image
from diffusion import DDPM, TinyInpaintUNet

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--seed', type=int, required=True)
a = p.parse_args()
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.set_num_threads(1)
state = torch.load(a.checkpoint, map_location='cpu')
model = TinyInpaintUNet(base=state['base_channels'])
model.load_state_dict(state['model'])
ddpm = DDPM(timesteps=state['timesteps'])
for variant in ('clean', 'hiding', 'creating'):
    out = Path('/home/user/outputs/dispatch_images') / variant
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted((Path('/home/user/data/synthetic/test') / variant).glob('*.png')):
        result, _, _ = dispatch_image(Image.open(path).convert('RGB'), model, ddpm, 16, 32,
                                      torch.device('cpu'), prediction_type=state['prediction_type'])
        result.save(out / path.name)
    print('inferred', variant, flush=True)
