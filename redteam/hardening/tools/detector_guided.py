"""Private-search probe: pick, per image, the candidate the protected detector likes best.

Candidates: the input and DISPATCH outputs with seeds 0, 1, 2.  The method runs
its *own copy* of the public Faster R-CNN weights (copied next to this probe
before the evaluation, as a determined method could bundle them) and keeps the
candidate maximising  sum(person scores >= 0.5) - 2 * sum(non-person scores >= 0.3).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "redteam/hardening/defenses/sc_detector_guided/weights.pth"


def detector(device):
    from torchvision.models import resnet50
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection._utils import overwrite_eps
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops import misc as misc_nn_ops

    bb = _resnet_fpn_extractor(resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d), 3)
    m = FasterRCNN(bb, num_classes=91, min_size=640, max_size=640, box_score_thresh=0.05)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    overwrite_eps(m, 0.0)
    return m.eval().to(device)


@torch.no_grad()
def utility(model, arr: np.ndarray, device) -> float:
    r = model([torch.from_numpy(arr).permute(2, 0, 1).float().div(255).to(device)])[0]
    s, l = r["scores"].cpu().numpy(), r["labels"].cpu().numpy()
    return float(s[(l == 1) & (s >= 0.5)].sum() - 2.0 * s[(l != 1) & (s >= 0.3)].sum())


def main() -> None:
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".detguided_", dir=out_dir.parent))
    cands = [in_dir]
    for seed in (0, 1, 2):
        d = work / f"s{seed}"
        subprocess.run([sys.executable, str(ROOT / "method/dispatch.py"), "--input-dir", str(in_dir), "--output-dir", str(d),
                        "--seed", str(seed)], check=True, cwd=ROOT)
        cands.append(d)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{max((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count()))[1]}")
    model = detector(device)
    picks = [0] * len(cands)
    for p in sorted(in_dir.glob("*.png")):
        arrs = [np.asarray(Image.open(c / p.name).convert("RGB")) for c in cands]
        best = int(np.argmax([utility(model, a, device) for a in arrs]))
        picks[best] += 1
        Image.fromarray(arrs[best]).save(out_dir / p.name)
    print(f"detector-guided picks (input, s0, s1, s2): {picks}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
