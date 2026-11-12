"""Every run installs a candidate, invokes train/test, records scores, restores."""
import argparse
from pathlib import Path
import shutil
from common import ROOT, AUDIT, STATE, reset, empty_outputs, evaluate, exec_logged, dump
from shortcut_candidates import prepare, mutate_after_train

p = argparse.ArgumentParser()
p.add_argument('name')
a = p.parse_args()
name = a.name
folder = AUDIT / name
folder.mkdir(parents=True, exist_ok=True)
reset()
try:
    if name == 'reference':
        evaluate(folder)
    elif name.startswith('reference_seed') or name in {'better_longer_training', 'weak_one_epoch'}:
        if name.startswith('reference_seed'):
            seed, epochs = int(name.removeprefix('reference_seed')), 60
        else:
            seed, epochs = 0, (120 if name == 'better_longer_training' else 1)
        ckpt = AUDIT / f'artifacts/seed{seed}_epoch{epochs}.pt'
        if not ckpt.exists():
            rc = exec_logged(['python', 'redteam/train_variant.py', '--seed', str(seed), '--epochs', str(epochs)], folder / 'training_log.txt')
            if rc:
                raise RuntimeError('training failed')
        empty_outputs()
        rc = exec_logged(['python', 'redteam/infer_variant.py', '--seed', str(seed), '--checkpoint', str(ckpt)], folder / 'inference_log.txt')
        if rc:
            raise RuntimeError('inference failed')
        # Cache images inside the ignored audit artifacts for repeatable evaluations.
        cached = AUDIT / 'artifacts' / name
        if cached.exists():
            shutil.rmtree(cached)
        shutil.copytree(ROOT / 'outputs/dispatch_images', cached)
        dump(folder / 'config.json', {'training_seed': seed, 'inference_seed': seed, 'epochs': epochs,
             'training_images': 2000, 'image_size': 64, 'grids': 16, 'steps': 32,
             'batch_size': 64, 'base_channels': 32, 'timesteps': 80, 'learning_rate': 0.0002})
        evaluate(folder)
    else:
        prepare(name)
        rc = exec_logged(['bash', 'scripts/evaluate_train.sh'], folder / 'train_log.txt')
        if rc:
            raise RuntimeError(f'train scoring failed for {name}')
        mutate_after_train(name)
        evaluate(folder, run_train=False, allow_failure=name == 'shortcut_stale_failure')
finally:
    reset()
