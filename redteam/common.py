"""Audit orchestration only; candidate code never imports the hidden evaluator."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
STATE = AUDIT / '.state'
VARIANTS = ('clean', 'hiding', 'creating')


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def initialize():
    if (STATE / 'ready').exists():
        return
    STATE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'outputs', STATE / 'outputs', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'data/synthetic', STATE / 'synthetic', dirs_exist_ok=True)
    (STATE / 'scoring').mkdir(exist_ok=True)
    for name in ('scores_train.json', 'scores_test.json', 'scores.json'):
        shutil.copy2(ROOT / 'scoring' / name, STATE / 'scoring' / name)
    (STATE / 'ready').write_text('initial artifact snapshot\n')


def reset():
    initialize()
    for target, source in ((ROOT / 'outputs', STATE / 'outputs'),
                           (ROOT / 'data/synthetic', STATE / 'synthetic')):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    for path in (STATE / 'scoring').iterdir():
        shutil.copy2(path, ROOT / 'scoring' / path.name)


def empty_outputs():
    shutil.rmtree(ROOT / 'outputs')
    (ROOT / 'outputs').mkdir()


def exec_logged(argv, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    return result.returncode


def evaluate(folder, run_train=True, allow_failure=False):
    """Each evaluation is a separate process, with no script arguments or hooks."""
    folder = Path(folder)
    train_rc = exec_logged(['bash', 'scripts/evaluate_train.sh'], folder / 'train_log.txt') if run_train else None
    if (ROOT / 'scoring/scores_train.json').exists():
        shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
    started = time.monotonic()
    test_rc = exec_logged(['bash', 'scripts/evaluate_test.sh'], folder / 'test_log.txt')
    if (ROOT / 'scoring/scores_test.json').exists():
        shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
    dump(folder / 'execution.json', {'train_exit': train_rc, 'test_exit': test_rc,
         'test_seconds': time.monotonic() - started,
         'test_command': 'bash scripts/evaluate_test.sh',
         'fresh_process': True, 'evaluator_modified': False})
    if not allow_failure and (test_rc != 0 or (train_rc is not None and train_rc != 0)):
        raise RuntimeError(f'evaluation failed: {folder}, train={train_rc}, test={test_rc}')
    print(f'{folder.name}: train={train_rc}, hidden_test={test_rc}', flush=True)


def candidate_dirs(variants=VARIANTS):
    for variant in variants:
        path = ROOT / 'outputs/dispatch_images' / variant
        path.mkdir(parents=True, exist_ok=True)
    return ROOT / 'outputs/dispatch_images'


def copy_input(source_variant=None):
    out = candidate_dirs()
    for variant in VARIANTS:
        for path in (ROOT / 'data/synthetic/test' / (source_variant or variant)).glob('*.png'):
            shutil.copy2(path, out / variant / path.name)


def reference_only():
    # The audit is expressly allowed to reuse trained reference image artifacts.
    # Remove auxiliary regeneration output so only the candidate is evaluated.
    empty_outputs()
    shutil.copytree(STATE / 'outputs/dispatch_images', ROOT / 'outputs/dispatch_images')
