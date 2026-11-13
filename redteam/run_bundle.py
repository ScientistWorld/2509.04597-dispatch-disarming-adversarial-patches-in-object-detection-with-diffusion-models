"""One CPU job; checkpointed individual cases survive a scheduler interruption."""
from pathlib import Path
import subprocess
import sys
from common import AUDIT, ROOT, reset, dump

# Cheap audit checks first, so results are saved before the longer training runs.
cases = ['reference', 'weak_identity', 'weak_no_rectification',
 'shortcut_blank', 'shortcut_clean_copy', 'shortcut_visible_labels',
 'shortcut_rebuild_answers', 'shortcut_selective_omission', 'shortcut_empty_allowlist',
 'shortcut_cache_score_forgery', 'shortcut_stale_failure', 'shortcut_ground_truth',
 'shortcut_replace_fixed_inputs', 'shortcut_bruteforce_templates', 'shortcut_blank_creating_only',
 'better_longer_training', 'weak_one_epoch', 'reference_seed0', 'reference_seed1', 'reference_seed2']
failed = []
for name in cases:
    evidence = AUDIT / name / 'execution.json'
    if evidence.exists():
        print('already completed', name, flush=True)
        continue
    print('RUN', name, flush=True)
    result = subprocess.run(['bash', str(AUDIT / name / 'run.sh')], cwd=ROOT)
    if result.returncode:
        failed.append(name)
        print('FAILED', name, flush=True)
reset()
dump(AUDIT / 'bundle_status.json', {'cases': cases, 'failed': failed})
if failed:
    sys.exit(1)
