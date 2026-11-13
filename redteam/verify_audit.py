"""Lightweight evidence/restoration checks; no model or evaluator execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'redteam'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_tree(current, saved):
    current_files = {p.relative_to(current): p for p in current.rglob('*') if p.is_file()}
    saved_files = {p.relative_to(saved): p for p in saved.rglob('*') if p.is_file()}
    differing = sorted(str(p) for p in current_files.keys() & saved_files.keys()
                       if digest(current_files[p]) != digest(saved_files[p]))
    return {'files': len(saved_files), 'missing': sorted(map(str, saved_files.keys() - current_files.keys())),
            'extra': sorted(map(str, current_files.keys() - saved_files.keys())), 'differing': differing}


def main():
    hashes = {name: digest(ROOT / name) == expected
              for name, expected in read(AUDIT / 'initial_hashes.json').items()}
    assert all(hashes.values()), hashes
    trees = {
        'outputs': compare_tree(ROOT / 'outputs', AUDIT / '.state/outputs'),
        'data/synthetic': compare_tree(ROOT / 'data/synthetic', AUDIT / '.state/synthetic'),
    }
    for value in trees.values():
        assert not any(value[key] for key in ('missing', 'extra', 'differing')), value
    scores = {p.name: digest(ROOT / 'scoring' / p.name) == digest(p)
              for p in (AUDIT / '.state/scoring').iterdir() if p.is_file()}
    assert all(scores.values()), scores
    assert read(AUDIT / 'reference/scores_test.json') == read(ROOT / 'scoring/scores_test.json')
    cases = read(AUDIT / 'bundle_status.json')['cases']
    exits = {}
    for name in cases:
        folder = AUDIT / name
        for filename in ('run.sh', 'scores_train.json', 'scores_test.json', 'train_log.txt', 'test_log.txt', 'execution.json'):
            assert (folder / filename).is_file(), (name, filename)
        execution = read(folder / 'execution.json')
        assert execution['test_command'] == 'bash scripts/evaluate_test.sh'
        exits[name] = execution['test_exit']
    assert len(cases) == 20 and sum(value == 0 for value in exits.values()) == 19
    assert exits['shortcut_stale_failure'] != 0
    ranks = read(AUDIT / 'ranking.json')
    paying = [name for name, value in ranks['ranks'].items() if name.startswith('shortcut_') and value >= 0]
    assert len(paying) == 9, paying
    assert ranks['longer_training_gain_over_matched_seed0'] < ranks['seed_rank_spread']
    assert not any(spec['reference_at_best_bound'] for spec in ranks['metrics'])
    # System/preexisting changes are recorded, not silently overwritten.
    excluded = (':!redteam/**', ':!REDTEAM.md', ':!README.md', ':!manifest.json', ':!trajectory/**', ':!action.yaml')
    diff = subprocess.check_output(['git', 'diff', '--name-status', 'f942a52', '--', '.', *excluded], cwd=ROOT, text=True)
    assert not diff, diff
    # Check local links now; restoration.json is written below.
    for target in re.findall(r'\]\(([^)]+)\)', (ROOT / 'REDTEAM.md').read_text()):
        if target == 'redteam/restoration.json':
            continue
        assert (ROOT / target).exists(), target
    result = {
        'iteration_start_commit': 'f942a52',
        'initial_hash_matches': hashes,
        'artifact_trees_match_initial_snapshot': trees,
        'score_file_matches_initial_snapshot': scores,
        'reference_reevaluation_matches_original_scores': True,
        'original_job_launcher_restored_with_git': (ROOT / 'scripts/run.sh').read_bytes() == (AUDIT / 'original_run.sh').read_bytes(),
        'gym_diff_outside_audit_and_system_paths': diff,
        'successful_hidden_tests': 19,
        'deliberate_failed_hidden_tests': 1,
        'paying_shortcut_cases': paying,
        'notes': 'README.md, manifest.json and trajectory had preexisting/system-owned changes; action.yaml is required completion metadata. Current-iteration job logs were relocated under redteam/job_logs and root jobs/ restored with git.',
    }
    assert result['original_job_launcher_restored_with_git']
    (AUDIT / 'restoration.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
