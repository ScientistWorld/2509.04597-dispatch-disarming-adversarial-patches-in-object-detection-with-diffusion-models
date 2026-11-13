"""Replay the stated rank with disclosed, measured audit-cohort deviations.

Analysis only: candidate implementations never import this file or reference.json.
Future attempts and their deviations are unknown, so these are illustrative ranks,
not predictions of the eventual production leaderboard's numerical values.
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / 'redteam'


def load(path):
    return json.loads(path.read_text())


def method_metrics(experiments, experiment, method='dispatch'):
    return experiments.get(experiment, {}).get('results', {}).get(method, {})


def main():
    metadata = load(ROOT / 'scoring/reference.json')['experiments']
    reference = load(AUDIT / 'reference/scores_test.json')['experiments']
    rows = {}
    excluded = []
    for name in load(AUDIT / 'bundle_status.json')['cases']:
        execution = load(AUDIT / name / 'execution.json')
        if execution['test_exit']:
            excluded.append(name)
            continue
        rows[name] = load(AUDIT / name / 'scores_test.json')['experiments']

    specs = []
    for experiment in reference:
        for metric, spec in metadata[experiment]['metrics'].items():
            values = [method_metrics(row, experiment).get(metric) for row in rows.values()]
            values = [value for value in values if isinstance(value, (float, int))]
            sd = statistics.pstdev(values)
            assert sd > 0, (experiment, metric)
            best = 1.0 if spec['higher_is_better'] else 0.0
            ref_value = method_metrics(reference, experiment)[metric]
            specs.append({'experiment': experiment, 'metric': metric,
                          'weight': metadata[experiment]['weight'],
                          'coefficient_magnitude': abs(spec['coefficient']),
                          'higher_is_better': spec['higher_is_better'],
                          'reference': ref_value, 'population_sd': sd,
                          'observed_attempts': len(values), 'best_bound': best,
                          'reference_at_best_bound': ref_value == best})

    ranks = {}
    contributions = {}
    for name, experiments in rows.items():
        terms = {}
        for spec in specs:
            experiment, metric = spec['experiment'], spec['metric']
            value = method_metrics(experiments, experiment).get(metric)
            if value is None:
                continue  # The stated gym rank skips missing entries.
            direction = 1.0 if spec['higher_is_better'] else -1.0
            terms[f'{experiment}/{metric}'] = (
                spec['weight'] * spec['coefficient_magnitude'] * direction
                * (value - spec['reference']) / spec['population_sd'])
        contributions[name] = terms
        ranks[name] = sum(terms.values())

    seeds = [f'reference_seed{seed}' for seed in range(3)]
    seed_ranges = []
    for spec in specs:
        experiment, metric = spec['experiment'], spec['metric']
        values = [method_metrics(rows[name], experiment)[metric] for name in seeds]
        seed_ranges.append({'experiment': experiment, 'metric': metric,
                            'values_seed012': values, 'minimum': min(values),
                            'maximum': max(values), 'spread': max(values) - min(values)})
    seed_ranks = [ranks[name] for name in seeds]
    result = {
        'formula': 'sum(weight * abs(coefficient) * direction * (value-reference) / population_sd); skip missing',
        'cohort': list(rows), 'excluded_failed_test_exits': excluded,
        'cohort_rule': 'one dispatch candidate per successful audit case; recorded reference included; baseline rows are not repeated as attempts',
        'metrics': specs, 'ranks': ranks, 'contributions': contributions,
        'seed_metric_ranges': seed_ranges, 'seed_ranks': seed_ranks,
        'seed_rank_spread': max(seed_ranks) - min(seed_ranks),
        'longer_training_gain_over_matched_seed0': ranks['better_longer_training'] - ranks['reference_seed0'],
        'longer_training_gain_over_recorded_reference': ranks['better_longer_training'],
    }
    (AUDIT / 'ranking.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, value in ranks.items():
        print(f'{name}: {value:+.8f}')
    print('three-seed rank spread:', result['seed_rank_spread'])
    print('longer training matched gain:', result['longer_training_gain_over_matched_seed0'])


if __name__ == '__main__':
    main()
