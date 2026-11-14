"""Replay the gym's rank formula on saved scores (analysis only, not part of the gym).

rank(m) = sum_exp weight_exp * sum_metric |coef| * dir * (v_m - v_ref) / sd_metric

dir = +1 for higher-is-better, -1 otherwise; v_ref is the reference method
(dispatch, seed 0); sd_metric is the population standard deviation across a
cohort of attempts.  Two cohorts are reported:
  honest  - every run that passed the evaluator (no failure, no constraint
            violation), reference included, the undefended detector excluded;
  all     - additionally the runs scored as failed (worst values).
Usage: python redteam/hardening/rank_check.py <reference_scores.json> <probe_scores.json> <status.json>...
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ref_scores, probe_scores = (json.loads(Path(p).read_text())["experiments"] for p in sys.argv[1:3])
    status = {}
    for p in sys.argv[3:]:
        status.update(json.loads(Path(p).read_text()))
    spec = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]

    methods: dict[str, dict] = {}
    for scores in (ref_scores, probe_scores):
        for exp, payload in scores.items():
            for m, vals in payload["results"].items():
                methods.setdefault(m, {})[exp] = vals
    ok = {m for m in methods if status.get(m, "ok") == "ok" and m != "vanilla"}

    def sd(exp, metric, cohort):
        vals = [methods[m][exp][metric] for m in cohort]
        s = statistics.pstdev(vals)
        return s if s > 1e-9 else 1.0

    out = {}
    for name, cohort in (("honest", sorted(ok)), ("all", sorted(m for m in methods if m != "vanilla"))):
        ranks = {}
        for m in methods:
            total = 0.0
            for exp, e in spec.items():
                for metric, md in e["metrics"].items():
                    d = 1.0 if md["higher_is_better"] else -1.0
                    total += e["weight"] * abs(md["coefficient"]) * d * (
                        methods[m][exp][metric] - methods["dispatch"][exp][metric]) / sd(exp, metric, cohort)
            ranks[m] = round(total, 4)
        out[name] = dict(sorted(ranks.items(), key=lambda kv: -kv[1]))
        out[name + "_cohort"] = cohort
        out[name + "_sd"] = {f"{exp}.{metric}": round(sd(exp, metric, cohort), 4) for exp, e in spec.items() for metric in e["metrics"]}
    seeds = [out["honest"][m] for m in ("dispatch", "seed1", "seed2") if m in out["honest"]]
    out["honest_seed_range"] = round(max(seeds) - min(seeds), 4)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
