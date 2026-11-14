# Evaluation Targets

All three experiments are produced by one run of each defense on a mixed input
set (see `briefing/evaluation.md`).  A defense that fails, or whose clean or
object person mAP.5 is more than 0.05 below the undefended detector, gets the
worst value of every metric below.

## Primary: inria_digital_hiding (weight 0.40)

Hiding patches of the paper's default size (0.2 x box diagonal) on every person.

- **Primary metric:** `attack_mAP5` (person AP@0.5 on attacked images), higher is better.
- **Supporting:** `attack_AR` (AR@100, IoU 0.50:0.95).
- **Constraint metrics:** `clean_mAP5`, `clean_AR` on clean images (gate: `clean_mAP5`
  at most 0.05 below the undefended detector).
- Reference (hidden test): undefended 0.489 / DISPATCH 0.943 attack mAP.5; clean 0.976 / 0.966.

## Primary: creating_digital (weight 0.35)

Targeted, physical-style creating patches in background regions (APRICOT protocol).

- **Primary metric:** `untargeted_ASR`, lower is better.
- **Supporting:** `targeted_ASR`, `untargeted_mAP5`, `targeted_mAP5` (lower is better).
- **Constraint metric:** `object_mAP5` (person AP@0.5 on the same images; gate at -0.05).
- Reference (hidden test): undefended 0.992 / DISPATCH 0.658 untargeted ASR.

## Secondary (headroom): inria_hiding_large_patch (weight 0.25)

Hiding patches at 0.3 x box diagonal, where the paper reports every defense degrading
(Appendix A.5).

- **Primary metric:** `attack_mAP5`, higher is better.
- **Supporting:** `attack_AR`.
- Reference (hidden test): undefended 0.032 / DISPATCH 0.413.

## How to run

`bash scripts/evaluate.sh` scores every `defenses/*/defend.sh` on the public slice
and writes `scoring/scores.json` (identical metrics to the hidden test).
