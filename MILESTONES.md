# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-25 00:00] - core_claim_plus
- Implemented and ran the actual DISPATCH regenerate-and-rectify pipeline on a scaled synthetic object-detection benchmark.
- Reproduced the core hiding-defense claim and an additional creating-attack suppression setting: DISPATCH improves hiding attack mAP.5/AR from 0.000/0.000 to 0.545/0.545 and reduces creating ASR from 0.325 to 0.250.
- Kept evaluation method-agnostic: `eval/` reads method output images and writes `scoring/scores.json` without importing from `method/`.

## Stop Justification
- Completed at milestone `core_claim_plus`.
