# Reproduction Checklist

- [x] `briefing/problem.md`, `briefing/evaluation.md` (method-agnostic), `briefing/method.md`, `briefing/overview.md`
- [x] `scoring/reference.json` (paper numbers kept; scaled experiments noted), `TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`, `EXPERIMENTS.md`
- [x] `scoring/scores.json`, `scores_train.json`, `scores_test.json` from the reference runs
- [x] `eval/train`, `eval/test` (independent), `defenses/` contract, `method/` (paper's DISPATCH + pretrained LDM)
- [x] `scripts/download.sh` (INRIA-Person, LDM checkpoint, detector weights), `evaluate*.sh`, `reproduce.sh`, `run.sh`
- [x] `environment/container.def`, `environment/setup.sh` (CUDA PyTorch)
- [x] `HARDENING.md`
- [x] `python validate.py --post-design` passes
