# Hardening report

## What changed, in one paragraph

The audited gym was a 64x64 toy (colour-threshold "detector", rendered stick
figures, a tiny pixel DDPM trained from scratch) whose test inputs, labels,
clean counterparts and generator seeds were public, and which a 66-template
enumeration solved perfectly (`REDTEAM.md`).  Fixing that toy was not possible
without leaving the paper's problem, so the gym was rebuilt on the paper's own
data and method:
- **Data:** INRIA-Person, the paper's hiding benchmark.
- **Protected detector:** a fixed COCO-pretrained Faster R-CNN, one of the paper's detectors.
- **Hiding attacks:** white-box AdvPatch-style patches with EOT, plus a smooth naturalistic patch.
- **Creating attacks:** targeted patches with an APRICOT-style print-and-capture degradation.
- **Reference method:** the paper's actual DISPATCH, using the authors' pretrained LDM
  `inpainting_big` (VQ-f4 + EMA UNet, upstream DDIM, 512x512, N = 32, s = 5) with
  the paper's rectification (blurred L2 map, KMeans k = 2, Eq. 7).

The hidden test runs every `defenses/*/defend.sh` on 480 images it composes at
test time from patches and placements only `eval/test/` holds.  Everything below
was measured with the final code (`redteam/hardening/run_hardening.sh` for the
probes, `scripts/run.sh` for the final reference files); raw files are in
`redteam/hardening/results/`.

Ranks replay the specified formula (`redteam/hardening/rank_check.py`):
weight x |coefficient| x direction x (value - reference) / SD.  The SD is taken
over the 13 runs that passed the hidden evaluator, the reference included.
Failed runs score the worst value in every metric; under this SD that is
-26.76 for all of them.

## 1. Do better results score higher?

**Before** (red team, old gym).  A genuinely longer-trained reference
(120 vs 60 epochs) scored **-0.206** below the recorded reference.  Its gain over
the matched seed was 0.051, smaller than the 0.066 seed range, so sensitivity was
not established.  Weaker variants did score lower: one epoch -1.017, no
rectification -0.382, identity -2.429.

**After** (hidden test, 480 images, every method on the same inputs).  Columns
are hiding mAP.5 / AR, clean mAP.5, large-patch mAP.5 / AR, creating untargeted
/ targeted ASR, object mAP.5, and rank:

| method | hiding | clean | large patch | creating ASR | object | rank |
|---|---|---|---|---|---|---|
| **dispatch (reference, seed 0)** | 0.943 / 0.676 | 0.966 | 0.413 / 0.480 | 0.658 / 0.517 | 0.992 | 0 |
| dispatch seed 1 | 0.934 / 0.676 | 0.963 | 0.402 / 0.488 | 0.683 / 0.550 | 0.986 | -0.149 |
| dispatch seed 2 | 0.939 / 0.681 | 0.961 | 0.433 / 0.491 | 0.608 / 0.517 | 0.997 | +0.068 |
| better: 3 averaged regenerations | 0.953 / 0.692 | 0.966 | 0.466 / 0.507 | 0.650 / 0.533 | 0.994 | **+0.203** |
| better: two passes of DISPATCH | 0.950 / 0.678 | 0.941 | 0.567 / 0.520 | 0.442 / 0.317 | 0.983 | **+0.594** |
| better: mask dilated 8 px | 0.868 / 0.655 | 0.928 | 0.315 / 0.362 | 0.158 / 0.058 | 0.969 | **+0.667** |
| weaker: N = 16 grids | 0.922 / 0.674 | 0.953 | 0.438 / 0.459 | 0.758 / 0.633 | 0.974 | **-0.681** |
| simple: Gaussian blur | 0.686 / 0.601 | 0.966 | 0.079 / 0.227 | 0.983 / 0.967 | 0.988 | -2.579 |
| simple: resize to 224 and back | 0.623 / 0.608 | 0.974 | 0.061 / 0.193 | 0.992 / 0.983 | 0.997 | -2.595 |
| no defense (vanilla) | 0.489 / 0.536 | 0.976 | 0.032 / 0.151 | 0.992 / 0.992 | 0.991 | -3.190 |
| paper ablation: regenerated only | gated (ungated: 0.869 / 0.635) | 0.915 | 0.310 / 0.370 | 0.142 / 0.050 | 0.956 | -26.762 |
| paper ablation: DISPATCH(G) | gated (ungated: 0.957 / 0.697) | 0.924 | 0.604 / 0.581 | 0.542 / 0.492 | 0.983 | -26.762 |

- **Seed noise.** The reference's three seeds span a rank of **0.217**.
- **Three averaged regenerations.**
  - Per metric, it beats the reference by more than the seed range on hiding mAP.5
    (+0.010 vs a seed range of 0.009), hiding AR (+0.016 vs 0.005), large-patch
    mAP.5 (+0.053 vs 0.031) and large-patch AR (+0.027 vs 0.011).
  - Its aggregate gain (+0.20) is about one seed range, because most aggregate
    noise comes from creating ASR (seed range 0.075 on 120 images).
  - On the public slice it is also above the reference (+0.044).
- **Two passes and mask dilation** gain far more than the seed noise.
  - Each trades clean mAP.5 for much better large-patch or creating results,
    and each stays inside the clean gate on the hidden test.
  - Each fails the gate on the public slice, where its clean loss is larger.
    These are honest alternatives that a scientist may find, not shortcuts.
- **Weaker variants** score lower. N = 16 grids loses, as the paper's Table 7
  says; the simple baselines and no defense lose far more.

## 2. Major shortcuts: fix, and score before / after

"Before" ranks are the red team's (old gym, 19-run cohort).  "After" ranks are
on the hidden test of the new gym.  Every shortcut was rerun through the hidden
evaluator as a real defense under `redteam/hardening/defenses/`.

| shortcut | fix | before | after |
|---|---|---|---|
| clean copy (copy the public clean counterpart) | No input has a clean counterpart: each source image is used for one variant only. Inputs are composed at test time, and sources are locked while a defense runs. `sc_read_answers` recovered 0 of 480 clean sources. | +0.653 | -3.190 (= no defense) |
| visible labels / ground-truth replacement | Labels exist only in `eval/test/assets` (pinned, unreadable during defense runs). Methods are scored only on the detector's output on their images. | +1.210 | -3.190 |
| rebuild answers from generator seeds | No generator: real INRIA images; hidden salt, crops, flips, ids and patches live only in `eval/test/`. | +1.210 | -3.190 |
| brute-force template enumeration | Real images and a learned detector leave nothing to enumerate. Template-matching the released patches gives +3.257 on the public slice and nothing on the hidden test, a train-tuned method losing on test. | +1.210 | -3.169 |
| selective omission / blank creating only | Inputs from all experiments are mixed under opaque ids, so a defense cannot answer only one experiment. Any missing output fails the run, and a failed run gets the worst value in every metric. | +0.093 / +0.067 | -26.762 (`sc_partial`) |
| empty allowlist in `scores_train.json` | The test discovers methods from `defenses/*/defend.sh` and never reads `scores_train.json`. Emptying it changes nothing (`sc_empty_allowlist` is scored normally). | 0 (tie) | -3.190 |
| replace fixed inputs | Inputs are composed from pinned sources (SHA-256 of every source image, the detector weights and the patches). Altering one source image made every defense fail (`tamper_demo_status.json`). | +0.653 | failed (worst values) |
| stale scores after a crash / forged score files | Every failure (exit code, timeout, missing, unreadable or mis-sized output, any evaluator exception) is recorded with worst values, and the score file is always rewritten. `sc_crash_forge` forged perfect files, then crashed: worst values, forgery overwritten. | no valid score | -26.762 |
| blank / destroy-everything | Hard constraint: clean or object person mAP.5 more than 0.05 below no defense means a failed run, with worst values everywhere. | -1.017 | -26.762 |

## 3. Shortcuts left, and why

- **Detector-guided private search** (`sc_detector_guided`, rank +0.234 vs seed
  range 0.217).
  - It runs its own copy of the public Faster R-CNN weights and picks, per
    image, among the input and three DISPATCH seeds.
  - It gains only clean/object mAP.5 (by returning the untouched input) and
    creating ASR within seed noise; hiding and large-patch scores are *lower*.
  - The evaluator's own weights are locked during defense runs, but a method
    can bundle a copy. Closing this fully needs a private detector.
  - Left to the grader: `DIRECTION.md` forbids querying any detector.
- **Public-dataset lookup.**
  - INRIA-Person is public, so a defense that ships INRIA and matches inputs to
    their sources could recover clean images.
  - Mitigations:
    - the INRIA directory is unreadable during defense runs;
    - the inputs are randomly cropped, flipped and renamed, using a hidden salt;
    - `DIRECTION.md` forbids it.
  - It cannot be closed without non-public images.
- **Cross-image exploitation.** Each hidden patch appears on about 40 inputs of one
  run. A transductive method could find the repeated pattern. `DIRECTION.md`
  requires per-image processing; the grader must check.
- **Chmod locks.** They stop accidental or casual reads. A method running as the
  same user could undo them, which is a visible tamper for the grader.

## 4. Reference score before / after

- **Before (old toy gym).**
  - Hiding recall 0.493, AR 0.370, precision 0.514; clean recall 0.836.
  - Creating ASR 0.058, object recall 0.753; ablation recall 0.493.
- **After (new gym, hidden test).** The run is deterministic and reproduced
  exactly in two jobs.
  - Hiding mAP.5 0.943, AR 0.676; clean mAP.5 0.966, AR 0.697.
  - Large-patch mAP.5 0.413, AR 0.480.
  - Creating: untargeted ASR 0.658, targeted ASR 0.517, untargeted mAP.5 0.808,
    targeted mAP.5 0.651, object mAP.5 0.992.
  - Public slice: hiding 0.903, large patch 0.490, creating ASR 0.740, clean 0.968.
- **Against the paper** (Faster R-CNN, AdvPatch, Table 1): vanilla
  0.510 -> DISPATCH 0.932 attacked mAP.5, and clean 0.963 -> 0.965.
  Reproduced: 0.489 -> 0.943, and clean 0.976 -> 0.966.

## 5. Headroom

- **No scored metric sits at its bound** for the reference.
- **Hiding attack mAP.5** (0.943) is close to the clean level (0.966).
  - Its headroom is small; AR has more (0.676).
  - This is why I **added `inria_hiding_large_patch`**, from the paper's own
    Appendix A.5, where defenses fall short: the reference reaches 0.413, the
    two-pass variant 0.567 and gray fill 0.604.
- **Creating** has wide headroom: the reference's ASR is 0.658, while mask
  dilation reaches 0.158.
- **No simple variant reaches a bound.** Simple baselines sit far below the
  reference; it sits far below every bound except the guard metric
  `object_mAP5`.
- **Brute force does not solve the test.**
  - Enumerating templates needs the patches, which are hidden.
  - Searching candidates with the detector stays within noise of the reference.

## 6. Weights and coefficients

| experiment | weight | metric coefficients |
|---|---|---|
| hiding | 0.40 | attack_mAP5 1.0, attack_AR 0.5, clean_mAP5 0.5, clean_AR 0.15 |
| large patch (new) | 0.25 | attack_mAP5 1.0, attack_AR 0.5 |
| creating | 0.35 | untargeted_ASR 1.0, targeted_ASR 0.6, untargeted_mAP5 0.6, targeted_mAP5 0.4, object_mAP5 0.1 |

- The rectification-ablation experiment was dropped as a scored experiment; it
  re-scored the hiding predictions a second time.
- The clean side of the paper's claim is now enforced by the hard gate. Its
  scored coefficients are moderate (clean 0.5, object 0.1):
  - Honest defenses differ there only by seed-level noise (±0.005 AP).
  - With SD normalisation, larger coefficients let that noise dominate the rank.
    At clean 1.0 / object 0.5, the reference's own seed spread was 0.37.

## 7. Not fixed, stated plainly

- **Appendix A.2 does not reproduce under attack.**
  - DISPATCH(G), gray fill, beats DISPATCH on attacked images in this
    reproduction, e.g. large patches 0.604 vs 0.413.
  - It loses 0.052 clean mAP.5 on the hidden test (0.040 on the public slice),
    so the gate fails it on the hidden test but not on the public slice; on the
    public slice it ranks -0.32.
  - A method that removes more aggressively sits close to that gate. The gate
    threshold (0.05) was set before these runs and was not tuned to them.
- **Appendix A.1 does reproduce.**
  - Rectification beats the regenerated image on hiding (0.943 vs 0.869) and
    clean (0.966 vs 0.915).
  - Full regeneration suppresses creating patches better (ASR 0.142), at a
    clean cost that fails the gate.
- **The hidden test needs one GPU.** Each diffusion-based defense takes about
  4 minutes on an H100; on CPU it is infeasible.
- **Substitutions:**
  - torchvision Faster R-CNN at 640 px instead of MMDetection YOLOv3 / Faster
    R-CNN / DETR;
  - our own white-box patches (two AdvPatch-style, one smooth stand-in for
    NatPatch; no AdvTexture);
  - simulated physical creating patches instead of APRICOT photographs;
  - no physical, adaptive-attack or latency experiments.
  - The paper's numbers for these are kept unchanged in `scoring/reference.json`.
  - The old file's transcription of Table 3 (DISPATCH creating rows) was
    wrong, and was corrected from the paper's PDF text.
- **Creating ASR noise.** Creating ASR on 120 images has a seed range of 0.075,
  the largest single noise source.
- **`validate.py` warnings.** It prints soft warnings that the attack builders
  and print-and-capture composition use random-number generators. These are
  benchmark-construction code; scoring reads the defenses' outputs.
