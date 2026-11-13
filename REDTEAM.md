# Does this gym reward better research?

The gym fails this audit. A genuine increase in training produced a gain smaller than the seed spread, so reliable sensitivity to better methods is **not established**. Clearly weaker methods do score lower. More decisively, finite template enumeration solves every scored metric perfectly, and methods that omit the task can beat or tie the reference. The synthetic problem therefore does not establish that a higher score represents a better defense on the paper's problem.

All test metrics below come from actual invocations of the unchanged `bash scripts/evaluate_test.sh`, with no arguments. The historical reference is the original `scoring/scores_test.json`; [reevaluating its saved artifacts](redteam/reference/scores_test.json) reproduced it exactly. The different synthetic values embedded in `scoring/reference.json` are not the comparison reference.

The eventual normalization depends on future attempts, which are unavailable. For numerical comparisons I replayed the specified formula using population standard deviations across the **19 successful audit candidates**, including the reference once and excluding the deliberately failed run. Each candidate contributes its `dispatch` row; repeated vanilla rows are not additional attempts. [ranking.json](redteam/ranking.json) records the cohort, all 13 metric terms, deviations, contributions and seed ranges; [rank_audit.py](redteam/rank_audit.py) is analysis only. Reported ranks are relative to reference zero, not a claim to know the future leaderboard's numerical scale:

`rank = sum(experiment_weight * abs(coefficient) * direction * (value - reference) / metric_sd)`, skipping missing values, with direction +1 for higher-is-better and -1 otherwise.

**Quality sensitivity.** The stronger-budget probe continued the same seed-0 training trajectory from 60 to 120 epochs. It retained the model, training data, masks, objective, optimizer and inference procedure. Training loss fell from 0.0058964 to 0.0033084. Its rank improved by **0.051299** over matched seed 0, less than the **0.065519** three-seed range, and remained **0.206114 below the recorded reference**. More training is genuine method work, but lower training loss does not prove a better defense: hiding recall fell while creating suppression improved. This is an inconclusive positive-quality check, not proof that the rank reverses scientific quality.

| Variant; entry point | Hidden hiding recall / AR / precision ↑ | Clean recall ↑ | Creating ASR ↓ / object recall ↑ | Rank vs reference |
|---|---:|---:|---:|---:|
| [Recorded reference](redteam/reference/run.sh) | .493151 / .369863 / .514286 | .835616 | .057692 / .753425 | 0 |
| [Matched 60 epochs, seed 0](redteam/reference_seed0/run.sh) | .520548 / .420091 / .527778 | .808219 | .134615 / .753425 | −.257413 |
| [120 epochs, seed 0](redteam/better_longer_training/run.sh) | .465753 / .406393 / .548387 | .780822 | .115385 / .753425 | −.206114 |
| [One epoch](redteam/weak_one_epoch/run.sh) | 0 / 0 / 0 | 0 | 0 / 0 | −1.017315 |
| [Rectification removed](redteam/weak_no_rectification/run.sh) | .410959 / .292237 / .422535 | .835616 | .115385 / .780822 | −.382424 |
| [Identity, no defense](redteam/weak_identity/run.sh) | 0 / 0 / 0 | .753425 | .461538 / .753425 | −2.429159 |

Each entry point has its full `scores_test.json` alongside it. All three weaker variants lose more than the seed range. The rectification ablation reuses the original regenerated images, as authorized; it performs no retraining. A robust improvement above seed variation was not demonstrated by the longer-training probe.

**Seed spread and reproduction.** I trained the reproduced method independently at seeds 0, 1 and 2 for 60 epochs, and seeded inference separately with the matching seed. Each used 2,000 clean training images, base channels 32, 80 diffusion timesteps, batch size 64, AdamW at 0.0002, grids 16 and 32 inference steps. The original computation and preprocessing were retained; single-thread execution, cached input tensors and zero loader workers reduced CPU overhead. Full configs, evaluator logs and scores are in [seed 0](redteam/reference_seed0/), [seed 1](redteam/reference_seed1/) and [seed 2](redteam/reference_seed2/); epoch losses are in `redteam/training_history_seed*.json`.

| Hidden metric | Seed 0 | Seed 1 | Seed 2 | Max−min |
|---|---:|---:|---:|---:|
| Hiding attack recall (`attack_mAP5`) | .520548 | .561644 | .506849 | .054795 |
| Hiding attack AR | .420091 | .447489 | .424658 | .027397 |
| Hiding attack precision | .527778 | .630769 | .587302 | .102991 |
| Clean recall | .808219 | .780822 | .780822 | .027397 |
| Clean precision | .893939 | .876923 | .876923 | .017016 |
| Creating targeted mAP5 proxy | .096154 | .134615 | .096154 | .038462 |
| Creating targeted ASR proxy | .115385 | .134615 | .096154 | .038462 |
| Creating untargeted mAP5 proxy | .115385 | .134615 | .096154 | .038462 |
| Creating untargeted ASR proxy | .134615 | .134615 | .115385 | .019231 |
| Creating object recall | .753425 | .780822 | .753425 | .027397 |
| Normalized audit rank | −.257413 | −.231615 | −.191893 | **.065519** |

The three rectification-ablation metrics repeat the hiding attack metrics and have the same spreads. The seed range is an empirical noise check, not a confidence interval. All three retrained seeds rank below the historical artifact; its original training/inference RNG state was not recorded. Exact reevaluation of saved images establishes artifact reproducibility, not reproducibility of that historical score from a new training run.

On the synthetic benchmark, the reproduced pipeline beats its baseline in the paper's reported direction: hiding recall rises from vanilla **0 to .493151**, creating ASR falls from **.461538 to .057692**, and clean recall rises from **.753425 to .835616**. Rectification also improves over regeneration alone: recall **.410959 → .493151**, AR **.292237 → .369863**, precision **.422535 → .514286**. Thus the qualitative toy effect is present; the vanilla baseline is clearly below the reference and there is room above it.

The departures from the [paper](paper/2509.04597.md), especially Section 3.4 and the experimental settings, are substantial:

| Aspect | Paper | Reproduced gym |
|---|---|---|
| Inpainting backbone | Original MS-COCO pretrained latent diffusion inpainting model, no additional fine-tuning | Tiny pixel-space conditional UNet/DDPM trained from scratch on generated scenes, masked x0 MSE; no latent autoencoder or original weights |
| Regeneration | LDM with DDIM, 512×512, grids 32, 5 sampling steps | 64×64, grids 16, 32 steps over an 80-timestep schedule; DDIM-style skipped updates with freshly noised context |
| Rectification | Two complementary checkerboards, recombination, normalized RGB L2 discrepancy, smoothing, KMeans with two clusters, replacement of the high-discrepancy region | These outer computational steps are present in `method/dispatch.py`; matching them does not make the substituted backbone the original pretrained LDM |
| Data and attacks | INRIA-Person with optimized AdvPatch/NatPatch/AdvTexture; APRICOT creating attacks; physical and adaptive evaluations | 80 generated test scenes of one/two fixed-color people; hand-coded occlusion patches and creating triggers, with labels and exact clean counterparts shipped publicly |
| Protected detector | Pretrained object detectors including YOLOv3, Faster R-CNN, DETR, SSD and RetinaNet | Color thresholds and connected components; no learned protected detector checkpoint |
| Measurements | Detection mAP.5, AR, attack success, efficiency; average of three stochastic runs | Recall at IoU .5 called mAP5, recall averaged at .5/.6/.7, precision, four overlap thresholds on the same creating false-positive event; one saved output per input, no latency score |

The original hidden slice has 52 images and 73 people, selected deterministically from the 80 public test scenes by the same stratified hash procedure exposed in the train evaluator. Creating “targeted” and “untargeted” proxies differ by overlap threshold, not by attack target class. The paper's natural-image, learned-detector and computational-efficiency claims are not tested here. In particular, enumeration of 66 renderer shapes at every legal position achieves perfect results without diffusion or labels. This is stronger evidence than a mere concern about scale: the finite substitute task is too small and structured to validate progress on the paper's problem.

**Shortcut results.** Every row below is a real shortcut attempt with its own `run.sh`; the score link contains the exact hidden-test output. “Perfect” means all nine higher-is-better terms equal 1 and all four lower-is-better terms equal 0. A tie counts as a paying shortcut. The normalization caveat above applies to numerical ranks; perfect outputs, selective omission and the empty-results tie pay under any positive metric scaling.

| Shortcut; runnable entry point | What it does | Hidden test score vs reference | Works? |
|---|---|---|---|
| [shortcut_clean_copy](redteam/shortcut_clean_copy/run.sh) | Copies each public clean counterpart into all defended variants | [Scores](redteam/shortcut_clean_copy/scores_test.json): hiding recall/AR .753425, precision .859375; creating ASR .038462, object recall .753425; rank **+.653213** | **Works** in audit cohort; some clean metrics decline, so not dominance for every possible normalization |
| [shortcut_visible_labels](redteam/shortcut_visible_labels/run.sh) | Reads public test boxes and paints blue rectangles for the exposed detector | [Perfect](redteam/shortcut_visible_labels/scores_test.json); rank **+1.210138** | **Works**, dominates reference |
| [shortcut_rebuild_answers](redteam/shortcut_rebuild_answers/run.sh) | Rebuilds boxes from filenames and public `make_example(index, 'test')`, whose seed is index + 100000; reads neither annotation file nor clean counterparts | [Perfect](redteam/shortcut_rebuild_answers/scores_test.json); rank **+1.210138** | **Works**, dominates reference |
| [shortcut_bruteforce_templates](redteam/shortcut_bruteforce_templates/run.sh) | Enumerates renderer shapes and positions, ignores known patch colors, and paints inferred detector-colored boxes; uses no labels, paired clean answers or generator seeds | [Perfect](redteam/shortcut_bruteforce_templates/scores_test.json); rank **+1.210138** | **Works**, dominates reference; test is too small |
| [shortcut_selective_omission](redteam/shortcut_selective_omission/run.sh) | Supplies clean copies only for creating; supplies no hiding or clean output directories | [Scores](redteam/shortcut_selective_omission/scores_test.json): creating ASR .038462, object recall .753425; hiding and ablation entries missing; rank **+.092656** | **Works** under any positive scaling of present terms; hiding task omitted |
| [shortcut_empty_allowlist](redteam/shortcut_empty_allowlist/run.sh) | Leaves blank images on disk, then clears every `results` method list in mutable `scores_train.json` | [All three hidden `results` are empty](redteam/shortcut_empty_allowlist/scores_test.json); rank **0, tie** | **Works: paying shortcut**, no task solved |
| [shortcut_blank_creating_only](redteam/shortcut_blank_creating_only/run.sh) | Supplies constant black creating images and omits other variants | [Scores](redteam/shortcut_blank_creating_only/scores_test.json): all five creating metrics 0, hiding/ablation entries missing; rank **+.066993** | **Works** in audit cohort despite zero object recall; not dominance for every normalization |
| [shortcut_ground_truth](redteam/shortcut_ground_truth/run.sh) | Replaces visible ground truth with one constant box and patch region; paints the same rectangle for every input | [Perfect](redteam/shortcut_ground_truth/scores_test.json); rank **+1.210138** | **Works**, moves ground truth; the mutable stratification also changes the slice to 53 images |
| [shortcut_replace_fixed_inputs](redteam/shortcut_replace_fixed_inputs/run.sh) | Replaces raw hiding/creating test PNGs with clean counterparts, then returns input unchanged | [Scores](redteam/shortcut_replace_fixed_inputs/scores_test.json) match clean-copy shortcut; rank **+.653213** | **Works** in audit cohort; replaces supposedly fixed task artifacts |
| [shortcut_blank](redteam/shortcut_blank/run.sh) | Supplies constant black images for every variant | [All 13 metrics 0](redteam/shortcut_blank/scores_test.json); rank **−1.017315** | Fails to pay when every experiment is scored |
| [shortcut_cache_score_forgery](redteam/shortcut_cache_score_forgery/run.sh) | Blank images plus authored prediction caches, null/±1000 train values, and a forged test score file | [Hidden test recomputes all 13 as 0](redteam/shortcut_cache_score_forgery/scores_test.json); rank **−1.017315** | Fails; successful test does not trust those numeric values or caches |
| [shortcut_stale_failure](redteam/shortcut_stale_failure/run.sh) | Performs the same forgery, then deletes required hiding PNGs | [Test exits 1](redteam/shortcut_stale_failure/test_log.txt); prewritten score file remains, but **no valid test-written score** | Fails as a paying shortcut; exposes missing failure-value handling |

The brute-force shortcut tries **66 width/height templates and 8,316,000 positions across 240 input images**, or 34,650 positions per image; [search.json](redteam/shortcut_bruteforce_templates/search.json) records the count. It uses the visible renderer/color family and returns detector-specific rectangles, so it is a shortcut outside the stated research direction, not a better general defense. No latency metric penalizes enumeration. Label-derived renderings leave a one-column gap between adjacent boxes so connected components do not merge them; initial unsuccessful renderings and scores are retained with `initial_` prefixes.

Nine of the twelve tested shortcut kinds pay, including one tie. Candidate implementations use only future-visible sources or expressly authorized reuse of reference outputs. The paper and `reference.json` are used for audit analysis only. The candidates do not change detector thresholds, evaluator code, interpreter behavior, imports used by evaluation, or its process. There is no separate fixed protected-model checkpoint to retrain in this gym; replacement of fixed raw input artifacts is demonstrated instead. Merely moving a displayed vanilla baseline would not alter the specified absolute comparison against the recorded reference.

**Scoring contract checks.** Only three synthetic experiments and 13 metric terms are emitted by the hidden script. The INRIA, APRICOT and physical experiments still described in `reference.json` are not measured and therefore cannot contribute under the stated missing-value rule. No emitted reference metric is already at its best bound:

| Experiment; weight | Metric | Reference | Better bound | Coefficient magnitude |
|---|---|---:|---:|---:|
| Hiding; .25 | attack_mAP5 | .493151 | 1 | 1 |
| Hiding; .25 | attack_AR | .369863 | 1 | .5 |
| Hiding; .25 | attack_precision | .514286 | 1 | .35 |
| Hiding; .25 | clean_mAP5 | .835616 | 1 | .25 |
| Hiding; .25 | clean_precision | .910448 | 1 | .15 |
| Creating; .15 | targeted_mAP5 | .019231 | 0 | .4 |
| Creating; .15 | targeted_ASR | .019231 | 0 | .6 |
| Creating; .15 | untargeted_mAP5 | .038462 | 0 | .7 |
| Creating; .15 | untargeted_ASR | .057692 | 0 | 1 |
| Creating; .15 | object_mAP5 | .753425 | 1 | .25 |
| Rectification ablation; .10 | attack_mAP5 | .493151 | 1 | .8 |
| Rectification ablation; .10 | attack_AR | .369863 | 1 | 1 |
| Rectification ablation; .10 | attack_precision | .514286 | 1 | .35 |

Directions in the metadata match these metric definitions. Creating coefficients are negative, but the specified formula correctly uses their magnitudes and obtains direction from `higher_is_better`; their signs do not reverse the lower-is-better requirement. Dividing each term by its across-attempt standard deviation removes units and makes the coefficient a relative weight. The ranking service itself is not in this workspace, so this audit replays the supplied contract rather than claiming to have tested its external implementation. Finite, bounded detector metrics survive the impossible-value forgery check.

The additive constraint check **fails**. In the blank-creating-only shortcut, suppression earns +.146048 while erasing every real object costs only −.079055, net +.066993. Hiding losses are absent. Object preservation is present as a metric but is not a requirement enforced by the sum. Even if different future deviations reversed this specific tradeoff, selective omission and the empty-results tie would still pay. The ablation experiment also scores the very same hiding predictions a second time, changing their effective weight rather than providing independent evidence.

Completeness and failure handling **fail**. The hidden evaluator uses keys from agent-writable `scoring/scores_train.json` as an experiment-specific allowlist. It does not take the union with methods found on disk. Clearing that list makes the hidden script successfully overwrite `scores_test.json` with empty results while blank candidate images remain present. Providing only creating outputs also succeeds without any `dispatch` hiding/ablation entry. These runs show that neither all methods nor all experiments are obligatorily written. A missing requested PNG instead raises `FileNotFoundError` before score writing, leaving stale scores in place. A failed run is not assigned the worst value for every metric/experiment. I exclude that exit-1 run from ranking because the leftover file was not written by its hidden test.

Numeric score forgery **fails on a successful evaluation**: PNGs are redetected, prediction caches are overwritten, train numeric fields are ignored, and an authored `scores_test.json` is replaced. The trusted weakness is method selection and mutable task data, not acceptance of authored detection scores or timing/success flags. Timings and success flags are not scored here.

The next gym should keep hidden labels and task inputs independent of agent-writable files, remove paired clean answers and replayable test seeds, require every submitted method to receive every scored metric, assign worst values on failure while replacing stale scores, and enforce preservation constraints before ranking benefits. A less finite natural-image/learned-detector problem with a scored resource budget is needed to rule out the demonstrated enumeration shortcut. Fixing score completeness alone would not fix the problem substitution.

**Execution and restoration.** Twenty candidate cases ran: 19 hidden evaluations exited successfully and one deliberately failed. Every case has a runnable `run.sh`, raw train/test score snapshots, test log and execution record. For shortcut cases, `train_exit: null` in `execution.json` means the public train evaluation ran successfully earlier in `scenario.py`, before the final file mutation; its output is in `train_log.txt`. Each hidden evaluation is a new subprocess after the method finishes and depends only on saved files; cases were bundled within scheduled containers, not submitted as 20 individual fresh-container jobs. No surviving method process or Python-environment change is needed by any shortcut.

Job logs are archived under `redteam/job_logs/`. The first training job was cancelled after repeated scheduler drains lost progress; atomic per-epoch model/optimizer/RNG resume allowed the replacement job to finish. The last short job added the blank-creating shortcut and replayed ranks from saved results. No heavy computation ran on the login node. All audit-authored method, data, scoring and launcher changes outside `redteam/` have been restored; [restoration.json](redteam/restoration.json) records hash and file-tree checks. Preexisting README/manifest/trajectory changes and scheduler-owned metadata are not audit edits. The required `action.yaml` is used only to signal completion. No reproduction milestone is advanced by this scoring audit.
