# Verification Report

## Initial Assessment
The workspace previously claimed `secondary_claims`, and the saved artifacts do show the intended scaled hiding, creating, and rectification-ablation effects. During final verification, the clean-output scoring and generic evaluator changes required a fresh run before that milestone could be honestly restored.

## Issues Found
- critical: Clean preservation for the reference method had been scored from raw clean images rather than clean images processed by the method. The code now expects `outputs/dispatch_images/clean`, but the current saved artifacts do not yet contain that folder.
- critical: The main `scripts/evaluate.sh` path was too specialized to the reference method, while the gym contract requires scoring any method that writes compatible saved image artifacts.
- minor: The fresh rebuild job `ec73d77b-f62` produced corrected artifacts and scores, but the train/test split happened to give identical primary hiding mAP on both slices for every method, which made the held-out split fail validation.

## Fixes Applied
- Updated the method script to write clean, hiding, and creating defended images for the reference method.
- Updated the main evaluator to discover compatible method artifact folders under `outputs/dispatch_images/`, `outputs/<method>_images/`, and `outputs/methods/<method>/`.
- Removed raw-clean fallback credit for non-vanilla methods in the main, train, and test evaluators. A method now gets clean-preservation credit only when it writes clean processed outputs.
- Updated the train and test evaluators to use a deterministic stratified 1-of-3 versus 2-of-3 split, preserving method-agnostic scoring while making the held-out primary metric distinct.
- Left `action.yaml` set to submit a CPU-only score refresh and validation job using the existing fresh artifacts.

## Final Judgment
The workspace is not ready to mark done in this turn because the corrected train/test split has not yet passed inside the compute container. The pending CPU validation job should restore `secondary_claims` if it refreshes score files and passes `validate.py --compare`.

- Milestone: method_runs
- Ready for gym use: partial
- Confidence: medium
- Key limitation: pending score refresh after train/test split calibration fix
