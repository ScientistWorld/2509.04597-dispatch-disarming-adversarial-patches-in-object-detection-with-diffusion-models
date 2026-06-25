# Verification Report

## Initial Assessment
The workspace previously claimed `secondary_claims`, and the saved artifacts do show the intended scaled hiding, creating, and rectification-ablation effects. During final verification, however, the latest committed evaluator changes require a fresh run before that milestone can be honestly restored.

## Issues Found
- critical: Clean preservation for the reference method had been scored from raw clean images rather than clean images processed by the method. The code now expects `outputs/dispatch_images/clean`, but the current saved artifacts do not yet contain that folder.
- critical: The main `scripts/evaluate.sh` path was too specialized to the reference method, while the gym contract requires scoring any method that writes compatible saved image artifacts.

## Fixes Applied
- Updated the method script to write clean, hiding, and creating defended images for the reference method.
- Updated the main evaluator to discover compatible method artifact folders under `outputs/dispatch_images/`, `outputs/<method>_images/`, and `outputs/methods/<method>/`.
- Removed raw-clean fallback credit for non-vanilla methods in the main, train, and test evaluators. A method now gets clean-preservation credit only when it writes clean processed outputs.
- Left `action.yaml` set to submit a CPU test rebuild that will regenerate fresh artifacts and refresh all score files under the corrected scoring contract.

## Final Judgment
The workspace is not ready to mark done in this turn because the corrected evaluator has not yet been validated against fresh artifacts. The pending CPU rebuild should restore `secondary_claims` if it reproduces the previously observed hiding recovery, creating suppression, and rectification-ablation improvements while scoring the processed clean split.

- Milestone: method_runs
- Ready for gym use: partial
- Confidence: medium
- Key limitation: pending fresh score refresh after clean-output scoring fix
