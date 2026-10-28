# Lab 3 Post-Lab — Logs and Results

Open this file in VS Code Markdown preview for quick access by post-lab section.
All links below refer to locally saved artifacts.

| Section / purpose | Saved log or result |
|---|---|
| Step 2 — model architecture summaries | [step2_model_summaries.txt](runs/postlab/step2_model_summaries.txt) |
| Step 3 — training without augmentation | [20261028_113110_colab_step3_step3_train_correction_noaug.txt](runs/postlab/step3_training/20261028_113110_colab_step3_step3_train_correction_noaug.txt) |
| Step 3 — training with augmentation | [20261028_113110_colab_step3_step3_train_correction_aug.txt](runs/postlab/step3_training/20261028_113110_colab_step3_step3_train_correction_aug.txt) |
| Step 3 — results and settings | [STEP3_RESULTS.md](runs/postlab/STEP3_RESULTS.md) |
| Step 4 — correction without augmentation | [step4_correct_correction_noaug.txt](runs/postlab/step4_20261028_130412/step4_correct_correction_noaug.txt) |
| Step 4 — correction with augmentation | [step4_correct_correction_aug.txt](runs/postlab/step4_20261028_130412/step4_correct_correction_aug.txt) |
| Step 4 — evaluate.py tables and plot output | [step4_evaluate.txt](runs/postlab/step4_20261028_130412/step4_evaluate.txt) |
| Step 4 — results summary | [STEP4_RESULTS.md](runs/postlab/step4_20261028_130412/STEP4_RESULTS.md) |
| Step 4 — prediction verification | [prediction_verification.json](runs/postlab/step4_20261028_130412/prediction_verification.json) |
| Step 4 — completion verification | [verification.json](runs/postlab/step4_20261028_130412/verification.json) |
| Step 5 — original and modified architecture summaries | [step5_model_summaries.txt](runs/postlab/step5_20261028_131536/step5_model_summaries.txt) |
| Step 5 — modified training without augmentation | [step5_train_correction_mod_noaug.txt](runs/postlab/step5_20261028_131536/step5_train_correction_mod_noaug.txt) |
| Step 5 — modified training with augmentation | [step5_train_correction_mod_aug.txt](runs/postlab/step5_20261028_131536/step5_train_correction_mod_aug.txt) |
| Step 5 — modified correction without augmentation | [step5_correct_correction_mod_noaug.txt](runs/postlab/step5_20261028_131536/step5_correct_correction_mod_noaug.txt) |
| Step 5 — modified correction with augmentation | [step5_correct_correction_mod_aug.txt](runs/postlab/step5_20261028_131536/step5_correct_correction_mod_aug.txt) |
| Step 5 — final evaluate.py tables and plots | [step5_evaluate.txt](runs/postlab/step5_20261028_131536/step5_evaluate.txt) |
| Step 5 — results summary | [STEP5_RESULTS.md](runs/postlab/step5_20261028_131536/STEP5_RESULTS.md) |
| Step 5 — verification | [verification.json](runs/postlab/step5_20261028_131536/verification.json) |
| Step 5 — accuracy CSV | [step5_accuracy.csv](runs/postlab/step5_20261028_131536/step5_accuracy.csv) |
| Combined chronological terminal output | [terminal_output.txt](runs/postlab/terminal_output.txt) |

## Figures

- Step 1: [patches](plots/correction_patches.png) and [augmented patches](plots/correction_patches_aug.png).
- Step 3: [loss without augmentation](plots/loss_correction_noaug.png) and [loss with augmentation](plots/loss_correction_aug.png).
- Steps 4–5: [CED — 18 methods](plots/ced.png).
- Step 4: [SnoutNet without correction](plots/best_worst_snoutnet.png), [corrected without augmentation](plots/best_worst_snoutnet_correction_noaug.png), and [corrected with augmentation](plots/best_worst_snoutnet_correction_aug.png).

The initial Step 1 terminal run predated the combined logging request; its preview
figures are available above. The original in-lab outputs and log index remain in
[TA_CHECKOFF.md](TA_CHECKOFF.md).

## Current status

Steps 1–5 are implemented and all required experiments and outputs have been generated. Step 3 training
used Colab A100; Step 4 correction/timing used the local RTX 3050 to match the
original first-stage timing hardware. No Colab session was started for Step 4.

| First-stage model | Original mean error | Corrected, no augmentation | Corrected, augmentation |
|---|---:|---:|---:|
| SnoutNet | 57.0 px | 43.7 px | 43.6 px |
| VGG16 | 12.8 px | 12.1 px | 11.8 px |
| Mean ensemble | 31.2 px | 20.5 px | 20.4 px |

Step 5 training and correction used the local RTX 3050. The report and submission package remain to be completed.
Correction improves these mean errors but does not improve every individual image.

Logs under `runs/` are local and Git-ignored. This index does not make them part of
a Git commit or submission automatically; package the required report/artifacts separately.

## Step 5 loss curves

- [Modified model, no augmentation](plots/loss_correction_mod_noaug.png)
- [Modified model, with augmentation](plots/loss_correction_mod_aug.png)

