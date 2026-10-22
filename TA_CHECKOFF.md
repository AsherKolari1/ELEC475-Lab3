# Lab 3 In-Lab — TA Checkoff

Verified locally on October 22, 2026: evaluation exited successfully, both tables
contain all six required rows, no TODO functions were skipped, and all four
required PNG files passed image-integrity checks.

## Accuracy and cost tables

```text
Model                   Min    Mean  Median     Max     Std
Baseline (mean)         0.6    79.7    70.0   848.2    56.1
SnoutNet                0.9    57.0    44.3  1084.5    57.4
SnoutNet-A              1.8    44.8    36.9   690.3    39.6
SnoutNet-V              0.4    12.8     8.7   175.4    17.3
Ensemble (mean)         0.7    31.2    23.5   597.3    33.7
Ensemble (median)       0.4    29.4    20.2   676.4    38.0

Model                     Params   ms/img
Baseline (mean)                0      0.0
SnoutNet               5,617,794      1.4
SnoutNet-A            57,012,034      5.2
SnoutNet-V           134,268,738     12.6
Ensemble (mean)      196,898,566     19.2
Ensemble (median)    196,898,566     19.2
```

[Full saved terminal output](runs/ta_checkoff/evaluation.txt)

## Required figures

- [Cumulative error distribution — all six methods](plots/ced.png)
- [SnoutNet — four best and four worst predictions](plots/best_worst_snoutnet.png)
- [SnoutNet-A — four best and four worst predictions](plots/best_worst_alexnet.png)
- [SnoutNet-V — four best and four worst predictions](plots/best_worst_vgg16.png)

Open this file's Markdown preview in VS Code to access the tables and figures.
Show the figures as well as the tables to the TA.

## Rubric checklist

| Required item | Verified | Marks available |
|---|---|---:|
| Accuracy/cost tables: baseline and three models | Yes | 1 |
| CED plot | Yes | 1 |
| Three best/worst figures | Yes | 0.5 |
| Accuracy/cost tables: both ensembles | Yes | 0.5 |
| Total | Required outputs present; TA determines marks | 3 |

VGG16 has the lowest mean error (12.8 px). The mean and median ensembles have
mean errors of 31.2 px and 29.4 px respectively; combining models does not guarantee
beating the strongest individual model. Each ensemble requires all three models,
so its parameter count and measured inference cost are their sums.

To regenerate the live terminal tables and plots from saved predictions:

```bash
python evaluate.py
```

[Machine-readable verification](runs/ta_checkoff/verification.json)
