# Autoencoder experiment results

Profile: **pilot**. Generated from completed run artifacts; 6 trained runs.

Pilot training uses 30,000 train / 2,000 validation pairs, 12 epochs and a matched <=100k parameter budget. Pilot evaluation uses the first 16 committed test conditions, 256 model / reference samples, and a 20M proposal cap. These are exploratory results, not a replication of the paper tables.

Shared headline metrics remain raw-unit IMQ MMD² (unbiased, biased also saved), Gaussian reference widths 0.01 / 0.02, and clamp 10. No rounding/dequantization is used in training. Ballistics diagnostics include separate standardized and rounded-speed sensitivity scores.

Minimum accepted reference samples per condition: {'kinematics': 206, 'ballistics': 256}. The proposal cap can leave a shorter reference; the MMD estimator handles unequal sample counts.

## Evaluation

| Problem | Family | Ablation | Err_post | Err_resim | Failed | Total sampling ms |
|---|---|---|---|---|---|---|
| kinematics | standard | baseline | 0.07567 | 0.02047 | 0.000% | 2.1 |
| kinematics | cvae | baseline | 0.10428 | 0.05602 | 0.000% | 1.6 |
| kinematics | invertible | baseline | 0.09559 | 0.01694 | 0.000% | 3.1 |
| ballistics | standard | baseline | 0.01015 | 0.23530 | 0.024% | 2.5 |
| ballistics | cvae | baseline | 0.03253 | 0.12223 | 0.000% | 1.3 |
| ballistics | invertible | baseline | 0.00810 | 0.11778 | 0.049% | 3.4 |

Times are measured on the same CPU/container, including host conversion, with 8192-row decoder chunks. They are not comparable to the paper GPU figures. Checkpoint selection uses validation only, with a 100-unit penalty per invalid re-simulation; that selection score is not a headline metric.

## Validation probes

| Problem | Family | Ablation | Parameters | Best validation score | Epoch |
|---|---|---|---|---|---|
| kinematics | standard | baseline | 99830 | 0.01323 | 11 |
| kinematics | cvae | baseline | 98540 | 0.03579 | 4 |
| kinematics | invertible | baseline | 99364 | 0.01480 | 12 |
| ballistics | standard | baseline | 99830 | 0.29687 | 10 |
| ballistics | cvae | baseline | 98668 | 0.04207 | 2 |
| ballistics | invertible | baseline | 99364 | 0.14189 | 12 |

One-factor ablations are probes. A single seed and a short schedule do not establish architectural superiority. Full profiles use the existing provisional 1M/50-epoch schedule and <=3M parameters; team decisions on the definitive schedule, reference estimator/kernel and integer-speed handling remain open.

Detailed JSON accompanies this report. Each run keeps config, best/final checkpoints, training log, evaluation arrays, latent/reconstruction summaries and physical posterior plots under experiments/runs/.

![Baseline comparison](first_trial_of_encoders_gpu.png)
