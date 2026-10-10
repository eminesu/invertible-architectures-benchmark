# Autoencoder experiment results

Profile: **full**. Generated from completed run artifacts; 6 trained runs.

Pilot training uses 30,000 train / 2,000 validation pairs, 12 epochs and a matched <=100k parameter budget. Pilot evaluation uses the first 16 committed test conditions, 256 model / reference samples, and a 20M proposal cap. These are exploratory results, not a replication of the paper tables.

Shared headline metrics remain raw-unit IMQ MMD² (unbiased, biased also saved), Gaussian reference widths 0.01 / 0.02, and clamp 10. No rounding/dequantization is used in training. Ballistics diagnostics include separate standardized and rounded-speed sensitivity scores.

Minimum accepted reference samples per condition: {'kinematics': 444, 'ballistics': 111}. The proposal cap can leave a shorter reference; the MMD estimator handles unequal sample counts.

## Evaluation

| Problem | Family | Ablation | Err_post | Err_resim | Failed | Total sampling ms |
|---|---|---|---|---|---|---|
| kinematics | standard | baseline | 0.00528 | 0.00053 | 0.000% | 2173.9 |
| kinematics | cvae | baseline | 0.01694 | 0.00096 | 0.000% | 2171.8 |
| kinematics | invertible | baseline | 0.02121 | 0.00304 | 0.000% | 4827.1 |
| ballistics | standard | baseline | 0.00806 | 0.00257 | 0.015% | 2176.9 |
| ballistics | cvae | baseline | 0.00879 | 0.00138 | 0.067% | 2197.9 |
| ballistics | invertible | baseline | 0.00861 | 0.00388 | 0.069% | 4848.7 |

Times are measured on the same CPU/container, including host conversion, with 8192-row decoder chunks. They are not comparable to the paper GPU figures. Checkpoint selection uses validation only, with a 100-unit penalty per invalid re-simulation; that selection score is not a headline metric.

## Validation probes

| Problem | Family | Ablation | Parameters | Best validation score | Epoch |
|---|---|---|---|---|---|
| kinematics | standard | baseline | 2999078 | 0.00030 | 50 |
| kinematics | cvae | baseline | 2992008 | 0.00040 | 27 |
| kinematics | invertible | baseline | 2999992 | 0.00191 | 43 |
| ballistics | standard | baseline | 2999078 | 0.00399 | 46 |
| ballistics | cvae | baseline | 2992714 | 0.00133 | 18 |
| ballistics | invertible | baseline | 2999992 | 0.00268 | 50 |

One-factor ablations are probes. A single seed and a short schedule do not establish architectural superiority. Full profiles use the existing provisional 1M/50-epoch schedule and <=3M parameters; team decisions on the definitive schedule, reference estimator/kernel and integer-speed handling remain open.

Detailed JSON accompanies this report. Each run keeps config, best/final checkpoints, training log, evaluation arrays, latent/reconstruction summaries and physical posterior plots under experiments/runs/.

![Baseline comparison](third_trial_of_encoders_full_gpu.png)
