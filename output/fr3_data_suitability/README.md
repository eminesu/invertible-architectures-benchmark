# FR3 data suitability evidence

Generated on 2026-11-12 by [analyze_fr3_data_suitability.py](../../experiments/analyze_fr3_data_suitability.py). Read the [assessment](../../docs/reports/2026-11-12_fr3-data-suitability.md) first.

| Artifact | Contents |
|---|---|
| [metadata.json](metadata.json) | Environment, source hashes, Git revision and reproducibility limitations |
| [checks.json](checks.json) | Per-run/split integrity, normalization, geometry, array/config/checkpoint hashes, reference subset checks, GPU fit |
| [statistics.csv](statistics.csv) | Per-dimension ranges, quantiles, mean and population standard deviation |
| [uniformity.csv](uniformity.csv) | Conditional joint uniformity tests; 40 bins, 56-test Bonferroni family |
| [timings.csv](timings.csv) | Five raw timing repeats at each pilot size |
| [projections.csv](projections.csv) | Log-log and repeated-batch projections for train/validation budgets |
| [scaling.png](figures/scaling.png) | Measured median wall times on log axes |

| Training recipe | Joint histograms | Pose histograms | Conditional sampling and conditioning |
|---|---|---|---|
| 6D FIF | [Joints](figures/fr3_6d_fif_s0_joints.png) | [Pose](figures/fr3_6d_fif_s0_poses.png) | [Diagnostics](figures/fr3_6d_fif_s0_conditioning.png) |
| 6D CFMPE | [Joints](figures/fr3_6d_cfmpe_s0_joints.png) | [Pose](figures/fr3_6d_cfmpe_s0_poses.png) | [Diagnostics](figures/fr3_6d_cfmpe_s0_conditioning.png) |
| 5D FIF | [Joints](figures/fr3_5d_fif_s0_joints.png) | [Pose](figures/fr3_5d_fif_s0_poses.png) | [Diagnostics](figures/fr3_5d_fif_s0_conditioning.png) |
| 5D CFMPE | [Joints](figures/fr3_5d_cfmpe_s0_joints.png) | [Pose](figures/fr3_5d_cfmpe_s0_poses.png) | [Diagnostics](figures/fr3_5d_cfmpe_s0_conditioning.png) |

Full training arrays were reconstructed in RAM and discarded. Only the small GPU pilot arrays are saved under `pilot/` as uncompressed NPZ containing float32 x `(N, 7)` and y `(N, 9)`. They are ignored by Git; the script regenerates them. CPU/GPU agreement is checked before saving. The original training and evaluation assets were read only.

All reported zeros in fixed-width CSV are rounded; inspect full decimal precision and pass/fail flags in `checks.json` for tiny numerical discrepancies. Log axes in scientific figures show powers of ten.
