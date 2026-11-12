# FR3 data suitability assessment

Date: 2026-11-12. Scope: retrospective audit of the data recipes used by the existing FIF and CFMPE runs, covering both FR3 tasks. No models were trained or evaluated again. Shared classes, trained models, evaluation results, conditions and reference caches were left unchanged.

## Assessment and recommendation

**The reconstructed data is suitable for the current kinematics benchmark.** Across four run recipes, each with 1000000 training and 20000 validation rows, the integrity checks found no non-finite values, limit violations, exact duplicate joint/pose rows, or detected split leakage. Recomputed training-only normalization buffers matched all four checkpoints exactly. An independent 64-configuration JRL fixture also passed.

This establishes consistency of the current generator and saved training configuration. Original training arrays and their hashes were not saved, so it does not certify historical arrays byte for byte. The two tasks reuse joint draws within each method; the audited 4080000 rows are not all independent joint configurations.

**Recommendation:** generate our own training pairs using the shared, versioned generator and agreed seeds, and record array hashes at generation time. Retain the provided, committed evaluation conditions and agree on one common, verified GT cache per task/split. CFMPE's prior family, observation noise in other methods, and bounded-angle treatment must remain explicit comparison factors. Further scaling should wait for Paul's instructions.

## Source, generation and representation

Paul introduced the official-description kinematics in commit `93a858b` and registered the tasks in `5026d33`. The source is [Franka's reduced FR3 URDF and provenance](../../robot_ik/assets/fr3/PROVENANCE.md), using `franka_description` commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9`. The flange is `fr3_link8`; units are metres and radians. Visual, collision and inertial data were removed: this is a kinematics benchmark, with no self-collision or obstacle filtering.

The [shared task](../../robot_ik/tasks.py) samples joint values inside URDF limits, computes float64 FK on the CPU, and encodes the pose. Trainers cast the resulting arrays to float32 in RAM. There is no supplied raw training set used by these runs. This follows the sampling procedure in [IKFlow, Section IV-A](https://arxiv.org/pdf/2111.08933).

| Task | Joint target x | Pose condition y | Intrinsic constraints | Regular local posterior dimension |
|---|---|---|---|---|
| FR3 6D | 7 angles | 9 values: position, rotation columns r1 and r2 | 6 | 1 |
| FR3 5D | 7 angles | 6 values: position, flange z-axis | 5 | 2 |

**The issue's “6 joint values” should be corrected to “7 joint values.”** Six describes the full-pose task. Rotation columns obey unit-length and orthogonality constraints; nine encoded values do not mean nine independent constraints. In 5D, roll is free and joint 7 does not affect the condition.

FIF samples the nominal uniform box. The [CFMPE adapter](../../experiments/fr3_fif_cfmpe_adapters.py) first samples seven width scales c in [0.50000, 1.00000], independently replaces each row with all widths equal to 1 with probability 0.25000, then samples joints in the contracted intervals about their midpoints. Its arrays are `(x, y, c)`, with c shape `(N, 7)`. Both methods use training seed 10000 and validation seed 20000, but CFMPE consumes random numbers for c before drawing joints. Their training examples differ.

| Run | Train / validation rows | Float32 training-array size (MiB) | Parameters |
|---|---|---:|---:|
| `fr3_6d_fif_s0` | 1000000 / 20000 | 61.03516 | 2992038 |
| `fr3_6d_cfmpe_s0` | 1000000 / 20000 | 87.73804 | 2988817 |
| `fr3_5d_fif_s0` | 1000000 / 20000 | 49.59106 | 2994069 |
| `fr3_5d_cfmpe_s0` | 1000000 / 20000 | 76.29395 | 2999319 |

Sizes exclude temporary float64 arrays, FK workspaces and model memory. Small GPU-pilot examples are saved as NPZ files; full reconstructed training arrays are discarded. Shapes, dtypes, source/config/checkpoint hashes, array hashes and package versions are in [metadata](../../output/fr3_data_suitability/metadata.json) and [checks](../../output/fr3_data_suitability/checks.json).

## Correctness and distribution checks

All training and validation rows were re-simulated after float32 storage. The maximum absolute encoded-component discrepancy is shown in micro-units to avoid rounding a small nonzero error to zero. The acceptance threshold was 2.00000 micro-units per component, equivalently 0.000002 in stored units. This mixes metres and dimensionless rotation entries and is a storage-consistency check, not a physical pose metric.

| Run | Maximum train discrepancy (micro-units) | Maximum validation discrepancy (micro-units) | Failed rows |
|---|---:|---:|---:|
| 6D FIF | 0.41079 | 0.31116 | 0 |
| 6D CFMPE | 0.40847 | 0.28665 | 0 |
| 5D FIF | 0.41543 | 0.36795 | 0 |
| 5D CFMPE | 0.39723 | 0.32155 | 0 |

Re-simulation uses the same FK implementation, so an additional independent JRL fixture was checked with its documented flange-to-hand frame rotation. Maximum position-component error was 137.44859 nanometres; maximum rotation-matrix entry error was 0.35031 micro-units. Both passed.

Across all eight reconstructed splits:

- No NaNs/infinities, out-of-limit joints, conditional-box violations, or exact duplicate joint/encoded-pose rows were found. Joint-limit tolerance was 0.0000005 rad for float32 endpoint rounding.
- Exact joint-row overlap was zero between training and internal validation, and between either split and committed validation/test generating configurations. Committed validation/test sets also did not overlap. This detects exact configurations, not every possible near-duplicate or shared IK solution.
- Stored orientation columns/axes passed unit-length, orthogonality and, for 6D, reconstructed determinant checks. The maximum tested defect was 0.17330 micro-units. No position exceeded the conservative URDF reach bound of 1.31926 m.
- All 56 marginal uniformity checks passed a Bonferroni-corrected family threshold of 0.01000; minimum unadjusted p-value was 0.03878. For CFMPE, uniformity is tested after dividing joint offsets by the actual conditional half-width. Raw CFMPE joint marginals are intentionally more concentrated near interval centres. These checks support the sampling recipe; marginal tests do not prove the complete joint distribution.
- CFMPE had 250734 nominal training rows (25.07340%) and 4996 nominal validation rows (24.98000%) on each task. The random anchor mask does not enforce exactly one quarter.

The [per-dimension statistics](../../output/fr3_data_suitability/statistics.csv) contain ranges, means, standard deviations and tail quantiles for x, y and c. [Histograms](../../output/fr3_data_suitability/README.md) cover every joint and pose component. Uniform joint sampling does not imply uniform Cartesian coverage. Extreme but valid reachable poses were retained.

### Periodicity and local degeneracy

| Joint | Lower limit (rad) | Upper limit (rad) |
|---|---:|---:|
| 1 | -2.90070 | 2.90070 |
| 2 | -1.83610 | 1.83610 |
| 3 | -2.90070 | 2.90070 |
| 4 | -3.07700 | -0.11690 |
| 5 | -2.87630 | 2.87630 |
| 6 | 0.43980 | 4.62160 |
| 7 | -3.05080 | 3.05080 |

All seven joints are revolute, and adding a full turn preserves FK numerically. All allowed intervals are shorter than a full turn; none is a continuous joint. Their interval endpoints must not be identified. A torus representation for M-FFF can express angular periodicity, but it must still respect the bounded prior. Joint 6 crosses pi, so blindly returning every angle in (-pi, pi] can produce an equivalent but out-of-limit representative.

Jacobian conditioning was inspected on 10000 randomly selected training rows per run. Linear rows were scaled by a characteristic length of 1.00000 m; rotational rows use radians. Condition numbers depend on this convention.

| Run | Expected task rank | Rank-deficient rows | Median condition | P95 condition | Condition above 1000 |
|---|---:|---:|---:|---:|---:|
| 6D FIF | 6 | 0 | 19.01227 | 250.68973 | 81 |
| 6D CFMPE | 6 | 0 | 19.66965 | 194.38361 | 61 |
| 5D FIF | 5 | 0 | 16.46846 | 213.86534 | 67 |
| 5D CFMPE | 5 | 0 | 17.07727 | 161.29313 | 53 |

Rank tolerance was relative 0.0000000001. Poorly conditioned regions exist even though no numerical rank loss was observed in these samples. The regular posteriors are already lower-dimensional manifolds; singularities, boundary collapse and disconnected branches cannot be ruled out globally by this sample. There were no sampled nearest neighbours within 0.000001 in joint-limit-normalized coordinates; that near-duplicate check covers the 10000-row subsets only.

## Noise and fair comparison

**Discrete dequantization is unnecessary:** the targets are continuously sampled angles. Float32 storage alone is not evidence that jitter should be added.

IKFlow uses full-dimensional joint noise and conditions on its magnitude to address singular target distributions; the appendix gives noise scale 0.00100. This differs from perturbing the pose condition. [IKFlow, Section IV-C and Appendix A](https://arxiv.org/pdf/2111.08933).

Our assessment is to preserve the noiseless dataset. FIF already uses intrinsic latent dimensions 1 and 2 and observation sigma 0.00000. CFMPE uses `sigma_min=0.00010` in its flow path; that is not a SoftFlow replication. Whether extra smoothing improves either model requires a separate controlled training ablation. Data checks alone cannot establish that benefit or explain existing model errors.

The saved FFF, M-FFF and FMPE configs match the budget of 1000000/20000 rows, 50 epochs, batch 1000 and approximately 3000000 parameters; see [FFF/M-FFF](2026-11-11_fr3-fff-mfff.md) and [FMPE](2026-11-11_fr3-fmpe.md). The four audited models fit statistics on training data only; all saved mean/std buffers matched reconstruction with maximum difference 0.00000. FIF uses population standard deviation; CFMPE uses sample standard deviation and includes c in its context. No standard deviation was degenerate. M-FFF represents angles on circles rather than standardizing them as Euclidean coordinates.

Equal compute budgets do not make training objectives/distributions identical: CFMPE uses a prior family, while the other methods use the nominal prior. FFF/M-FFF also add pose noise 0.00100. Comparisons should disclose both differences; a strict nominal-prior-only comparison would need a separately agreed CFMPE control run. No such run was started.

Internal loss validation uses 20000 generated pairs. Model-selection evaluation uses 200 committed poses; final evaluation uses 1000 committed poses, with 1000 GT samples per pose in the existing caches. These are different validation purposes.

## Local GPU pilot and scaling estimate

Hardware: NVIDIA GeForce GTX 1650 Ti, 4096 MiB. PyTorch 2.14.1+cu126, float64 FK and float32 outputs, four CPU threads. The GPU was idle before the audit. The existing training generator remains on CPU even when training uses CUDA; a local alternate path tested CUDA FK without changing it.

The pilot used nominal 6D data at four sizes, five repeats each, after three warm-up passes. CUDA synchronization brackets timings. Generation includes joint sampling, transfers, FK, pose encoding and returning float32 poses to CPU. Save timings use uncompressed NPZ; load timings include reading x/y and transferring to CUDA. Read timings are warm filesystem-cache measurements, not cold-disk throughput.

| Rows | CPU generation (s) | GPU generation (s) | NPZ save (s) | Load + CUDA transfer (s) |
|---|---:|---:|---:|---:|
| 1000 | 0.00441 | 0.00554 | 0.00142 | 0.01175 |
| 5000 | 0.01799 | 0.00879 | 0.00202 | 0.01136 |
| 10000 | 0.04916 | 0.01682 | 0.00344 | 0.01370 |
| 20000 | 0.09331 | 0.03014 | 0.00287 | 0.01288 |

Medians are fitted with `log(time) = log(a) + alpha * log(N)`:

| Operation | Alpha | R-squared in log space | Power-law projection to 1000000 rows (s) | Repeated 20000-row batches projection (s) |
|---|---:|---:|---:|---:|
| CPU generation | 1.03159 | 0.99038 | 5.14587 | 4.66558 |
| GPU generation | 0.55400 | 0.91831 | 0.22106 | 1.50720 |
| NPZ save | 0.27548 | 0.81304 | 0.00962 | 0.14336 |
| Load + CUDA transfer | 0.04236 | 0.40302 | 0.01543 | 0.64386 |

**Use the repeated-batch estimate for provisional planning**, especially on the GPU: launch and I/O overhead dominate small sizes, and a 50-fold extrapolation of a sublinear fit is optimistic. The load fit has weak explanatory power. Neither estimate is a measured full-dataset CUDA run or a confidence interval. Including 20000 validation rows projects GPU generation to 1.53734 s by repeated batches; save/load costs are additional. Actual reconstruction of the existing 1000000-row CPU training recipes took 5.14597–5.79009 s, consistent with the CPU scale estimate.

This uses the empirical log-log fitting approach only; Kaplan et al.'s language-model loss exponents do not predict robot-data generation cost. [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361). The pilot does not estimate model-training time or GT tracing effort, nor separately benchmark 5D or CFMPE-family generation. No new GPU batch exceeded 20000 rows. Full-size CUDA generation and broader scaling await Paul's direction.

Raw repeats, alternative projections and the plot are available in [timings](../../output/fr3_data_suitability/timings.csv), [projections](../../output/fr3_data_suitability/projections.csv) and [scaling figure](../../output/fr3_data_suitability/figures/scaling.png).

## Reference-cache scope and reproduction

The four existing local GT caches were read without regeneration. All stored `samples` and `oracle` arrays were finite. For each array, 16 evenly spaced samples per pose were checked: 76800 configurations total, all passing FK and joint-limit checks. Cached diagnostics report closed loops, with 4–8 loops per pose. However, they lack a generating-configuration-on-fiber diagnostic, and these checks do not establish branch completeness or correct posterior mass weights. Local cache hashes differ from the original cache hashes documented by the team. Agree on shared verified references before interpreting small cross-machine metric differences.

Run the audit again into a new directory:

```powershell
.venv/Scripts/python.exe experiments/analyze_fr3_data_suitability.py --out output/fr3_data_suitability_review
```

This reconstructs existing training-size data on CPU, reads the four trained models, and runs only the bounded new GPU pilot. It never trains models or generates GT. The first audit's machine-readable results are in [output/fr3_data_suitability](../../output/fr3_data_suitability/README.md). Numerical JSON preserves tiny nonzero errors in decimal notation; report tables and CSV use five decimal places.

Validation: 14 tests passed, covering audit corruption controls, independent FK cross-check and existing kinematics invariants:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_fr3_data_suitability.py tests/test_robot_ik_crosscheck.py tests/test_robot_ik_kinematics.py -q
```
