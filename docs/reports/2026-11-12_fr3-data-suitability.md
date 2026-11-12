# FR3 data suitability assessment

This report checks the FR3 data used by our FMPE, CFMPE, FIF, FFF, M-FFF, IKFlow, cINN, ciLNN and iLNN experiments. The aim is to understand where the training pairs come from and whether they give us a reasonable basis for comparing the methods. Since training has already taken place, we checked the saved configurations and checkpoints and recreated the relevant data from their seeds. The original full training arrays were not saved, so this cannot establish that every historical array was identical byte for byte.

## 1. Data source and generation procedure

We generate our own training pairs using FR3 simulator. The robot description and simulator are provided in the repository; we do not load a separate supplied training dataset. The robot model comes from Franka's official description, reduced to the chain ending at the flange, `fr3_link8`. Its source version and modifications are listed in [PROVENANCE.md](../../robot_ik/assets/fr3/PROVENANCE.md).

For the nominal dataset, the procedure is straightforward:

1. Sample each of the seven joint angles uniformly within its URDF limits.
2. Run forward kinematics to obtain the flange position and orientation.
3. Encode the pose for either the 5D or 6D task and keep it paired with the sampled joints.

The [shared task](../../robot_ik/tasks.py) computes FK in float64 on CPU. The trainers then use float32 arrays: `x` has shape `(N, 7)`, while `y` has shape `(N, 9)` for 6D or `(N, 6)` for 5D. Joint values are in radians and positions are in metres. This follows the data-generation approach in [IKFlow, Section IV-A](https://arxiv.org/pdf/2111.08933). The simulator does not filter self-collisions or collisions with obstacles, so these are reachable kinematic examples rather than guaranteed collision-free configurations.

All main runs below use **1000000 training pairs and 20000 validation pairs**, with training seed 10000 and validation seed 20000 for model seed 0. The important differences are what happens around this common generation procedure:

| Method | FR3 tasks with saved main runs | Source of the clean pairs | Additional treatment during training |
|---|---|---|---|
| FMPE | 5D, 6D | Shared nominal generator | Flow-matching path; clean FK pairs are retained |
| FIF | 5D, 6D | Shared nominal generator | No observation noise in these runs |
| FFF | 5D, 6D | Shared nominal generator | Noise added to the encoded pose |
| M-FFF | 5D, 6D | Shared nominal generator | Same pose noise; angles represented on circles |
| IKFlow | 5D, 6D | Shared nominal generator | Joint-space padding and SoftFlow noise |
| cINN | 5D, 6D | Shared nominal generator | Conditional likelihood model; no extra noise added to the clean pairs |
| ciLNN | 5D, 6D | Shared nominal generator | Conditional liquid-flow model; no extra noise added to the clean pairs |
| iLNN base | 5D | Shared nominal generator | Unconditional likelihood model |
| iLNN pad3 | 5D, 6D | Shared nominal generator | Three padding channels, with noise on those channels |
| CFMPE | 5D, 6D | Width-conditioned joint-prior family | Seven prior-width values are included in the condition |

All trainers call [`make_dataset`](../../data/toy_data.py). No draws were rejected in the audited FR3 runs. Replaying the shared generator reproduced the same nominal training and validation arrays as the earlier full FIF data check, on both tasks. This is why those clean-data checks also apply to the other methods using the same recipe, sizes and seeds.

CFMPE is the exception. Its [adapter](../../experiments/fr3_fif_cfmpe_adapters.py) draws seven interval-width scales between 0.50000 and 1.00000 and samples joints inside the contracted intervals. Each row has probability 0.25000 of using all widths equal to 1.00000. It therefore receives `(x, y, c)` triples, with `c` shaped `(N, 7)`. The audited training set contains 250734 nominal-prior rows, or 25.07340%; it does not give CFMPE the same 1000000 nominal-prior examples as the other methods. Consuming random numbers for the widths also changes the subsequent joint draws, even with the same seed.

The provided data we reuse are the committed **200 model-selection poses and 1000 test poses per task**, under `metrics/conditions/`. Posterior reference samples are held in local GT caches. These are separate from the 20000 pairs used to monitor validation loss.

## 2. Joint values and pose representation

The FR3 arm has **seven joint values**. The name "6D" refers to the six constraints on the end-effector pose, rather than the number of joints.

| Task | Physical joint vector | Encoded pose | Independent pose constraints | Usual local solution dimension |
|---|---|---|---|---|
| FR3 6D | 7 angles | Position and the first two rotation-matrix columns: 9 values | 6 | 1 |
| FR3 5D | 7 angles | Position and the flange z-axis: 6 values | 5 | 2 |

The rotation columns and axis are constrained vectors, so the encoded size is larger than the number of independent constraints. In 5D the flange can rotate freely about its axis, and joint 7 does not change the condition.

Padding and embeddings change the tensors inside a model, not the robot data. For example, IKFlow works with a padded joint vector, iLNN pad3 adds three channels, and M-FFF embeds seven angles as cosine/sine pairs. The underlying FK target still contains seven physical joint angles.

## 3. FK consistency and joint limits

The full data checks covered the nominal and CFMPE-family recipes on both tasks. We then extended the review to the other saved methods by checking the common-generator replay and their stored normalization. The results below describe the reconstructed clean pairs, before training-time noise or padding.

| Data recipe | Maximum training discrepancy | Maximum validation discrepancy | Rows failing FK tolerance |
|---|---:|---:|---:|
| Nominal 6D | 0.41079 | 0.31116 | 0 |
| CFMPE-family 6D | 0.40847 | 0.28665 | 0 |
| Nominal 5D | 0.41543 | 0.36795 | 0 |
| CFMPE-family 5D | 0.39723 | 0.32155 | 0 |

Discrepancies are maximum absolute encoded-component differences in **micro-units**, with a tolerance of 2.00000 micro-units. This keeps float32 rounding errors visible in the table. It is a consistency check on the stored representation, not a position or rotation accuracy metric for a trained model.

No reconstructed joint vector exceeded its limits, allowing 0.50000 microradians for endpoint rounding. Orientation vectors passed the unit-length and orthogonality checks, and the 6D rotation reconstruction had the expected determinant. All positions stayed inside a conservative reach bound derived from the URDF.

Reusing the same FK code alone would not catch every implementation error. The independent JRL fixture provides an additional check: all 64 configurations passed after accounting for the flange-to-hand frame rotation. Its maximum position-component difference was 137.44859 nanometres.

The original detailed results remain in [checks.json](../../output/fr3_data_suitability/checks.json). The broader review, including checkpoint and source hashes, is in [method_inventory.json](../../output/fr3_data_suitability/method_inventory.json).

## 4. Distribution, periodicity and difficult configurations

The checked clean datasets contain no NaNs, infinities or exact duplicate joint/pose rows. No exact joint configurations overlapped between training and internal validation, or with the generating configurations of the committed validation/test poses. These checks do not exclude every possible near-duplicate or two different joint configurations reaching the same pose.

The [statistics](../../output/fr3_data_suitability/statistics.csv) and [histograms](../../output/fr3_data_suitability/README.md) show the ranges and marginal distributions of joints and pose components. The nominal plots were originally produced from FIF's data; the matching shared-generator replay now connects those plots to the other fixed-prior methods as well. Uniform joint sampling does not produce a uniform workspace distribution. CFMPE's raw joint marginals are more concentrated around interval centres, as expected from its width family.

All 56 marginal uniformity checks passed the corrected test threshold. For CFMPE, this was checked after dividing joint offsets by their conditional half-widths. That supports the sampling recipe, although marginal tests cannot prove the whole joint distribution is correct.

All seven joints are revolute and their FK is periodic over a full turn. However, every allowed interval is shorter than a full turn. The two ends of a joint's allowed interval should therefore not be treated as the same point of the bounded prior. This matters for M-FFF: a circle representation does not remove joint limits. Joint 6 also crosses pi, so returning every angle in the usual (-pi, pi] window can give an equivalent angle outside the permitted interval.

We also checked the task Jacobian on 10000 training configurations per recipe. Position rows were scaled by a characteristic length of 1.00000 m before calculating condition numbers.

| Recipe | Expected rank | Rank-deficient samples | Median condition number | P95 condition number | Condition number above 1000 |
|---|---:|---:|---:|---:|---:|
| Nominal 6D | 6 | 0 | 19.01227 | 250.68973 | 81 |
| CFMPE-family 6D | 6 | 0 | 19.66965 | 194.38361 | 61 |
| Nominal 5D | 5 | 0 | 16.46846 | 213.86534 | 67 |
| CFMPE-family 5D | 5 | 0 | 17.07727 | 161.29313 | 53 |

Most sampled configurations were regular, but some were poorly conditioned. We would keep these valid examples in the benchmark and report the difficult regions rather than quietly filter them out. The posteriors are already lower-dimensional solution sets: normally one-dimensional for 6D and two-dimensional for 5D. The sample does not rule out singularities, boundary collapse or disconnected solution branches elsewhere. Near-duplicate checks on the same 10000-row subsets also found no neighbours within 0.000001 in joint-limit-normalized coordinates.

## 5. Noise and dequantization

The sampled angles are continuous, so we do not see a reason to add discrete dequantization to the raw FR3 dataset. The ballistics dequantization option is unrelated to this task. Training noise is a separate question, especially because an exact IK posterior occupies a lower-dimensional set in joint space.

The methods handle this differently:

- **IKFlow** adds full-dimensional noise to the padded joint vector and conditions on its magnitude. The main runs use SoftFlow scale 0.00100, with noise level zero when sampling. This follows the approach in [IKFlow, Section IV-C](https://arxiv.org/pdf/2111.08933).
- **FFF and M-FFF** add pose noise of 0.00100 to the encoded observation. The 6D FFF control also uses 0.01000. This can move the condition away from an exact valid rotation encoding, even though the underlying clean FK pair is valid.
- **FMPE and CFMPE** use `sigma_min = 0.00010` in their flow path. This is different from the SoftFlow procedure.
- **FIF** uses observation sigma 0.00000 in the main FR3 runs and intrinsic latent dimensions 1 and 2.
- **cINN and ciLNN** use the clean pairs with conditional likelihood training. **iLNN pad3** adds noise of 0.02000 to its padding channels. For unconditional iLNN, `sigma_y = 0.05000` sets a likelihood width; it does not mean that the saved FK poses were perturbed.

For the iLNN variants we checked the saved configs and [Umut's corresponding training code](https://github.com/eminesu/invertible-architectures-benchmark/blob/d81bc95b4453af794ea60c3ed5961c5dc66670c7/experiments/train_ilnn.py), since the local trainer version does not expose all the recorded conditional/padding options.

Our recommendation is to retain the clean joint-pose pairs and record each method's noise treatment separately. The existing FFF noise control is useful, but it is not a common SoftFlow ablation across all methods. The data checks alone cannot tell us whether adding more noise would improve a particular model.

## 6. Comparison budget and standardization

The main runs share the 1000000/20000 pair budget and 50 requested epochs. The parameter counts are close to the intended 3000000 budget, with two exceptions worth keeping visible:

| Method | 5D parameters | 6D parameters | Batch size |
|---|---:|---:|---:|
| FMPE | 2995847 | 2997335 | 1000 |
| CFMPE | 2999319 | 2988817 | 1000 |
| FIF | 2994069 | 2992038 | 1000 |
| FFF | 2996942 | 2999054 | 1000 |
| M-FFF | 2989846 | 2991952 | 1000 |
| IKFlow | 2995560 | 2989116 | 128 |
| cINN | 2993256 | 2988112 | 1000 |
| ciLNN | 3024112 | 3064432 | 1000 |
| iLNN base | 2943472 | No base run | 1000 |
| iLNN pad3 | 2983840 | 2983840 | 1000 |

ciLNN is slightly above the nominal parameter budget. IKFlow's smaller batch also gives about 7.8 times as many optimizer steps per epoch, so equal epoch counts do not imply equal optimization effort. The completed FMPE controls at smaller parameter budgets and alpha 0, and the FFF noise control, use the same clean nominal data. Smoke tests and unfinished IKFlow search/long runs are recorded in the inventory but are excluded from this main-budget comparison.

Normalization is fitted on training data. Across **19 main runs and 7 completed controls**, all 26 checkpoint normalization checks passed, with maximum absolute buffer difference 0.00000. The extension checked 24 checkpoints against a fresh nominal-generator replay and reused the existing full family-data audit for the two CFMPE checkpoints.

FIF uses population standard deviation; the other Euclidean models use sample standard deviation. M-FFF stores angular scales and uses circle coordinates instead of centring the angles as Euclidean inputs. CFMPE also standardizes the prior-width context. None of these differences requires a different physical FK dataset.

The comparison therefore has a shared data source, but it is not a comparison with every training choice held fixed. The CFMPE prior family, different noise treatments, IKFlow batch size and ciLNN parameter counts should be stated alongside model results.

## 7. Local GPU pilot and scaling estimate

The earlier pilot used our GTX 1650 Ti with 4096 MiB memory. It tested the common nominal 6D generation/loading path at four sizes, with five repeats after warm-up. CUDA was synchronized around the measurements. FK used float64 and the returned arrays were float32.

| Rows | CPU generation (s) | GPU generation (s) | NPZ save (s) | Load and CUDA transfer (s) |
|---|---:|---:|---:|---:|
| 1000 | 0.00441 | 0.00554 | 0.00142 | 0.01175 |
| 5000 | 0.01799 | 0.00879 | 0.00202 | 0.01136 |
| 10000 | 0.04916 | 0.01682 | 0.00344 | 0.01370 |
| 20000 | 0.09331 | 0.03014 | 0.00287 | 0.01288 |

Generation includes joint sampling, FK, pose encoding and transfers needed to return CPU arrays. Loading was measured with a warm filesystem cache. The existing training generators still perform FK on CPU; selecting CUDA for a model does not automatically move data generation to the GPU.

Fitting `log(time) = log(a) + alpha * log(N)` gives CPU exponent 1.03159 and GPU exponent 0.55400. The fitted GPU time at 1000000 rows is 0.22106 s, but we would not use that as a firm estimate: fixed launch overhead strongly affects the small-size measurements. Repeating the measured 20000-row batches instead gives **1.50720 s** for 1000000 rows, or **1.53734 s** including 20000 validation rows, before save/load costs. The earlier full CPU reconstructions took 5.14597-5.79009 s per training dataset.

This is a rough generation-cost estimate using log-log fitting, inspired by the scaling approach in [Kaplan et al.](https://arxiv.org/pdf/2001.08361). It does not transfer their language-model loss exponents to robotics. It also does not estimate training time, GT tracing, or the separate cost of each method's stochastic augmentation. The raw [timings](../../output/fr3_data_suitability/timings.csv), [projections](../../output/fr3_data_suitability/projections.csv) and [scaling plot](../../output/fr3_data_suitability/figures/scaling.png) are available for inspection.

No new GPU batch exceeded 20000 rows. Larger GPU generation and scaling experiments should wait for new instructions; this revision only extended the CPU replay and saved-checkpoint review.

## 8. Suitability assessment and recommendation

Based on these checks, we consider the generated FR3 pairs suitable for our current kinematics benchmark. They reproduce the paired poses within storage precision, respect joint limits and show no obvious data corruption. The common-generator replay and normalization checks support using this assessment across the method families, rather than only FIF and CFMPE.

We would **generate our own training data with the shared, versioned generator**, while reusing the **provided validation/test pose sets**. The nominal dataset should use agreed seeds, and each run should record its data hashes, prior and noise settings. CFMPE's width-family experiment should remain clearly identified when it is compared with fixed-prior training.

For posterior evaluation, we should also agree on shared GT cache files. The existing cache subset checks passed FK and limit checks, but they do not prove that every solution branch was found or correctly weighted. That remains a reference-quality limitation, separate from the correctness of the joint-pose training pairs.

The evidence is collected under [output/fr3_data_suitability](../../output/fr3_data_suitability/README.md). The original numerical audit and its 14 passing tests are retained, and the [method inventory](../../output/fr3_data_suitability/method_inventory.json) adds the broader data-source and normalization review. No models were retrained, no model evaluations were rerun, and no GT caches were regenerated for this revision.
