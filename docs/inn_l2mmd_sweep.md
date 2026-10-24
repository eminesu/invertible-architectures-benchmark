# INN (L2 + MMD): why the baseline failed and what the sweep showed

Scope: the kinematics INN (L2 + MMD) baseline `kin_inn_l2mmd_s0` did worse than the paper (Err_post 0.0200 vs 0.017,
Err_resim 599,845 vs 0.086). Eight variants were trained on both benchmarks (`sweep_inn.sh l2mmd_wide`), with one seed
each. Files: `experiments/sweep_inn.sh` (modes `l2mmd_wide`, `l2mmd_best`), `experiments/diagnose_inn_latent.py`,
`experiments/plot_inn_sweep.py` (writes `experiments/figures/inn_l2mmd_sweep.png`), `models/inn.py`.

## 1. Key findings

| # | Finding | Evidence | Status |
|---|---|---|---|
| 1 | The baseline's failure is a small number of extreme outliers produced by the inverse pass, not a badly matched latent. | The latent on real data is N(0, I)-like (std 0.999 / 0.991, mean ‖z‖² 1.98 of 2). 1.95% of z draws give a re-simulation error above 1, up to 8.2e9; the median over the 1000 test conditions is 0.0119. | measured |
| 2 | Lowering the clamp or the MMD weight removes the blow-up: kinematics Err_resim falls from 599,845 to between 0.0071 and 0.1615, six to eight orders of magnitude. | all non-marginal variants in section 6 | measured |
| 3 | No kinematics variant beats the paper on both metrics. Closest: `clamp0.5` (0.0175 / 0.0346), `clamp1.0_wz10` (0.0177 / 0.0237), `wz10` (0.0180 / 0.0188). They pass on re-simulation and are 3-6% above the paper's 0.017 on Err_post. `lr5e-4` beats Err_post (0.0153) but not Err_resim (0.1615). | section 6 | measured |
| 4 | Under the biased MMD² estimator no kinematics variant passes (best 0.0200 vs 0.017). | `(biased)` column | measured; the paper's estimator is unknown |
| 5 | The marginal-MMD variant fails on both benchmarks: Err_post 0.4566 (kinematics) and 0.1426 (ballistics), roughly 16-30x worse than the other variants. | section 6 | measured; explanation inferred |
| 6 | `w_z` 1 trades accuracy of the posterior for re-simulation: Err_resim 0.0105 but Err_post 0.0270 (the worst of the working variants). | section 6 | measured |
| 7 | Ballistics is robust: all non-marginal variants pass on both metrics. Lowering `w_z` (alone or with a clamp) improves its re-simulation 3.7-9.7x; the clamp alone makes it 1.6-2.6x worse. | section 6 | measured |
| 8 | Changing only the learning rate (`lr5e-4`) also moved kinematics Err_resim from 599,845 to 0.16, so the baseline's blow-up depends on the training trajectory and one seed cannot separate a fix from luck. | `lr5e-4` row | measured; implication inferred |
| 9 | The biased minus unbiased Err_post gap is 0.0046-0.0057 in all 22 runs, including the failed ones, so it is an estimator artifact at n = 1000 and not a model property. Under the biased estimator no kinematics INN (L2 + MMD) run passes the paper. | section 4.4, 4.5 | measured; the paper's estimator is unknown |

Status meaning: **measured** = read directly off the logs or result tables; **inferred** = my explanation, consistent with
the evidence but not tested directly.

## 2. Setup

| Item | Value |
|---|---|
| Fixed for every run | 8 coupling blocks, 3 hidden layers of width 303, ReLU, about 3M parameters, Adam, 50 epochs, batch 1000, 1M training samples, weight decay 1e-5, cosine schedule, gradient clip 10, `w_y` 1, standardised units |
| Baseline INN (L2 + MMD) | clamp 2.0, `w_z` 100, learning rate 1e-3, joint MMD on `[y_hat, z]` vs `[y, eps]` |
| Varied | one or two of: `--clamp`, `--w-z`, `--latent-mmd`, `--lr` (section 5) |
| Seeds | one (0) per run |
| Hardware | training on an RTX 3060 (`cuda`), scoring on CPU |
| Selection caveat | the variants are compared on the same 1000 test conditions that produce the final numbers, so the best variant's numbers are optimistic. A setting should be confirmed on other seeds (section 9). |

## 3. What the settings in the run names mean

The baseline has no suffix. Each suffix is one command-line option of `experiments/train_inn.py`.

| In the run name | Option | What it is | Values | Effect on performance (measured) | Reason for the effect | Status |
|---|---|---|---|---|---|---|
| `wz` | `--w-z` | Weight of the MMD term. The loss is `w_y * L2(y) + w_z * MMD` with `w_y` = 1. Adam ignores the overall loss scale, so only the ratio `w_z / w_y` matters: a smaller `w_z` is the same as a stronger y loss. | 100 (baseline), 10, 1 | Kinematics post / resim: 0.0200 / 599,845 (100), 0.0180 / 0.0188 (10), 0.0270 / 0.0105 (1). Ballistics: 0.0087 / 0.0041, 0.0085 / 7.95e-04, 0.0085 / 4.23e-04. | A trade-off between two goals. The y loss makes the forward map accurate, which makes the re-simulation error small. The MMD loss makes z match N(0, I) and stay independent of y, which makes the sampled posterior the right shape. Too much MMD weight (100) leaves y fitted poorly (validation y error 0.0018 vs 0.00025 for the ML INN) and the inverse can blow up. Too little (1) lets the latent drift, so Err_post rises to 0.0270. About 10 is the balance in these runs. | inferred, consistent with the numbers |
| `marg` | `--latent-mmd marginal` | Which sets the MMD compares. Joint (default): `[y_hat, z]` against `[y, eps]`, where `eps ~ N(0, I)` is independent of y. Marginal: only `z` against `eps`. | joint (baseline), marginal | Marginal MMD: kinematics 0.4566 / 160,416, ballistics 0.1426 / 8.71. 16-30x worse Err_post than any other variant. | The joint MMD forces z to look like N(0, I) and to be independent of y. The marginal MMD only forces z to look like N(0, I) overall, so z may still carry information about y. At test time z is drawn independently of y*, so if z depends on y the samples come from the wrong distribution. Example: if z were a function of y whose values over the training set happened to look like N(0, I), the marginal MMD would be zero but a fresh z paired with y* would be meaningless. | inferred, not yet tested (section 1b of the diagnostic measures the z-y correlation) |
| `clamp` | `--clamp` | Bound on each coupling block's log-scale, `s = clamp * 0.636 * atan(s_raw)`, so each scale factor lies in about (e^-clamp, e^clamp). | 2.0 (baseline), 1.0, 0.5 | Kinematics resim 599,845 -> 0.1582 (1.0) -> 0.0346 (0.5). Ballistics resim 0.0041 -> 0.0065 -> 0.0106 (worse). | A lower clamp stops the inverse from stretching x exponentially. It helps where the inverse blew up (kinematics) and only restricts the model where it did not (ballistics). | inferred |
| `lr` | `--lr` | Learning rate. | 1e-3 (baseline), 5e-4 | Kinematics 0.0153 / 0.1615, ballistics 0.0085 / 0.0030. | Gentler optimisation matches the latent better, but nothing bounds the inverse, so some outliers remain. | inferred |
| (not a network option) | none | "biased" and "unbiased" name the two ways of estimating the MMD that scores Err_post (section 4). No network is biased or unbiased. | | The `(biased)` column is 0.0046-0.0057 higher than `post` in every run. | See section 4. | measured |

## 4. The MMD and its two estimators

### 4.1 Definition used in this repository

| Item | Value |
|---|---|
| Quantity | MMD^2(P, Q) = E k(X, X') + E k(Y, Y') - 2 E k(X, Y), with X, X' independent draws from P and Y, Y' independent draws from Q |
| Kernel (`metrics/mmd.py`) | k(a, b) = sum over h in (0.05, 0.2, 0.9) of 1 / (1 + abs(a - b)^2 / h^2), so k(a, a) = 3. The paper does not give a kernel. |
| Err_post (evaluation) | MMD^2 between the 1000 model samples and the ground-truth posterior samples of one test condition, in raw x units; the table value is the mean over the 1000 conditions |
| MMD in training (INN L2 + MMD) | the same kernel on a batch of 1000, `[y_hat, z]` against `[y, eps]`, in standardised units, unbiased estimator (`mmd_torch`) |

### 4.2 The two estimators

```
unbiased (U-statistic, the repo's default and headline number)
  U = 1/(n(n-1)) * sum_{i != j} k(x_i, x_j)  +  1/(m(m-1)) * sum_{i != j} k(y_i, y_j)  -  2/(nm) * sum_{i,j} k(x_i, y_j)

biased (V-statistic, the "(biased)" column)
  V = 1/n^2 * sum_{i,j} k(x_i, x_j)  +  1/m^2 * sum_{i,j} k(y_i, y_j)  -  2/(nm) * sum_{i,j} k(x_i, y_j)

exact relation between them
  V - U = (3 - Ux)/n + (3 - Uy)/m,    Ux = mean of k(x_i, x_j) over i != j,  Uy likewise
```

They differ only in whether the i = j terms are kept. In code: the unbiased branch computes
`(k_aa.sum() - np.trace(k_aa)) / (n * (n - 1))`, the biased branch `k_aa.mean()`.

### 4.3 Why the unbiased estimator is unbiased, and how this was checked

Argument. For i != j, x_i and x_j are independent draws from P, so E[k(x_i, x_j)] = E k(X, X'). Averaging only those
terms therefore has expectation E k(X, X'), and the same holds for the y and cross terms, so E[U] = MMD^2(P, Q). The
biased version also contains the i = j terms, where k(x_i, x_i) = 3 is a constant and not an estimate of E k(X, X'),
so E[V] = MMD^2 + (3 - E k(X, X'))/n + (3 - E k(Y, Y'))/m. This is always above MMD^2 (k never exceeds 3), by an amount
that depends on n, m and the kernel but not on how well the model fits.

I did not rely on the argument or on the repo's docstring alone. Checks run with the repo's own `mmd` function:

| Check | What was done | Result |
|---|---|---|
| Algebra | On random 4-D samples (n = m = 1000), compare V - U with `(3 - Ux)/n + (3 - Uy)/m` | equal to 1.2e-16 (0.004728462 both ways) |
| Centred on the truth | P = N(0, 0.3^2 I), Q = N(0.15, 0.35^2 I) in 4-D. Reference MMD^2 from n = m = 10,000 over 6 replicates: 0.06675 +- 0.00074. Then 400 replicates at n = m = 1000. | unbiased mean 0.06653 +- 0.00031 (-0.00023 from the reference, within noise); biased mean 0.07153 +- 0.00031 (+0.00478, as the formula predicts) |
| Identical distributions (true value 0) | 400 replicates at n = m = 1000 | unbiased mean -0.00004 +- 0.00003, with 60% of the individual estimates negative; biased mean +0.00468 +- 0.00003 |
| Dependence on sample size | identical N(0, I) samples, from an earlier check | biased 0.0283, 0.0061, 0.0012 at n = 200, 1000, 5000; unbiased -0.0003, +0.0004, +0.0001 (about 0 at every n) |

What "unbiased" does and does not mean here:

| Point | Explanation |
|---|---|
| Unbiased in expectation, not exact each time | A single estimate scatters (standard deviation about 0.006 at n = 1000 for the pair above) and can be negative, as the 60% above shows. Only the average over many fresh samples equals MMD^2. |
| Needs independent samples | The model's 1000 draws per condition are independent. The ground-truth samples come from rejection sampling with a tolerance (sigma 0.01 kinematics, 0.02 ballistics), so the estimate targets the distance to that sampler's distribution, which approximates the true posterior. |
| It is MMD^2, not MMD | Taking a square root would reintroduce a bias. |
| Averaging over the 1000 conditions keeps it unbiased | The mean of unbiased estimates is unbiased. |

### 4.4 The MMD values of these runs

Err_post is this MMD^2 (section 6 has the full results). The biased minus unbiased gap is 0.0046 to 0.0057
(mean 0.0051) in all 22 runs, from the best run to the marginal-MMD failures. A real model effect would
not look like that. It is the constant sampling term of the formula above, and it stays below its upper bound
3(1/n + 1/m) = 0.006 at n = m = 1000.

| run | Err_post, unbiased | Err_post, biased | biased - unbiased |
|---|---|---|---|
| kin_inn_ml_s0 | 0.0037 | 0.0083 | 0.0046 |
| kin_inn_l2mmd_s0 | 0.0200 | 0.0247 | 0.0047 |
| kin_cinn_s0 | 0.0035 | 0.0081 | 0.0046 |
| kin_inn_l2mmd_clamp0.5_s0 | 0.0175 | 0.0222 | 0.0047 |
| kin_inn_l2mmd_clamp0.5_wz10_s0 | 0.0197 | 0.0244 | 0.0047 |
| kin_inn_l2mmd_clamp1.0_s0 | 0.0193 | 0.0239 | 0.0046 |
| kin_inn_l2mmd_clamp1.0_wz10_s0 | 0.0177 | 0.0223 | 0.0046 |
| kin_inn_l2mmd_wz10_s0 | 0.0180 | 0.0227 | 0.0047 |
| kin_inn_l2mmd_wz1_s0 | 0.0270 | 0.0317 | 0.0047 |
| kin_inn_l2mmd_marg_s0 | 0.4566 | 0.4616 | 0.0050 |
| kin_inn_l2mmd_lr5e-4_s0 | 0.0153 | 0.0200 | 0.0047 |
| bal_inn_ml_s0 | 0.0193 | 0.0249 | 0.0056 |
| bal_inn_l2mmd_s0 | 0.0087 | 0.0143 | 0.0056 |
| bal_cinn_s0 | 0.0101 | 0.0157 | 0.0056 |
| bal_inn_l2mmd_clamp0.5_s0 | 0.0081 | 0.0137 | 0.0056 |
| bal_inn_l2mmd_clamp0.5_wz10_s0 | 0.0085 | 0.0141 | 0.0056 |
| bal_inn_l2mmd_clamp1.0_s0 | 0.0081 | 0.0137 | 0.0056 |
| bal_inn_l2mmd_clamp1.0_wz10_s0 | 0.0085 | 0.0141 | 0.0056 |
| bal_inn_l2mmd_wz10_s0 | 0.0085 | 0.0140 | 0.0055 |
| bal_inn_l2mmd_wz1_s0 | 0.0085 | 0.0141 | 0.0056 |
| bal_inn_l2mmd_marg_s0 | 0.1426 | 0.1483 | 0.0057 |
| bal_inn_l2mmd_lr5e-4_s0 | 0.0085 | 0.0141 | 0.0056 |

MMD used inside training (validation batches, joint, unbiased, the INN (L2 + MMD) baseline): kinematics -3.3e-04,
ballistics -3.5e-05. Both are near zero, matching the diagnostic that the latent matches N(0, I). They are not comparable
with Err_post, which uses different data and units.

### 4.5 Which estimator the paper used, and the effect on the verdict

The paper does not say. Its Eq. (2) averages over all index pairs, which read as a sample formula is the biased form, but
that is the training loss and nothing says the evaluation used it. The repo reports the unbiased value as the headline
and the biased one next to it; its docstring notes the biased value is not used for reporting because its offset
(about 0.006) is the same size as the paper's best numbers.

| Estimator | Kinematics INN (L2 + MMD) vs the paper's 0.017 | Other rows |
|---|---|---|
| unbiased | `lr5e-4` passes (0.0153); the others are above it, from 3% (`clamp0.5`) to 59% (`wz1`), and the marginal run is far above | all pass except the marginal-MMD runs |
| biased | none pass: the best is `lr5e-4` at 0.0200 (baseline 0.0247, `clamp0.5` 0.0222, `clamp1.0_wz10` 0.0223, `wz10` 0.0227) | all pass except the marginal-MMD runs |

So if the paper's numbers are biased estimates, the kinematics INN (L2 + MMD) row is not matched by any setting tried.
This cannot be decided from the paper.

## 5. Every change we made, and why

| Change | What it does | Why we tried it | Result |
|---|---|---|---|
| `--clamp 0.5`, `1.0` (baseline 2.0) | Each coupling scale factor is `exp(clamp * 0.636 * atan(s))`, so it lies in about (0.61, 1.65) for 0.5, (0.37, 2.7) for 1.0 and (0.14, 7.4) for 2.0. A lower clamp limits how strongly any one block can stretch or squeeze. | the diagnostic showed the inverse pass stretching x to 3.6e5 standardised units | removes the blow-up on kinematics (0.0346 and 0.1582); slightly hurts ballistics, where there was no blow-up |
| `--w-z 10`, `1` (baseline 100) | The loss is `w_y * L2(y) + w_z * MMD`. Adam ignores the overall loss scale, so only the ratio `w_z / w_y` matters. Lower `w_z` means a stronger y loss. | the baseline fitted y about 7x worse than the ML INN (validation y error 0.0018 vs 0.00025) | `w_z` 10: kinematics resim 0.0188, post 0.0180; `w_z` 1 improves resim further but post gets worse |
| `--clamp` and `--w-z 10` together | both effects | to see whether they add up | clamp 0.5: resim 0.0071; clamp 1.0: 0.0237, which is worse than `w_z` 10 alone (0.0188) |
| `--latent-mmd marginal` | MMD of z vs eps only, as Eq. (3) is written, instead of the joint `[y_hat, z]` vs `[y, eps]` | the paper writes the MMD on z | fails on both benchmarks |
| `--lr 5e-4` (baseline 1e-3) | gentler optimisation | the baseline's training curve was very noisy (provisional re-simulation error between about 10 and 1e6 from epoch 5 on) | best kinematics post (0.0153), resim 0.1615 |

Results relative to the baseline (post change in percent, resim as a factor):

| Variant | Kinematics post (vs baseline) | Kinematics resim (vs baseline) | Ballistics post (vs baseline) | Ballistics resim (vs baseline) |
|---|---|---|---|---|
| baseline (clamp 2.0, w_z 100, lr 1e-3, joint MMD) | 0.0200 | 599,845.02 | 0.0087 | 0.0041 |
| clamp0.5 | 0.0175 (-12%) | 0.0346 (1.73e+07x lower) | 0.0081 (-7%) | 0.0106 (2.59x higher) |
| clamp0.5_wz10 | 0.0197 (-2%) | 0.0071 (8.45e+07x lower) | 0.0085 (-2%) | 0.0011 (3.73x lower) |
| clamp1.0 | 0.0193 (-3%) | 0.1582 (3.79e+06x lower) | 0.0081 (-7%) | 0.0065 (1.59x higher) |
| clamp1.0_wz10 | 0.0177 (-12%) | 0.0237 (2.53e+07x lower) | 0.0085 (-2%) | 9.58e-04 (4.28x lower) |
| wz10 | 0.0180 (-10%) | 0.0188 (3.19e+07x lower) | 0.0085 (-2%) | 7.95e-04 (5.16x lower) |
| wz1 | 0.0270 (+35%) | 0.0105 (5.71e+07x lower) | 0.0085 (-2%) | 4.23e-04 (9.69x lower) |
| marg | 0.4566 (+2183%) | 160,415.95 (still extreme) | 0.1426 (+1539%) | 8.7105 (2.12e+03x higher) |
| lr5e-4 | 0.0153 (-24%) | 0.1615 (3.71e+06x lower) | 0.0085 (-2%) | 0.0030 (1.37x lower) |

The sweep is `bash experiments/sweep_inn.sh l2mmd_wide`; 8 variants x 2 benchmarks = 16 runs, about 7-8 hours on the RTX 3060.

## 6. Full results (console format)

`resim` is the plain mean for kinematics and the mean clamped at 10 for ballistics, as printed by `sweep_inn.sh`.

### Kinematics

| run | post | (biased) | paper | resim | paper | ms/y* | vs paper (lower is better) |
|---|---|---|---|---|---|---|---|
| kin_inn_ml_s0 | 0.0037 | 0.0083 | 0.025 | 2.30e-04 | 0.015 | 283.8 | post better, resim better |
| kin_inn_l2mmd_s0 | 0.0200 | 0.0247 | 0.017 | 599,845.02 | 0.086 | 208.0 | post worse, resim worse |
| kin_cinn_s0 | 0.0035 | 0.0081 | 0.015 | 1.10e-05 | 0.008 | 320.3 | post better, resim better |
| kin_inn_l2mmd_clamp0.5_s0 | 0.0175 | 0.0222 | 0.017 | 0.0346 | 0.086 | 218.8 | post worse, resim better |
| kin_inn_l2mmd_clamp0.5_wz10_s0 | 0.0197 | 0.0244 | 0.017 | 0.0071 | 0.086 | 199.4 | post worse, resim better |
| kin_inn_l2mmd_clamp1.0_s0 | 0.0193 | 0.0239 | 0.017 | 0.1582 | 0.086 | 206.0 | post worse, resim worse |
| kin_inn_l2mmd_clamp1.0_wz10_s0 | 0.0177 | 0.0223 | 0.017 | 0.0237 | 0.086 | 184.8 | post worse, resim better |
| kin_inn_l2mmd_wz10_s0 | 0.0180 | 0.0227 | 0.017 | 0.0188 | 0.086 | 184.3 | post worse, resim better |
| kin_inn_l2mmd_wz1_s0 | 0.0270 | 0.0317 | 0.017 | 0.0105 | 0.086 | 216.7 | post worse, resim better |
| kin_inn_l2mmd_marg_s0 | 0.4566 | 0.4616 | 0.017 | 160,415.95 | 0.086 | 149.8 | post worse, resim worse |
| kin_inn_l2mmd_lr5e-4_s0 | 0.0153 | 0.0200 | 0.017 | 0.1615 | 0.086 | 281.3 | post better, resim worse |

### Ballistics

| run | post | (biased) | paper | resim | paper | ms/y* | vs paper (lower is better) |
|---|---|---|---|---|---|---|---|
| bal_inn_ml_s0 | 0.0193 | 0.0249 | 0.047 | 0.0183 | 0.019 | 291.7 | post better, resim better |
| bal_inn_l2mmd_s0 | 0.0087 | 0.0143 | 0.06 | 0.0041 | 3.668 | 173.7 | post better, resim better |
| bal_cinn_s0 | 0.0101 | 0.0157 | 0.047 | 0.1019 | 0.437 | 267.7 | post better, resim better |
| bal_inn_l2mmd_clamp0.5_s0 | 0.0081 | 0.0137 | 0.06 | 0.0106 | 3.668 | 169.1 | post better, resim better |
| bal_inn_l2mmd_clamp0.5_wz10_s0 | 0.0085 | 0.0141 | 0.06 | 0.0011 | 3.668 | 142.7 | post better, resim better |
| bal_inn_l2mmd_clamp1.0_s0 | 0.0081 | 0.0137 | 0.06 | 0.0065 | 3.668 | 176.4 | post better, resim better |
| bal_inn_l2mmd_clamp1.0_wz10_s0 | 0.0085 | 0.0141 | 0.06 | 9.58e-04 | 3.668 | 136.6 | post better, resim better |
| bal_inn_l2mmd_wz10_s0 | 0.0085 | 0.0140 | 0.06 | 7.95e-04 | 3.668 | 142.5 | post better, resim better |
| bal_inn_l2mmd_wz1_s0 | 0.0085 | 0.0141 | 0.06 | 4.23e-04 | 3.668 | 154.6 | post better, resim better |
| bal_inn_l2mmd_marg_s0 | 0.1426 | 0.1483 | 0.06 | 8.7105 | 3.668 | 123.9 | post worse, resim worse |
| bal_inn_l2mmd_lr5e-4_s0 | 0.0085 | 0.0141 | 0.06 | 0.0030 | 3.668 | 246.8 | post better, resim better |

## 7. Why the baseline failed

Measured with `experiments/diagnose_inn_latent.py` on held-out data (healthy model = `kin_inn_ml_s0`):

| Quantity | Healthy INN (ML) | Failing INN (L2 + MMD) |
|---|---|---|
| per-dimension std of z | 0.994, 0.993 | 0.999, 0.991 |
| mean ‖z‖² (target 2) | 1.975 | 1.980 |
| fraction of entries with abs(z) > 3 (N(0,1): 0.0027) | 0.0025 | 0.0032 |
| largest abs(z) per dimension | 3.96, 3.83 | 5.08, 4.59 |
| re-simulation error, median | 4.33e-05 | 2.82e-04 |
| 90th percentile | 2.34e-04 | 4.31e-03 |
| 99th percentile | 6.90e-04 | 56.9 |
| 99.9th percentile | 1.56e-03 | 1.26e+08 |
| maximum | 0.103 | 8.19e+09 |
| draws with error > 1 | 0 | 1.95% |
| mean norm of z, all draws / worst 60 draws | 1.25 / 3.23 | 1.25 / 3.09 |
| largest standardised abs(x) produced | 5.2 | 361,761 |
| validation y error (training summary) | 0.00025 | 0.0018 |

What this shows:

| Cause | Evidence | Status |
|---|---|---|
| The latent is not the problem. | z statistics match N(0, I) in both models. | measured |
| Large-z draws are where it breaks, but only the failing model breaks there. | In both models the worst draws have a large z (about 3.1-3.2 vs 1.25 on average); the healthy model's worst error is 0.10, the failing model's is 8.2e9. | measured |
| The inverse pass stretches x without bound. | The largest standardised x is 361,761 vs 5.2. The largest error 8.2e9 corresponds to one coordinate of about 9e4 (square root of the error), and x1 enters the output linearly. | measured |
| The forward y fit is weaker. | Validation y error 7x worse; median re-simulation error 6.5x worse, 90th percentile 18x worse. | measured |
| Only the ML loss controls the flow's volume. | The ML loss contains `log det J`; the L2 + MMD loss only constrains the forward map and the latent distribution, so nothing limits how the inverse behaves away from the training data. Capping the scales (small clamp) removes most of the blow-up. | inferred |
| The mean explodes because it is an average over samples. | One sample at 1e8 raises a per-condition mean by about 1e5. The per-condition quartiles in the JSON (q3 = 317,944) show many conditions are affected, while the median over conditions is 0.0119. | measured |
| The training log showed it. | Provisional re-simulation error reached 1e5-1e6 from epoch 37 on while the validation MMD stayed near -3e-4 late in training (the unbiased estimator can be negative). | measured |

The paper's own INN (L2 + MMD) row is also its worst on re-simulation (0.086 on kinematics, 3.668 on ballistics), so a
weaker row is expected; ours was heavier-tailed than that.

## 8. Why some variants work and others fail

| Observation | Explanation | Status |
|---|---|---|
| Lower clamp fixes the kinematics blow-up (and 0.5 is better than 1.0: 0.0346 vs 0.1582). | It bounds the stretching per block, so the inverse cannot amplify exponentially. | inferred, consistent with the monotone trend |
| Lower `w_z` also fixes it (100 -> 10 -> 1: resim 599,845 -> 0.0188 -> 0.0105). | A stronger y loss makes the forward map accurate in more of the space the inverse visits, so the re-simulation error shrinks. | inferred |
| `w_z` 1 gets Err_post 0.0270, worse than `w_z` 10 (0.0180). | With the MMD almost switched off the latent no longer matches N(0, I) or stays independent of y, so the sampled posterior has the wrong shape. The balance lies near `w_z` 10. | inferred |
| The marginal MMD fails on both benchmarks. | The joint MMD compares `[y_hat, z]` with `[y, eps]`, where eps is independent of y, so it also forces z to be independent of y. The marginal MMD only requires z to look like N(0, I) overall; z may still depend on y. At test time z is drawn independently of y*, so the samples come from the wrong distribution. Test: run `python experiments/diagnose_inn_latent.py kin_inn_l2mmd_marg_s0` and compare the correlation between z and y (section 1b) with the baseline's. | inferred, not yet tested |
| `lr5e-4` has the best kinematics post but resim 0.16. | A gentler optimiser matches the latent better, but nothing bounds the inverse, so a few outliers remain. Its large change from 599,845 also means the baseline's blow-up is partly a matter of the training trajectory. | inferred |
| Ballistics works for every non-marginal setting. | There was no blow-up to fix: the baseline's largest per-condition re-simulation error was 0.0975. The y fit was the limiting factor, which is why lower `w_z` helped (4-10x) and a lower clamp, which only restricts the model, hurt (1.6-2.6x). | measured (the maximum), the rest inferred |
| Kinematics ranking differs from the paper: our ML INN (0.0037) beats every L2 + MMD variant (0.0153-0.0270), but in the paper L2 + MMD (0.017) beats ML (0.025). | The paper's training budget, optimiser, `sigma` and `w_z` are unknown. | open |

## 9. Status against the paper and next steps

| Row | Best setting found | Err_post / Err_resim | Paper | Status |
|---|---|---|---|---|
| kinematics INN (ML) | baseline | 0.0037 / 2.30e-04 | 0.025 / 0.015 | passes |
| kinematics cINN | baseline | 0.0035 / 1.10e-05 | 0.015 / 0.008 | passes |
| kinematics INN (L2 + MMD) | `w_z` 10, or `clamp1.0_wz10`, or `clamp0.5` | about 0.0175-0.0180 / 0.019-0.035 | 0.017 / 0.086 | resim passes; Err_post is 3-6% above the paper |
| ballistics INN (ML), INN (L2 + MMD), cINN | baseline (`w_z` 10 improves the L2 + MMD row) | see section 6 | 0.047 / 0.019, 0.060 / 3.668, 0.047 / 0.437 | pass |

Recommendation: use `w_z` 10 for the INN (L2 + MMD) row on both benchmarks (kinematics 0.0180 / 0.0188, ballistics
0.0085 / 7.95e-04). It is a single change that works on both and removes the blow-up. `clamp 1.0` with `w_z` 10 is the
alternative (0.0177 / 0.0237 and 0.0085 / 9.58e-04).

Caveats before reporting it: one seed; the setting was chosen on the test conditions; the differences between the top
variants (0.0175-0.0180 in Err_post, 0.007-0.035 in Err_resim) are within what a different seed could change; and the
paper's estimator, clamp and training budget are unknown, so "3-6% above" may not be meaningful.

Replication on two more seeds (core rows plus the two candidate settings), kinematics first:

```bash
PROBLEMS=kinematics SEEDS="1 2" EVAL=1 bash experiments/sweep_inn.sh l2mmd_best
```

That is 5 runs per seed (about 1.8 hours of training per seed); drop `PROBLEMS=kinematics` to also run ballistics. Afterwards,
compare the three-seed means of `w_z` 10, `clamp1.0_wz10` and the baseline.

## 10. Reproduce

```bash
EVAL=1 bash experiments/sweep_inn.sh l2mmd_wide        # the 16 sweep runs (kinematics and ballistics)
python experiments/diagnose_inn_latent.py kin_inn_l2mmd_s0
python experiments/diagnose_inn_latent.py kin_inn_ml_s0
python experiments/plot_inn_sweep.py                    # experiments/figures/inn_l2mmd_sweep.png
```
