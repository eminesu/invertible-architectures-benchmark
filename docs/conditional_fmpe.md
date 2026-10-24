# Prior-conditioned FMPE — decisions and findings

Iyi's proposal, discussed with Paul: condition FMPE not only on the observation
y\* but also on the **prior** it should assume, so that one trained model gives
the posterior under many priors without retraining. In Bayesian terms this is
an **amortized posterior over priors** (prior-conditioned posterior
estimation): the network learns p(x | y, c) for a whole family of priors
p_c(x), not a reparameterization of one fixed posterior.

Both benchmarks are done:
- **Ballistics:** c = λ, the Poisson rate of the launch speed v0 (benchmark 15).
- **Kinematics:** c = (σ₁, σ₂, σ₃), the standard deviations of the three
  joint angles (benchmark 0.5 each). The rail offset's σ₀ = 0.25 stays fixed
  (option (b) of the kinematics spec).

Base method: FMPE, Dax et al. (2023), arXiv:2305.17161, as implemented in
`models/fmpe.py` (docs/fmpe.md).

## Files

| Path | Purpose |
|---|---|
| `models/conditional_fmpe.py` | `ConditionalFMPE(..., prior_dims=k)`: FMPE whose conditioning vector is [y, c]; `sample(y_star, c, n)` |
| `data/toy_data.py` | `PRIOR_PARAMS` (names, benchmark values, training range per problem); `sample_prior(..., lam=)` / `sample_prior(..., sigmas=)`; `sample_prior_given`, `make_conditional_dataset(problem, ...)`, `prior_tag` |
| `metrics/ground_truth.py`, `metrics/evaluate.py`, `metrics/benchmarks.py` | Optional `prior_params` argument: rejection ground truth under that prior, cached per setting; None = benchmark prior, unchanged |
| `experiments/train_conditional_fmpe.py` | Training (`--problem`); same schedule as `train_fmpe.py`, per-epoch checkpoint with resume |
| `experiments/evaluate_conditional_fmpe.py` | `metrics.evaluate` per prior setting; `--plain-baseline` scores plain FMPE against each setting |
| `experiments/plot_conditional_fmpe_learning_curve.py` | Learning curve: train / val loss from `log.csv` |
| `experiments/plot_conditional_fmpe_posterior.py` | GT vs. conditional-FMPE trajectories / arms and marginals, one row per prior setting |
| `tests/test_conditional_fmpe.py` | Benchmark parameters reproduce the benchmark priors, both datasets, prior-dependent GT, cache keys, shapes, a toy recovery test |

```bash
# ballistics
.venv/bin/python experiments/train_conditional_fmpe.py --problem ballistics --device mps --run-name bal_cfmpe_s0
.venv/bin/python -m metrics.ground_truth ballistics:12 ballistics:18
.venv/bin/python experiments/evaluate_conditional_fmpe.py bal_cfmpe_s0 --prior 12 15 18 --plain-baseline bal_fmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_learning_curve.py bal_cfmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_posterior.py bal_cfmpe_s0 --index 200 346 966 --prior 12 15 18
# kinematics
.venv/bin/python experiments/train_conditional_fmpe.py --problem kinematics --device mps --run-name kin_cfmpe_s0
.venv/bin/python -m metrics.ground_truth kinematics:0.35,0.35,0.35 kinematics:0.65,0.65,0.65 kinematics:0.35,0.65,0.5
.venv/bin/python experiments/evaluate_conditional_fmpe.py kin_cfmpe_s0 \
    --prior 0.5,0.5,0.5 0.35,0.35,0.35 0.65,0.65,0.65 0.35,0.65,0.5 --plain-baseline kin_fmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_learning_curve.py kin_cfmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_posterior.py kin_cfmpe_s0 --index 499 401 882 \
    --prior 0.35,0.35,0.35 0.5,0.5,0.5 0.65,0.65,0.65 0.35,0.65,0.5
```

## Method

Shared by both problems:

- **Network.** c is appended to y: the conditioning vector [y, c] goes into the
  input layer where y went, standardized with the training mean / std like y.
  The gated residual blocks still gate on (t, x_t); nothing else changes. At the
  3M budget the width solver keeps width 497. That gives 2,993,932 parameters
  for ballistics and 2,995,423 for kinematics, against 2,993,435 / 2,993,932 for
  plain FMPE: the input layer gains 1 / 3 columns.
- **Data.** Per training sample: each prior parameter c_j ~ U[range], then
  x ~ p(x | c), y = f(x) (`make_conditional_dataset`).
- **Schedule.** Identical to the plain-FMPE headlines `bal_fmpe_s0` / `kin_fmpe_s0`:
  - 1M train / 20k val samples (seeds 10000 / 20000), 50 epochs, batch 1000
  - Adam 1e-3, weight decay 1e-5, cosine schedule, grad clip 10, σ_min = 1e-4
  - time prior p(t) ∝ t (α = 1, the headlines' effective setting, docs/comparison.md)
  - trained on MPS: 29 min (ballistics), 28 min (kinematics)
- **Ground truth.** The shared rejection sampler (σ = 0.02 / 0.01, 1000 samples
  per y\*, ≤ 50M / 2·10⁹ proposals), with the prior replaced by p(x | c). The
  benchmark parameters give the benchmark prior draw for draw (tested), so they
  reuse the benchmark cache file. Called without `prior_params`, all shared
  functions behave exactly as on main: same draws, same cache files (checked
  against main's code).
- **Test set.** The committed 1000 benchmark y\* for every prior setting. They
  were drawn under the benchmark prior, so under a narrower prior some y\* are
  tail events and get fewer ground-truth samples (`n_gt_min`). A y\* with a single
  GT sample cannot be scored by the unbiased MMD and drops out of the mean.
- **Headline solver:** midpoint × 16 (32 NFE), as for plain FMPE.

Ballistics:

- λ ~ U[10, 20], symmetric around 15, so 15 sits in the middle and the test values
  12 and 18 are interior. All 1M training trajectories have exactly one ground
  impact (the strict forward process did not raise).
- **Integer v0.** v0 stays integer (Poisson) and dequantization stays off, as
  for every other model. The integer-v0 floor on Err_post (docs/comparison.md)
  applies here too, and it varies with λ (Finding 3).
- GT shortfalls:
  - λ = 12: 6 y\* get fewer than 1000 samples (2 below 100, one with a single
    sample, so the mean is over 999 y\*).
  - λ = 18: 2 y\* fall short (min 445).

Kinematics:

- **Which σ to vary.** σ₁, σ₂, σ₃ independently, each ~ U[0.25, 0.75] (benchmark
  0.5, ±50 %). σ₀ of the rail stays 0.25. A constant input would have zero
  training std, and the rail offset is a plain shift, so it is left out of c.
  The joint angles are the multi-solution part.
- **Test settings:**
  - nominal (0.5, 0.5, 0.5): the no-regression check
  - tight (0.35, 0.35, 0.35) and wide (0.65, 0.65, 0.65)
  - mixed (0.35, 0.65, 0.5): uses each joint's σ separately
- GT shortfalls:
  - tight: 21 y\* below 1000 samples (one with a single sample, mean over 999 y\*)
  - mixed: 3 y\* below 1000 (min 5)
  - wide: none
  - nominal (benchmark cache): min 444

## Results (seed 0, midpoint × 16, 1000 y\* × 1000 samples, CPU)

Err_post is the unbiased MMD² mean. Err_resim is the mean squared re-simulation
error. Plain FMPE is the headline run, which assumes the benchmark prior,
scored against the same ground truth in the same run.

**Ballistics**

| λ of the ground truth | Conditional FMPE Err_post | Plain FMPE Err_post | Conditional Err_resim | Plain Err_resim |
|---|---|---|---|---|
| 12 | **0.0105** | 0.0354 | 5.7e-5 | 4.5e-5 |
| 15 (benchmark) | **0.0071** | **0.0071** | 5.7e-5 | 4.5e-5 |
| 18 | **0.0048** | 0.0296 | 7.0e-5 | 4.5e-5 |

Medians (conditional): 0.0105 / 0.0066 / 0.0038. No sample of either model has
an undefined f(x).

**Kinematics**

| σ₁, σ₂, σ₃ of the ground truth | Conditional FMPE Err_post | Plain FMPE Err_post | Ratio | Conditional better on | Conditional Err_resim | Plain Err_resim |
|---|---|---|---|---|---|---|
| 0.5, 0.5, 0.5 (benchmark) | 0.0017 | **0.0016** | – | – | 3.1e-6 | 2.7e-6 |
| 0.35, 0.35, 0.35 | **0.0017** | 0.0374 | 22.5× | 99 % of y\* | 4.5e-6 | 2.7e-6 |
| 0.65, 0.65, 0.65 | **0.0017** | 0.0155 | 9.1× | 95 % of y\* | 7.7e-6 | 2.7e-6 |
| 0.35, 0.65, 0.5 | **0.0017** | 0.0445 | 26.6× | 96 % of y\* | 5.1e-6 | 2.7e-6 |

Medians (conditional): 0.0002–0.0004; the means are carried by a few hard y\*,
as for plain FMPE. Inference takes ~400 s per setting for 10⁶ samples on CPU,
for both models and both problems (same network size and NFE).

### Finding 1 — no regression on the benchmark prior

- **Ballistics, λ = 15:** 0.00713 vs. 0.00712, paired Wilcoxon over the 1000 y\*
  p = 0.15. No measurable difference.
- **Kinematics, nominal σ:** 0.00167 vs. 0.00164. The difference is 0.00004
  (2 %), median 0.00001, and the conditional model is worse on 52 % of y\*. It is
  statistically detectable (p = 0.022) but smaller than any difference between
  methods in docs/comparison.md. The plain model scores 0.0016 on all 3 seeds,
  so this is a small but real cost of spreading the same capacity over a family
  of priors, not seed noise. It is still at the plain-FMPE level and ahead of
  every non-FMPE model.
- Err_resim stays orders of magnitude below the ground truth's own σ² in both
  problems: samples hit y\* as precisely as the GT does.

### Finding 2 — the model uses the prior

Off the benchmark prior, the plain FMPE can only produce the benchmark
posterior, and it scores 3–27× worse:

- ballistics: 3.4× / 6.2× at λ = 12 / 18
- kinematics: 9–27×, and the conditional model is better on 95–99 % of y\*

Err_resim alone cannot show this: both models hit y\* equally well. The
difference is *which* x they put on the solution set, and that is what the
prior decides. The figures show it per y\*:

- **Ballistics, near impact (y\* = 3):** the v0 posterior mode follows the prior
  (≈ 9 / 14 / 17 for λ = 12 / 15 / 18). The second, low-angle mode of slow, flat
  shots, present at λ = 12, vanishes at λ = 18, where almost every solution is a
  steep lob.
- **Ballistics, middle impact (y\* = 8):** the angle mass moves from low
  (≈ 0.3 rad) to high (≈ 1.0 rad) angles, and the v0 tail widens.
- **Ballistics, far impact (y\* = 13):** the likelihood dominates (v0 ≈ 21 under
  all three priors) and the posterior barely moves, again matching the GT.
- **Kinematics, y\* = (1.13, 0.51):** under the wide prior a second arm
  configuration appears (elbow flipped: joint 1 ≈ +1.5, joint 3 ≈ −1.5 rad). It
  is weak at nominal and absent under the tight prior, where the rail instead
  has to drop below its 3σ band to reach the target.
- **Kinematics, far target (1.79, 0.60):** the arm is nearly stretched out. The
  joint-angle posteriors widen from tight → nominal → wide, and in the mixed row
  each joint follows its own σ (joint 1 narrow, joint 2 wide), so the model uses
  the σ components separately, not just their scale.
- **Kinematics, folded target (0.15, −1.10):** same pattern, with wider tails
  under the wide prior.

The conditional samples follow the ground truth in every row.

### Finding 3 — ballistics Err_post grows as λ falls; likely the integer-v0 floor

Err_post goes 0.0048 → 0.0071 → 0.0105 from λ = 18 to 12.

- It is not a ground-truth sample-count effect: only 2 y\* have < 100 GT samples
  at λ = 12.
- It is not a tail effect either: the error rises at every impact distance.

  | y\* range | λ = 18 | λ = 15 | λ = 12 |
  |---|---|---|---|
  | [0, 5) | 0.0029 | 0.0031 | 0.0045 |
  | [5, 10) | 0.0045 | 0.0077 | 0.0120 |

- The GT v0 posterior gets narrower as λ falls (mean per-y\* sd 3.45 / 2.75 /
  2.06). With fewer integer values carrying the mass, the continuous model
  samples sit relatively further from the integer ground truth under the MMD
  kernel. That is the integer-v0 floor of docs/comparison.md, scaled by the
  posterior width.

Likely, not proven; a dequantized ground truth (or rounding v0 of the samples)
would settle it. Kinematics has no integer dimension and shows no such trend:
0.0017 at every σ setting.

## Figures

**Learning curves**
- `experiments/figures/cfmpe_learning_curve_{bal,kin}_cfmpe_s0.png`: train and
  val flow-matching loss per epoch.
- Both runs fall and converge with no overfitting. Final values: ballistics
  4.595 / 4.607 (best val at epoch 47, equal to the final one); kinematics
  3.121 / 3.124 (best val at the last epoch, still creeping down as the cosine
  schedule ends).
- The val loss uses fixed (x0, t) draws, while the train loss is the epoch mean,
  so the small train/val gaps are noise level.
- The absolute values differ between problems and are not a quality measure:
  flow matching has an irreducible floor. Err_post / Err_resim measure quality.

**Posterior figures**
- `experiments/figures/cfmpe_posterior_bal_cfmpe_s0_y{200,346,966}.png`:
  y\* = 3.0, 8.0, 13.0, one row per λ ∈ {12, 15, 18}. Left: 100 ground-truth
  (gray) and 100 conditional-FMPE (blue) trajectories, with y\* marked. Right:
  the four marginals, with the prior mean marked on v0.
- `experiments/figures/cfmpe_posterior_kin_cfmpe_s0_y{499,401,882}.png`:
  y\* = (1.13, 0.51), (1.79, 0.60), (0.15, −1.10), one row per σ setting (tight,
  nominal, wide, mixed). Left: 100 GT and 100 model arm configurations, with
  re-simulated end points and y\*. Right: rail offset and the three joint angles.
- Same style as `plot_fmpe_posterior.py`. Marginal x-ranges are shared across
  rows, so shifts read directly.

**Valid region** (shaded green): the support or bulk of the prior.
- **Ballistics:**
  - launch |x0| ≤ 1.5 (3σ)
  - launch height 0 ≤ y0 ≤ 3.0 (the prior clamps y0 at 0; 3σ above the mean 1.5)
  - angle 9°–81° (the prior's hard bounds)
  - v0 between the 0.1 % and 99.9 % quantiles of Poisson(λ), so the v0 band
    moves with λ
- **Kinematics:** |x_j| ≤ 3σ_j per dimension, i.e. joint limits at three
  standard deviations of the assumed prior. The joint bands move with the
  conditioned σ; the rail band (|x0| ≤ 0.75) is drawn on the rail in the arm
  panel.

Each row title gives the fraction of samples inside all four bounds:
- Ballistics: model 96.7–99.2 %, GT 98.3–99.8 %.
- Kinematics: model and GT agree within a few points in every row. The fraction
  drops to ~35–60 % where the target forces large joint angles under a tight
  prior.

The GT also falls below 100 %, because these bounds describe the bulk of the
prior, not hard support (only the ballistics angle bound is hard).

## Open items for the team / supervisor

- **Ranges and test settings.** Used λ ∈ [10, 20] (test 12 / 15 / 18) and joint
  σ ∈ [0.25, 0.75] (test 0.35 / 0.5 / 0.65 and one mixed setting). Confirm, or
  widen. Test y\* are drawn under the benchmark prior, so priors further away
  push more y\* into the tail.
- **Which kinematics σ.** Chose joint-only σ₁..σ₃ (option b). Varying σ₀ too
  (option a) or a single shared joint scale are the alternatives.
- **Test y\* per prior.** We score every setting on the committed benchmark y\*.
  Drawing test y\* under each prior would match test conditions to prior; that
  would be a separate test set, not the benchmark's.
- **Joint-limit priors.** For the eventual 7-DoF transfer, Paul may prefer
  constraint priors (joint limits as hard bounds) over Gaussian widths. The
  mechanism carries over: c would hold the limits. Ask before building it.
- **Framing:** amortized-over-priors posterior estimation, as above.
- **Seeds:** one seed per problem. The kinematics nominal gap (Finding 1) is
  2 %. A second seed would show whether it is stable.
