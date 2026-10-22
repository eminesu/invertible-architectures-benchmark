# Prior-conditioned FMPE — decisions and findings

Iyi's proposal, discussed with Paul: condition FMPE not only on the observation
y\* but also on the **prior** it should assume, so that one trained model gives
the posterior under many priors without retraining. In Bayesian terms this is
an **amortized posterior over priors** (prior-conditioned posterior
estimation): the network learns p(x | y, λ) for a whole family of priors
p_λ(x), not a reparameterization of one fixed posterior.

Scope (agreed): ballistics first, with the Poisson rate λ of the launch speed
v0 as the prior parameter. Kinematics (Gaussian σs as the prior parameter) is
an open extension, see the last section.

Base method: FMPE, Dax et al. (2023), arXiv:2305.17161, as implemented in
`models/fmpe.py` (docs/fmpe.md).

## Files

| Path | Purpose |
|---|---|
| `models/conditional_fmpe.py` | `ConditionalFMPE`: FMPE whose conditioning vector is [y, λ]; `sample(y_star, lam, n)` |
| `data/toy_data.py` | `sample_prior(..., lam=)` and `make_conditional_dataset` (λ ~ U[10, 20] per sample); `LAM_RANGE` |
| `metrics/ground_truth.py`, `metrics/evaluate.py`, `metrics/benchmarks.py` | Optional `lam` argument: rejection ground truth under v0 ~ Poisson(λ), cached per λ |
| `experiments/train_conditional_fmpe.py` | Training; same schedule as `train_fmpe.py`, per-epoch checkpoint with resume |
| `experiments/evaluate_conditional_fmpe.py` | `metrics.evaluate` per λ; `--plain-baseline` scores plain FMPE against each λ |
| `experiments/plot_conditional_fmpe_learning_curve.py` | Figure 2: train / val loss from `log.csv` |
| `experiments/plot_conditional_fmpe_posterior.py` | Figure 1: GT vs. conditional-FMPE trajectories and marginals, one row per λ |
| `tests/test_conditional_fmpe.py` | λ = 15 reproduces the benchmark prior, dataset, λ-dependent GT, cache key, shapes, a toy recovery test |

```bash
.venv/bin/python experiments/train_conditional_fmpe.py --device mps --run-name bal_cfmpe_s0
.venv/bin/python -m metrics.ground_truth ballistics:12 ballistics:18
.venv/bin/python experiments/evaluate_conditional_fmpe.py bal_cfmpe_s0 --lam 12 15 18 --solver midpoint:16 --plain-baseline bal_fmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_learning_curve.py bal_cfmpe_s0
.venv/bin/python experiments/plot_conditional_fmpe_posterior.py bal_cfmpe_s0 --index 200 346 966
```

## Method

- **Data.** Per training sample: λ ~ U[10, 20], then x ~ p(x | λ) (the benchmark
  prior with v0 ~ Poisson(λ) instead of Poisson(15)), y = f(x). The triples
  (x, y, λ) are stored with raw λ; the network sees λ standardized with the
  training mean / std, like y. The range 10–20 is symmetric around the
  benchmark value 15, so λ = 15 is in the middle of the training range and the
  test values 12 and 18 are interior too. Every one of the 1M training
  trajectories has exactly one ground impact (the strict forward process did
  not raise), so the range needs no redraws.
- **Integer v0.** v0 stays integer (Poisson); dequantization stays off, exactly
  as for every other model. Nothing changes there: the integer-v0 floor on
  Err_post that docs/comparison.md describes applies here too, and it varies
  with λ (see Finding 3).
- **Network.** λ is one more conditioning dimension: c = [y, λ] goes into the
  input layer where y went; the gated residual blocks still gate on (t, x_t).
  No other change. At the 3M budget the width solver keeps width 497
  (2,993,932 parameters vs. 2,993,435 for plain FMPE; the input layer gains one
  column).
- **Schedule.** Identical to the plain-FMPE headline `bal_fmpe_s0`: 1M train /
  20k val samples (seeds 10000 / 20000), 50 epochs, batch 1000, Adam 1e-3, weight
  decay 1e-5, cosine, grad clip 10, σ_min = 1e-4, time prior p(t) ∝ t (α = 1, the
  headline's effective setting, docs/comparison.md). Trained on MPS, 29 min.
- **Ground truth.** The shared rejection sampler (σ = 0.02, 1000 samples per y\*,
  ≤ 50M proposals) with the prior's v0 rate set to λ. λ = 15 gives the
  benchmark prior draw for draw (tested), so it reuses the benchmark cache file.
- **Test set.** The committed 1000 benchmark y\* for every λ. They were drawn
  under λ = 15, so under λ = 12 a few far impacts are tail events of the prior:
  6 y\* get fewer than 1000 ground-truth samples (2 below 100, one with a single
  sample, which the unbiased MMD cannot score, so λ = 12 averages 999 y\*). Under
  λ = 18, 2 y\* fall short (min 445).
- **Headline solver:** midpoint × 16 (32 NFE), as for plain FMPE.

## Results (seed 0, midpoint × 16, 1000 y\* × 1000 samples, CPU)

Err_post is the unbiased MMD² mean; Err_resim the mean squared re-simulation error.

| λ of the ground truth | Conditional FMPE Err_post | Plain FMPE (assumes λ = 15) Err_post | Conditional Err_resim | Plain Err_resim |
|---|---|---|---|---|
| 12 | **0.0105** | 0.0354 | 5.7e-5 | 4.5e-5 |
| 15 | **0.0071** | **0.0071** | 5.7e-5 | 4.5e-5 |
| 18 | **0.0048** | 0.0296 | 7.0e-5 | 4.5e-5 |

Medians (conditional): 0.0105 / 0.0066 / 0.0038. No sample of either model has
an undefined f(x). Inference ~400 s per λ for 10⁶ samples on CPU for both
(same network size and NFE).

### Finding 1 — no regression at λ = 15

The conditional model matches the plain FMPE at λ = 15: Err_post 0.0071 for both
(plain re-scored in the same run; the committed headline is 0.0071 as well).
Err_resim is 5.7e-5 vs. 4.5e-5; both are three orders of magnitude below the
ground truth's own σ² = 4e-4, so samples hit y\* as precisely as the GT does.
Adding the prior as an input cost nothing on the benchmark prior.

### Finding 2 — the model uses the prior

At λ = 12 and 18 the plain FMPE, which can only produce the λ = 15 posterior,
scores 0.0354 and 0.0296. The conditional model scores 0.0105 and 0.0048, 3.4×
and 6.2× lower. Err_resim alone cannot show this: both models hit y\* equally
well, the difference is *which* x they put on the solution set, and that is what
the prior decides. Figure 1 shows it per y\*. Near impact (y\* = 3): the v0
posterior mode follows the prior (≈ 9 / 14 / 17 for λ = 12 / 15 / 18), and the
second, low-angle mode of slow flat shots that is present at λ = 12 vanishes at
λ = 18, where almost every solution is a steep lob. Middle impact (y\* = 8): the
angle mass moves from low (≈ 0.3 rad) to high (≈ 1.0 rad) angles and the v0
tail widens. The conditional samples follow the ground truth in each row. For a far impact (y\* = 13) the likelihood dominates (v0 ≈ 21 under all
three priors) and the posterior barely moves, again matching the ground truth.

### Finding 3 — Err_post grows as λ falls; likely the integer-v0 floor

Err_post goes 0.0048 → 0.0071 → 0.0105 from λ = 18 to 12. This is not a
ground-truth sample-count effect (only 2 y\* have < 100 GT samples at λ = 12) or
a tail effect: the error rises at every impact distance (y\* in [0, 5):
0.0029 / 0.0031 / 0.0045 for λ = 18 / 15 / 12; [5, 10): 0.0045 / 0.0077 / 0.0120).
The ground-truth v0 posterior gets narrower as λ falls (mean per-y\* sd 3.45 /
2.75 / 2.06), and with fewer integer values carrying the mass, the continuous
model samples sit relatively further from the integer ground truth under the
MMD kernel. That is the integer-v0 floor of docs/comparison.md, scaled by the
posterior width. Likely, not proven; a dequantized ground truth (or rounding v0
of the samples) would settle it.

## Figures

- `experiments/figures/cfmpe_learning_curve_bal_cfmpe_s0.png` (Figure 2): train
  and val flow-matching loss per epoch. Both fall and converge (final 4.595 /
  4.607, best val at epoch 47 equal to the final value); no overfitting. The val
  loss uses fixed (x0, t) draws, the train loss is the epoch mean, so the small
  train < val gap is the usual noise level, not divergence.
- `experiments/figures/cfmpe_posterior_bal_cfmpe_s0_y{200,346,966}.png` (Figure
  1): y\* = 3.0, 8.0, 13.0; one row per λ ∈ {12, 15, 18}. Left: 100 ground-truth
  (gray) and 100 conditional-FMPE (blue) trajectories, y\* marked; right: the
  four marginals, GT filled gray vs. model blue outline, with the prior mean
  marked on v0. Same style as `plot_fmpe_posterior.py`.

**Valid region** (shaded green in Figure 1): the support or bulk of the prior —
launch |x0| ≤ 1.5 (3σ), launch height 0 ≤ y0 ≤ 3.0 (the prior clamps y0 at 0;
3σ above the mean 1.5), angle 9°–81° (the prior's hard bounds), v0 between the
0.1 % and 99.9 % quantiles of Poisson(λ) (so the v0 band moves with λ). Each row
title gives the fraction of samples inside all four: 96.7–98.9 % for the model,
98.3–99.7 % for the ground truth on these y\*. The GT is below 100 % too because
the x0 / y0 / v0 bounds are a bulk definition, not hard support; only the angle
bound is hard.

## Open items for the team / supervisor

- **λ range and test values.** Used 10–20 for training and {12, 15, 18} for
  testing; confirm, or widen if Paul wants priors further from 15 (the test y\*
  are drawn under λ = 15, so far-away λ will push more y\* into the prior's tail).
- **Test y\* per λ.** We score every λ on the committed benchmark y\*. Drawing
  test y\* under each λ instead would match test conditions to prior; that is a
  separate test set, not the benchmark's.
- **Kinematics** (Gaussian σs as the prior parameter, joint-limit-style priors
  for the eventual 7-DoF transfer) — ask Paul before starting.
- **Framing:** amortized-over-priors posterior estimation, as above.
- **Seeds:** one seed so far; plain FMPE ballistics is seed-stable at 0.0071
  (docs/comparison.md), the conditional model has not been checked across seeds.
