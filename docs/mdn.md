# MDN with full covariance — decisions and findings

Baseline from Kruse et al. (2021), Eq. (9), implemented after the MDN
technical report (Kruse 2020, arXiv:2003.05739, Sec. 2).

## Files

| Path | Purpose |
|---|---|
| `data/toy_data.py` | Vendored priors + forward processes from `vislearn/inn_toy_data` (commit 357bc08) |
| `models/mdn.py` | Cholesky-of-precision parameterization, losses, sampling, network |
| `tests/test_mdn.py` | Numerical checks, incl. equivalence with FrEIA's `GaussianMixtureModel` |
| `experiments/plot_data.py` | Step 1: prior / forward-process plots → `experiments/figures/data_overview.png` |
| `experiments/toy_2d_sanity.py` | Step 5: tilted 2-D mixture recovery → `experiments/figures/toy_2d_sanity.png` |
| `experiments/plot_posterior.py` | Posterior samples for a few y* of a trained kinematics run |
| `experiments/train_mdn.py` | Training + timing + provisional re-simulation error; outputs in `experiments/runs/` (git-ignored) |

Setup:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```

## Build vs. reuse: implemented from scratch, verified against FrEIA

We use our own plain-PyTorch implementation, using **the same `U` entry layout
as FrEIA**. `tests/test_mdn.py::test_matches_freia_gaussian_mixture_model`
checks that latent codes, log-Jacobians and NLL agree with FrEIA to 1e-10.

Reasons for not using FrEIA's block directly (FrEIA commit 4212496):

- `nll_loss` computes `log(sum(w * exp(...)))` directly, with no `logsumexp`. It
  underflows to `inf` for points far from every component.
- Sampling a fixed component loops over the batch in Python and uses
  `torch.inverse` instead of a triangular solve. That would inflate the
  inference-time numbers we are asked to report.
- Its `nll_upper_bound` implements Eq. (15) literally, which has a problem (see below).

## Resolved: shape of `U`

The report says `U ∈ R^{b×K×N(N−1)/2}`. That is a typo. FrEIA expects
**`N(N+1)/2`** entries per component (10 for N=4):

- `U[..., :N]` is the diagonal in log space (`Ū_jj = exp(U_jj)`)
- `U[..., N:]` is the strict upper triangle in row-major order, used as-is

## Errors found in the MDN report

1. **Sampling factor (Sec. 2.3).** The report says `L = Ū⁻ᵀ`. From
   `Σ⁻¹ = ŪᵀŪ` we get `Σ = Ū⁻¹Ū⁻ᵀ`, so the correct factor is **`L = Ū⁻¹`**.
   With `Ū⁻ᵀ` the samples have covariance `(ŪŪᵀ)⁻¹ ≠ Σ`; on a random 4×4
   example the error is about 40% of the largest covariance entry. FrEIA uses
   `Ū⁻¹`, and so do we. The sample is `x = μ + solve_triangular(Ū, η, upper=True)`.
   `test_sampling_reproduces_full_covariance` would catch the wrong version.
2. **"Jensen bound" Eq. (15).** As printed, `−Σᵢ (log ωᵢ + log pᵢ)` is not an
   upper bound on the NLL. Jensen gives `−log Σ ωᵢpᵢ ≤ −Σ ωᵢ log pᵢ`, with ω as
   weights outside the log. The printed version makes every component fit every
   point, and its `Σ log ωᵢ` term pushes the weights toward uniform. We provide:
   - `jensen`: the valid weighted bound (default warm-up)
   - `eq15`: the report's expression, kept only for comparison with FrEIA
   - `exact`: Eq. (14) via `logsumexp`, used after warm-up

## Our choices (the paper does not specify these)

| Choice | Value | Notes |
|---|---|---|
| Components K | 16 (to sweep: 4, 8, 16, 32) | |
| Trunk | 4 × Linear + ReLU, width solved to fit ≤ 3M params | K=16 gives width 959 and 2.995M params |
| Head | Linear → K·(1 + N + N(N+1)/2) | 240 outputs at K=16, N=4; about 230k of the budget |
| Head init | weights ×0.1, bias 0 | Starts near identity precision and uniform weights |
| Normalization | x and y standardized with training-set statistics | NLL is reported in raw x-space |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5, cosine schedule, grad-clip 10 | |
| Warm-up | 500 steps of `jensen`, then `exact` | |
| Data | 1M train, 20k val, 1000 test y*, disjoint seeds | **Placeholder until the team agrees on a schedule** |
| Epochs / batch | 50 / 1000 | **Placeholder** |

## Results (K=16, seed 0, shared metrics from `emine-metrics`)

Scored with `metrics.evaluate` (`emine-metrics`, commit b0f43d8):
the same 1000 test y\* and cached rejection-sampling ground truth as every
other model, 1000 samples per y\*, unbiased MMD² (biased in brackets).

| Benchmark | Err_post | Err_resim (median [q1, q3]) | Inference, 1000 y\* × 1000 samples | Paper (Err_post / Err_resim / ms) |
|---|---|---|---|---|
| Kinematics | 0.0031 (0.0077) | 0.0007 (0.0003 [0.0002, 0.0006]) | 75 ms | 0.007 / 0.012 / 601 |
| Ballistics | 0.0071 (0.0127) | 0.0012 (0.0011 [0.0009, 0.0014]) | 74 ms | 0.048 / 0.184 / 175 |

Prior-only baseline for scale: Err_post 0.242 / 0.105, Err_resim 2.73 / 19.5.
On ballistics no sample was without a ground impact (`resim_failed` = 0).

Hardware: Apple M4, CPU, idle machine. Training took 18 / 28 min. Timings are
only comparable with other models measured on the same machine.

**We beat the paper's MDN on every metric, by 2–7× on Err_post and 15–150× on
Err_resim. This is not explained yet.** Our provisional Err_resim from
`train_mdn.py` agrees with the shared code, so it is not a metric bug on our
side. Candidate explanations, none of them checked:
- The paper's Err_post may use the biased estimator. Kinematics biased (0.0077)
  is close to the paper's 0.007, but ballistics (0.0127 vs 0.048) is not.
- A different kernel, ground-truth tolerance or sample count in the paper.
- A longer schedule or more data than the paper's. Ours is 1M samples × 50 epochs,
  still a placeholder.
- The paper notes its ballistics means are distorted by extreme outliers. Our
  model has none (median ≈ mean).

**Known weakness: low-density y\*.** At y\* = (1.5, 0), the paper's own
example point, only 115 of 1M training points lie within 0.05. There, about
55% of MDN samples re-simulate to within 0.05 of y\*, against 97% at a dense
point such as (1.0, 1.2). See `experiments/figures/posterior_kin_K16_s0.png`.
Test y\* are drawn from the prior, so the averaged Err_resim hides this. The K
sweep and a longer schedule should test whether it improves.

The result files of these runs are committed (force-added despite
`.gitignore`): `metrics/results/{kinematics,ballistics}_mdn_K16_s0.{json,npz}`
and `experiments/runs/{kin,bal}_K16_s0/{config.json,log.csv,summary.json}`.
Model weights are not in git.

## Sensitivity of Err_post to the evaluation settings

Checked with the cached ground truth, using the same K=16 models and
1000 samples per y\*. Values are mean Err_post, unbiased unless marked.

| Variant | Kin MDN | Kin prior | Bal MDN | Bal prior |
|---|---|---|---|---|
| Shared setting: IMQ (.05, .2, .9), raw x | 0.0030 | 0.242 | 0.0071 | 0.105 |
| same, biased estimator | 0.0077 | 0.246 | 0.0127 | 0.111 |
| same, x standardized by prior std | 0.0033 | 0.172 | 0.0010 | 0.068 |
| IMQ bandwidths ×2, raw | 0.0028 | 0.265 | 0.0076 | 0.175 |
| IMQ bandwidths ÷2, raw | 0.0030 | 0.176 | 0.0076 | 0.051 |
| Gaussian h = 0.2, raw | 0.0017 | 0.071 | 0.0069 | 0.011 |
| Bal: MDN v0 samples rounded to integers, raw | | | 0.0012 | |

Findings:
- **The MDN's Err_post barely depends on the bandwidths** (0.0028–0.0034 on
  kinematics, 0.0069–0.0076 on ballistics). The kernel scale alone does not
  explain the gap to the paper. Of the variants above, only the biased
  estimator comes close to the paper's kinematics number (0.0077 vs 0.007).
- **The kernel choice does change how well the metric separates models.** A
  narrow Gaussian kernel puts the prior only 1.6× above the MDN on ballistics,
  against 15× with the shared IMQ setting.
- **On ballistics, over 80% of the MDN's Err_post comes from v0 being
  integer-valued.** Every ground-truth v0 is an integer (Poisson prior), while
  the MDN outputs continuous values. Rounding only the MDN's v0 drops
  Err_post from 0.0071 to 0.0012. Standardizing x (0.0010) has nearly the
  same effect, because it shrinks the v0 axis by 3.9×. This affects every
  continuous model (INN, cINN, ...) equally, but it means the raw-unit
  ballistics Err_post mostly measures "is v0 an integer". The team should
  decide on standardization, rounding or dequantization *before* comparing
  models on ballistics.

## Caveats to discuss

- **Ballistics `x₄ = v0` is integer-valued.** The parameters are
  x = (x₁, x₂) launch position, x₃ launch angle, x₄ = v0 initial speed. The
  paper defines the prior as x₄ ~ Poisson(15) (Sec. 3.2), and upstream
  implements it with `np.random.poisson(15)`, so every training and
  ground-truth v0 is a whole number (mostly 9–21). The paper gives no reason
  for this choice. Also, upstream clips x₂ at 0, which the paper does not
  mention, so x₂ has a point mass at 0 (about 0.1% of draws).
  Two consequences:
  1. **Training.** A continuous density can put arbitrarily narrow spikes on
     integer values, so the exact NLL is unbounded below. In the K=16 run this
     did not happen: train and validation NLL stayed level and Err_resim is
     good. Watch for it at larger K or with longer schedules. The fix would be
     dequantization (add U(−½, ½) to v0 in the training data), agreed with the
     team so every model sees the same data.
  2. **Evaluation.** See "Sensitivity" above: over 80% of the MDN's raw-unit
     ballistics Err_post comes from this.
- **Upstream ballistics `forward_process` silently drops rows** when a
  trajectory has no ground impact in t ∈ [0, 6], which would misalign x and y.
  Our copy raises in strict mode and returns NaN otherwise. For model samples
  that start below ground, upstream (and so our copy) reports the impact at the
  apex. This only matters for re-simulation of out-of-prior samples.
- **Re-simulation error in `train_mdn.py` is provisional**: it is the mean
  squared Euclidean distance over 4000 samples × 1000 y*, used only to compare
  runs during development. Report the shared `metrics.evaluate` numbers.
- **Inference time**: the reported number is from `metrics.evaluate`, which is
  one call for all 1000 y* after warm-up. `train_mdn.py` logs its own
  variant (batches of 100 y*, median of 5). The paper does not say exactly
  what its ms figure measures. Compare against other models on the same
  machine only.
