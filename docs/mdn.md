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

## Caveats to discuss

- **Ballistics `x₄ = v0` is Poisson, so it is integer-valued.** `x₂` is clipped at 0,
  so it has a point mass there. A continuous density can put arbitrarily narrow
  spikes on such points, so the exact NLL is unbounded below. Check whether
  validation NLL keeps falling while re-simulation error does not improve. If
  so, consider dequantizing `v0` (add U(−½, ½)), and agree with the team so all
  models see the same data.
- **Upstream ballistics `forward_process` silently drops rows** when a
  trajectory has no ground impact in t ∈ [0, 6], which would misalign x and y.
  Our copy raises in strict mode and returns NaN otherwise. For model samples
  that start below ground, upstream (and so our copy) reports the impact at the
  apex. This only matters for re-simulation of out-of-prior samples.
- **Re-simulation error in `train_mdn.py` is provisional**: it is the mean
  squared Euclidean distance over 4000 samples × 1000 y*. The shared evaluation
  code should be the one we report. Posterior mismatch (MMD against
  rejection-sampled ground truth) is not implemented here, because it belongs
  to the shared code.
- **Inference time** = wall time to draw `n-eval-samples` posterior samples for
  each of 1000 y*, in batches of 100 y*, median of 5 runs after warm-up. The
  paper does not say exactly what its ms figure measures or what sample count
  it used. Compare relative to the other models on the same machine.
