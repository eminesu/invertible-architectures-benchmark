# Shared evaluation metrics

Every model (MDN, INN, cINN, autoencoder, …) is scored by `metrics/`; do not
re-implement it. Reference: Kruse et al. (2021), arXiv:2101.10763, Sec. 4,
Eqs. (2), (10), (11).

## Plugging in a model

A model only needs

```python
sample(y_star, n) -> x    # y_star: (M, dim_y) float32 numpy, x: (M, n, 4) numpy or torch, raw x units
```

```python
from metrics.evaluate import evaluate
evaluate(sample, 'kinematics', name='cinn_s0')   # prints the table row
```

This writes `metrics/results/<benchmark>_<name>.npz` (per-y* arrays for the
boxplots) and `.json` (summary). `metrics.evaluate.boxplot([...npz], 'fig.png')`
draws the log-scale boxplots of Figs. 4–5.

Prior-only sanity baseline (must score badly):

```bash
.venv/bin/python -m metrics.evaluate kinematics --prior-baseline
```

## Fixed pieces (identical for every model)

| Piece | Value | Where |
|---|---|---|
| Test conditions | 1000 y* = f(x_true), x_true ~ prior, seed 20210126, committed | `metrics/conditions/*.npz` |
| Ground truth | rejection sampling, 1000 accepted x per y*, seed 777, cached (not committed, regenerated deterministically) | `metrics/ground_truth.py`, `metrics/cache/` |
| Likelihood width σ | kinematics 0.01, ballistics 0.02 | `GT_SIGMA` |
| Model samples per y* | 1000 | `N_SAMPLES` |
| MMD kernel | sum of IMQ kernels `1/(1+‖a−b‖²/h²)`, h ∈ {0.05, 0.2, 0.9}, raw x units | `metrics/mmd.py` |
| MMD estimator | unbiased MMD² (U-statistic); biased reported alongside | `metrics/mmd.py` |
| Ballistics clamp | per-y* values clamped at 10 for the "clamped mean" | `CLAMP` |

### Why these choices

- **Kernel.** Neither the paper nor `inn_toy_data` pins one. The multi-scale IMQ
  with h ∈ {0.05, 0.2, 0.9} is what the same group's INN code (Ardizzone et al.
  2019 / FrEIA toy demos) uses.
- **σ.** The forward processes are deterministic, so y | x ~ N(f(x), σ²I) is
  imposed. Ground-truth samples then have their own re-simulation error dim_y·σ²
  (2e-4 kinematics, 4e-4 ballistics), well below the best paper numbers
  (0.008, 0.019), and the blur in x is below the smallest kernel bandwidth.
  Ballistics uses a larger σ because its upstream f is piecewise constant on a
  1500-step time grid (steps of ~0.01–0.03 in y).
- **Proposal cap.** Some y* lie far in the tails of p(y) and do not reach 1000
  accepted samples within the cap (kinematics: 2e9 proposals; 2 of 1000 y*
  short, min 444; ballistics: 5e7 proposals; 1 of 1000 short, min 111). They
  are scored against the samples they have.
- **Clamp.** The paper's clamped Table 2 means reach 4.36 and 3.67, so its
  clamp is at least that; 10 is the smallest round value consistent with every
  entry. Err_post cannot exceed 2·k(0) = 6 with this kernel, so in practice
  only Err_resim is clamped.
- **Unbiased MMD².** The biased estimator adds ≈ 3·(1/n + 1/m) = 0.006 at
  n = m = 1000, comparable to the paper's best Err_post values, so differences
  between good models would mostly be sample-size noise.
- **Ballistics failures.** A model x whose trajectory has no single ground
  impact has no f(x). Such samples are left out of Err_resim and reported as
  `resim_failed_fraction`.

## Validation

Kinematics, 1000 y* × 1000 samples:

| Model | Err_post (unbiased) | Err_post (biased) | Err_resim |
|---|---|---|---|
| prior (ignores y*) | 0.242 | 0.246 | 2.73 |
| MDN K=16 (`emine-mdn`, `kin_K16_s0`) | 0.0031 | 0.0077 | 0.0007 |
| Paper MDN, Table 1 | 0.007 | | 0.012 |

Ballistics (no trained model yet):

| Model | Err_post (unbiased) | Err_post (biased) | Err_resim mean | Err_resim median [q1, q3] |
|---|---|---|---|---|
| prior (ignores y*) | 0.105 | 0.110 | 19.5 | 14.8 [11.0, 22.9] |
| Paper MDN, Table 2 | 0.048 | | 0.184 | |

On ballistics the prior's Err_post is only ~2× the paper's MDN, while its
Err_resim is ~100× worse. In raw x units the kernel distance is dominated by
v0 (prior std ≈ 3.9, vs. ≈ 0.5 for the other coordinates), and the
posterior over v0 for a given impact point is broad. Check this once a
ballistics model is trained; if Err_post separates models poorly, the team
may want to standardize x before the MMD (for all models).

Sanity plots of one y*: `experiments/figures/ground_truth_{kinematics,ballistics}_0.png`
(`experiments/plot_ground_truth.py`).

## Open items for the team

- Agree on the unbiased vs. biased MMD² headline number (the paper does not say).
- Agree on the ballistics clamp value (the paper clamps but gives no value).
- Decide whether to standardize x before the MMD on ballistics (see above).
- σ, kernel and sample counts above must not change between models; changing
  any of them invalidates every number already reported.
