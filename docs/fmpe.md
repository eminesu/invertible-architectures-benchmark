# Flow Matching Posterior Estimation (FMPE) — decisions and findings

First state-of-the-art method compared against the replicated baselines.
Dax, Wildberger, Buchholz, Green, Macke, Schölkopf (2023), *Flow Matching for
Scalable Simulation-Based Inference*, NeurIPS 2023, arXiv:2305.17161, built on
the conditional flow matching objective of Lipman et al. (2023), arXiv:2210.02747.

A plain network v(t, x, y) is trained as a velocity field. Integrating
dx/dt = v(t, x, y\*) from x(0) ~ N(0, I) to t = 1 gives a posterior sample. The
network has no invertibility constraint, and sampling costs one network pass per
ODE function evaluation (NFE) instead of one pass in total.

## Files

| Path | Purpose |
|---|---|
| `models/fmpe.py` | Hand-rolled FMPE: gated residual MLP, CFM loss, fixed-step Euler / midpoint / RK4 sampler |
| `tests/test_fmpe.py` | Parameter count, time prior, solver convergence order, sample interface, a conditional-Gaussian recovery test |
| `experiments/train_fmpe.py` | Training; same data, seeds and schedule as `train_mdn.py` |
| `experiments/evaluate_fmpe.py` | Scores a run with `metrics.evaluate` at one or more `method:steps` settings, adds NFE and parameter count to the result JSON |
| `experiments/bayesflow_fmpe.py` | Cross-check with BayesFlow's `FlowMatching`, same data / budget / schedule, same metrics |
| `experiments/sweep_fmpe.sh` | BayesFlow training, then the NFE sweep of both implementations, one evaluation at a time on CPU |
| `experiments/plot_fmpe_posterior.py` | Sanity plot: samples for a few y\* against the rejection ground truth |
| `experiments/summarize_fmpe.py` | Markdown table from `metrics/results/*.json` |

```bash
.venv/bin/pip install -r requirements.txt   # adds bayesflow (Keras 3, torch backend)
.venv/bin/python experiments/train_fmpe.py --problem kinematics --device mps --run-name kin_fmpe_s0
.venv/bin/python experiments/train_fmpe.py --problem ballistics --device mps --run-name bal_fmpe_s0
experiments/sweep_fmpe.sh
```

## Method, checked against the paper and the authors' code

Checked against the arXiv HTML of the paper and against `dingo`
(github.com/dingo-gw/dingo, `core/posterior_models/flow_matching.py`,
`cflow_base.py`), the authors' code base:

- **Path (Eq. 5–6), optimal-transport path.** x_t = (1 − (1 − σ_min) t) x0 + t x1,
  target u = x1 − (1 − σ_min) x0. t = 0 is the Gaussian, t = 1 the posterior.
  Identical to dingo's `ot_conditional_flow`.
- **Loss.** ‖v(t, x_t, y) − u‖², summed over the 4 x-dimensions (dingo and
  BayesFlow average over dimensions instead, a factor 4 on the reported loss only).
- **Time prior (Sec. 3.3).** dingo draws t = u^(1/(1+α)), u ~ U(0, 1), i.e.
  density p(t) ∝ t^α: uniform at α = 0, weighted towards the posterior end for
  α > 0. See Finding 1 for the paper's formula and BayesFlow's version.
- **Network (Sec. 3.2, 4).** Residual MLP; the paper conditions on (t, θ) through
  gated linear units and reports this beats concatenating (t, θ, x). Our
  network does both: the input layer sees (t, x_t, y) concatenated and every
  residual block is gated by sigmoid(W [t, x_t]).
- The paper's SBI-benchmark appendix gives only sweep ranges (α ∈ {−0.5, −0.25,
  0, 1, 4}, 10–18 blocks, width 16–1024), not the chosen values, and no σ_min or
  solver for that benchmark (its GW experiment uses dopri5 at 1e-7 tolerance).

## Build vs. reuse: both, hand-rolled is the headline

The spec asked for BayesFlow first. We ran both, at the same ~3M parameters,
data and schedule, and scored both with `metrics.evaluate`:

- **Hand-rolled (`models/fmpe.py`) is the one to report.** It gives exact control
  over the parameter budget and the solver (so NFE is known exactly) and runs in
  the same plain-PyTorch stack as the MDN and autoencoder, so inference times
  compare like for like.
- **BayesFlow (`bayesflow 2.0.14`, `FlowMatching` + default `TimeMLP`)** is the
  library's default recipe, only the width changed to fit 3M (409 × 5 blocks,
  2,994,309 parameters). It is the independent cross-check that our numbers
  are not an artefact of our implementation.

BayesFlow did not get in the way of the fixed budget: the width is a constructor
argument and `bayesflow_fmpe.py` solves it to ≤ 3M. Practical costs:
Keras training was ~3–5× slower per step than our PyTorch loop on the same
MPS device; `bayesflow` pulls in Keras 3, pandas, seaborn, rich; and Keras picks
its torch device (MPS) at import time, so timing on CPU needs
`KERAS_TORCH_DEVICE=cpu` (the script's `--device` sets it).

## Findings

1. **Time prior: the paper's formula, dingo and BayesFlow disagree.**
   - The paper prints p_α(t) ∝ t^(1/(1+α)) and says α = 0 is uniform. That
     density is t at α = 0, not uniform, so the formula as printed is not what
     the text means.
   - dingo samples t = u^(1/(1+α)), density ∝ t^α. This is uniform at α = 0 and
     favours the posterior end t = 1 for α > 0, as the text says. The printed
     exponent is the sampling exponent, not the density.
   - BayesFlow's `FlowMatching` samples t = u^(1+α) (its comment repeats the
     paper's formula), density ∝ t^(−α/(1+α)). For α > 0 this favours the
     **noise** end t = 0, the opposite of the paper and dingo, and BayesFlow's
     default is α = 0.5 (E[t] = 0.40 instead of 0.5).
   - We use dingo's convention with α = 0 (uniform), where all three agree. The
     BayesFlow cross-check keeps its library default α = 0.5. Worth reporting
     upstream to BayesFlow, if confirmed by the team.
2. **The ODE solver matters more than NFE, and RK4 is the wrong default.** At
   equal NFE, midpoint and Euler beat RK4 by a wide margin (table below; e.g.
   kinematics at 32 NFE: midpoint Err_post 0.0016 / Err_resim 2.7e-6, RK4 0.0025 /
   1.2e-3). BayesFlow shows the same: its fixed-step `rk45` loses to its Euler at
   equal NFE on both benchmarks, so this is not an artefact of our code. RK4 only
   catches up at 64–256 NFE. Our first hypothesis was RK4's last stage, which
   evaluates the field at t = 1 where the OT field (E[x1 | x_t] − x)/(1 − t) is
   singular. `experiments/fmpe_solver_diagnostic.py` caps the network's time at
   t_max < 1: no effect on kinematics, about half the gap closed on ballistics. So
   that is at most part of the story. A plausible remainder, not verified: for
   this path an Euler step that ends at t = 1 lands exactly on the network's
   estimate of E[x1 | x_t], which favours low-order steps near the data end.
3. **Midpoint reaches the converged accuracy at 8–32 NFE.** Against a 256-NFE RK4
   reference (Err_post 0.0016 / 0.0071), midpoint × 4 (8 NFE) is already at
   0.0017 on kinematics, and midpoint × 16 (32 NFE) at 0.0071 on ballistics.
   BayesFlow's default solver (adaptive tsit5) took **331 NFE** and 1.7 h of CPU
   for the same kinematics accuracy (0.0016), 15× the cost of midpoint × 16.
4. **Err_resim alone rewards collapse; read it with Err_post.** Euler × 8 has the
   smallest kinematics Err_resim of the sweep's cheap settings (1.8e-5) but a 5×
   worse Err_post (0.0086): coarse Euler steps shrink the spread along the
   posterior, and a collapsed posterior still lands on y\*. FMPE's Err_resim at
   good settings (1e-6 – 5e-5) is also below the ground truth's own floor (2e-4 /
   4e-4, from the σ of the rejection likelihood), so below about 1e-4 the metric no
   longer ranks posteriors, only how exactly samples sit on f⁻¹(y\*).
5. **Ballistics: FMPE ties the MDN on Err_post, also beneath the v0 floor.**
   Both score 0.0071. Rounding v0 to integers (the diagnostic of
   `docs/metrics.md`) gives 0.0012 for both, so the tie is not just the shared
   integer-v0 floor. FMPE, like every continuous model, smooths the integer v0
   (`experiments/figures/fmpe_posterior_bal_fmpe_s0.png`, x4 panel).
6. **The price is inference time: ~1400× the MDN at 8 NFE.** The MDN runs its 3M
   network once per y\* and then samples a Gaussian mixture; FMPE runs its 3M
   network once per sample per NFE (1M states × NFE). On our CPU one NFE over the
   whole test set costs ~12 s.

## Our choices (the paper does not fix these for our problems)

| Choice | Value | Notes |
|---|---|---|
| Network | input Linear(t, x_t, y → 497), 6 gated residual blocks (2 × Linear 497 + GLU gate on (t, x_t)), SiLU, Linear(497 → 4) | 2,993,932 params (kinematics), 2,993,435 (ballistics) |
| Output init | zero weights and bias | initial field v = 0 |
| σ_min | 1e-4 | |
| Time prior | α = 0 (uniform) | sweep α later |
| Normalization | x and y standardized with training-set statistics | as MDN |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5, cosine, grad-clip 10 | as MDN |
| Data / schedule | 1M train, 20k val, seeds 10000+s / 20000+s; 50 epochs, batch 1000 | as MDN; **team placeholder** |
| Solver | fixed-step Euler / midpoint / RK4, NFE = stages × steps | headline at 32 NFE (RK4 × 8) |

Training took 32 min (kinematics, MPS) and 67 min (ballistics, CPU, sharing the
machine with another run). Final train / val CFM loss: 3.302 / 3.310 and
4.691 / 4.680.

## Results (seed 0, shared metrics)

All rows: `metrics.evaluate` (`emine-metrics`), the same 1000 test y\* and
cached rejection ground truth, 1000 samples per y\*, unbiased MMD² (biased in
brackets). Times are 1000 y\* × 1000 samples on the CPU of an otherwise idle
Apple M4, measured in one session, so they compare with each other; the MDN was
re-scored in the same session for this reason. Full sweep:
`.venv/bin/python experiments/summarize_fmpe.py`.

### Headline: FMPE at 32 NFE (midpoint × 16) vs. the baselines

| Benchmark | Model | Params | NFE | Err_post | Err_resim | Inference |
|---|---|---|---|---|---|---|
| Kinematics | MDN K=16 | 2.995M | 1 | 0.0031 (0.0077) | 6.7e-4 | 0.07 s |
| | Autoencoder (`emine-autoencoder`, PR 3) | ≤3M | 1 | 0.0053 | 4.1e-4 | 7.9 s ¹ |
| | **FMPE** | 2.994M | 32 | **0.0016** (0.0062) | **2.7e-6** | 414 s |
| | FMPE, cheapest good setting (midpoint × 4) | 2.994M | 8 | 0.0017 (0.0063) | 4.8e-5 | 101 s |
| Ballistics | MDN K=16 | 2.995M | 1 | 0.0071 (0.0127) | 1.2e-3 | 0.07 s |
| | Autoencoder (`emine-autoencoder`, PR 3) | ≤3M | 1 | 0.0081 | 2.4e-3 | 8.0 s ¹ |
| | **FMPE** | 2.993M | 32 | **0.0071** (0.0127) | **4.5e-5** | 374 s |

¹ From PR 3's result files, measured in a different session; not strictly
comparable in time.

Prior-only baseline for scale: Err_post 0.242 / 0.105, Err_resim 2.7 / 20.
No ballistics FMPE sample at ≥ 8 NFE midpoint/Euler was without a ground impact.

**Answer to the spec's question.** On kinematics a continuous flow with no
invertibility constraint beats the MDN on posterior accuracy (Err_post 0.0016 vs
0.0031, i.e. about half; the biased estimator shrinks the gap to 0.0062 vs
0.0077) and puts samples on f⁻¹(y\*) far more exactly. On ballistics it ties the
MDN on Err_post and wins on Err_resim. In both cases it costs three orders of
magnitude more inference time at matched parameters. The coupling-flow
baselines (INN, cINN) are not trained yet, so the comparison against them is
still open.

### NFE sweep (CPU times)

| Solver | NFE | Kinematics Err_post | Err_resim | Ballistics Err_post | Err_resim | Time (kin / bal) |
|---|---|---|---|---|---|---|
| Euler × 8 | 8 | 0.0086 | 1.8e-5 | 0.0123 | 5.8e-4 | 92 / 91 s |
| midpoint × 4 | 8 | 0.0017 | 4.8e-5 | 0.0077 | 4.9e-3 | 101 / 90 s |
| RK4 × 2 | 8 | 0.0888 | 6.3e-2 | 0.1858 | 7.5 | 103 / 89 s |
| RK4 × 4 | 16 | 0.0199 | 6.9e-3 | 0.0444 | 4.9e-1 | 206 / 185 s |
| Euler × 32 | 32 | 0.0022 | 1.4e-6 | 0.0074 | 1.9e-5 | 413 / 376 s |
| midpoint × 16 | 32 | 0.0016 | 2.7e-6 | 0.0071 | 4.5e-5 | 414 / 374 s |
| RK4 × 8 | 32 | 0.0025 | 1.2e-3 | 0.0100 | 2.2e-2 | 412 / 376 s |
| RK4 × 16 | 64 | 0.0016 | 2.0e-4 | 0.0072 | 1.6e-3 | 744 / 734 s |
| RK4 × 64 (MPS, reference) | 256 | 0.0016 | 1.1e-5 | 0.0071 | 8.8e-5 | not comparable |

### BayesFlow cross-check (same data, budget, schedule; library defaults, α = 0.5)

| Solver | NFE | Kinematics Err_post | Err_resim | Ballistics Err_post | Err_resim | Time (kin / bal) |
|---|---|---|---|---|---|---|
| Euler × 8 | 8 | 0.0082 | 6.2e-5 | 0.0118 | 9.7e-4 | 127 / 127 s |
| rk45 × 2 (fixed) | 12 | 0.0133 | 5.8e-2 | 0.0287 | 6.9e-1 | 195 / 192 s |
| rk45 × 4 (fixed) | 24 | 0.0039 | 1.3e-2 | 0.0136 | 1.3e-1 | 391 / 387 s |
| Euler × 32 | 32 | 0.0021 | 1.3e-5 | 0.0073 | 1.9e-4 | 523 / 518 s |
| tsit5, adaptive (library default) | 331 ² | 0.0016 | 1.8e-5 | not run ³ | | 6159 s / – |

² Counted by wrapping `FlowMatching.velocity` on 100 y\* × 1 sample; the
adaptive step count may differ slightly on the full batch.
³ Skipped: it would cost another ~2 h of CPU and the kinematics row already
makes the point.

BayesFlow lands at the same accuracy as our implementation at matched solver
and NFE (Euler × 32: 0.0021 vs 0.0022 on kinematics, 0.0073 vs 0.0074 on
ballistics), so the headline numbers are a property of the method, not of our
code. It is ~1.3× slower per NFE on the CPU (Keras overhead, `TimeMLP` with layer
norm + FiLM).

### Sanity plots

`experiments/figures/fmpe_posterior_{kin,bal}_fmpe_s0.png`: three test y\*, arms /
trajectories of FMPE samples and the 1-D marginals against the rejection ground
truth (RK4 × 8). The marginals match. The visible deviation is v0 on ballistics
(integer ground truth, continuous model), and on kinematics y\* #2, near the edge
of the reachable set, a few end points smear along the arc.

### Solver diagnostic (`experiments/fmpe_solver_diagnostic.py`, first 100 y\*)

| | t_max | Kin Err_post | Kin Err_resim | Bal Err_post | Bal Err_resim |
|---|---|---|---|---|---|
| midpoint × 16 | 1 | 0.0013 | 1.9e-6 | 0.0072 | 3.9e-5 |
| RK4 × 8 | 1 | 0.0021 | 1.2e-3 | 0.0106 | 2.3e-2 |
| RK4 × 8 | 0.9375 | 0.0020 | 1.5e-3 | 0.0074 | 1.1e-2 |
| RK4 × 8 | 0.99 | 0.0016 | 1.6e-3 | 0.0078 | 1.1e-2 |
| RK4 × 16 | 1 | 0.0013 | 2.0e-4 | 0.0073 | 1.6e-3 |

Median |v| just off the posterior (samples + 0.01 noise, standardized units) grows
from 0.3 at t = 0.5 to 4.6 (kin) / 2.9 (bal) at t = 1: large, but not the
1/σ_min blow-up the exact field would have there.

## Open items for the team / supervisor

- **Parameter budget for SOTA methods.** Here FMPE uses the baselines' ~3M and
  their schedule. Ask Paul: "SOTA at matched 3M params, or best-effort?" At
  matched size FMPE already wins on kinematics, so a best-effort run mainly
  matters for ballistics.
- **Report inference cost per NFE or wall-clock?** The ~1400× gap is the honest
  wall-clock at matched parameters. A smaller FMPE network may close much of it
  without losing accuracy; that is the next sweep (model size × NFE).
- **Solver choice** goes into the comparison: midpoint (or Euler) at 8–32 NFE,
  not RK4 and not BayesFlow's adaptive default. Worth agreeing before anyone
  reports a flow-matching number.
- **Time prior α.** Only α = 0 was run. The paper sweeps α ∈ {−0.5 … 4}; also
  confirm the BayesFlow sign issue (Finding 1) and report it upstream.
- **Exact density / calibration.** FMPE can give log p(x | y\*) via the divergence
  of v (not implemented). Decide whether calibration metrics are wanted.
- **Seeds.** Single seed (0) so far, like the MDN.
- Everything the MDN doc lists as a placeholder (data size, epochs, batch)
  applies here too.
