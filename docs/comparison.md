# Diffusion (NPSE) vs Flow Matching (FMPE) — comparison

Across three seeds and three model sizes, FMPE matches or beats diffusion at
every NFE and gets there cheaper. At 32 NFE the gap is small, and a
300k-parameter FMPE is as accurate as the 3M one at a fifth of the cost.
Method details: `docs/npse.md`, `docs/fmpe.md`. Sweep: `experiments/sweep_compare.sh`.

- **Kinematics, 32 NFE:** FMPE midpoint scores 0.0016 on all 3 seeds, diffusion
  ODE Euler 0.0017–0.0019. FMPE is ahead on every seed, by 0.00013–0.00023
  (paired p = 0.045–0.21). Both beat the MDN (0.0031).
- **8 NFE:** FMPE 0.0016–0.0017 against diffusion 0.0043–0.0048 on kinematics,
  0.0077 against 0.0083–0.0085 on ballistics; FMPE wins on all 3 seeds (p < 0.02).
- **Model size:** 300k and 1M parameters are as accurate as 3M for both methods.
  FMPE at 300k and 8 NFE scores 0.0017 in 19 s, against 414 s for the 3M
  headline. Smaller diffusion is even better at low NFE (0.0032 at 300k vs
  0.0048 at 3M, 8 NFE).
- **Ballistics:** every model, size and seed ties at 0.0071 from 16 NFE on; the
  v0 diagnostic shows the tie is real, not only the integer-v0 floor.
- **FMPE time prior:** α = 0 and α = 1 are equivalent; α = 4 is worse
  (kinematics midpoint × 16: 0.0022 vs 0.0016).
- **Sampler matters more than method:** fixed-step RK4 / RK45 and the reverse
  SDE lose to Euler / midpoint at equal NFE for both families.

## Setup

Shared: 1M training pairs, 50 epochs, batch 1000, Adam lr 1e-3 with cosine
schedule; ≤ 3M parameters; `metrics.evaluate` on the same 1000 test y\*, 1000
samples each.

| Run | Library | Network | Params (kin) | Time prior / schedule |
|---|---|---|---|---|
| Diffusion (NPSE) | BayesFlow 2.0.14 `DiffusionModel` | TimeMLP 5 × 409 | 2,994,309 | VP cosine log-SNR, v-prediction (library defaults) |
| FMPE, BayesFlow | BayesFlow 2.0.14 `FlowMatching` | TimeMLP 5 × 409 | 2,994,309 | library α = 0.5 (weights the noise end) |
| FMPE, hand-rolled | PyTorch, `models/fmpe.py` | 6 gated residual blocks × 497 | 2,993,932 | p(t) ∝ t (dingo α = 1) |
| MDN K = 16 | PyTorch | MLP | 2.995M | – |

The two BayesFlow rows differ only in the method: the cleanest diffusion-vs-FMPE
pair. The hand-rolled FMPE is the FMPE headline.

## Results (seed 0)

### Headline at 32 NFE

Err_post unbiased (biased); inference = 1000 y\* × 1000 samples on an Apple M4
CPU, measured in different sessions (±15%).

| Benchmark | Model | Sampler | Err_post | Err_resim | Inference |
|---|---|---|---|---|---|
| Kinematics | FMPE hand-rolled | midpoint × 16 | 0.0016 (0.0062) | 2.7e-6 | 414 s |
| | Diffusion | ODE Euler × 32 | 0.0018 (0.0064) | 6.2e-5 | 600 s |
| | FMPE BayesFlow | Euler × 32 | 0.0021 (0.0067) | 1.3e-5 | 523 s |
| | MDN K = 16 | 1 pass | 0.0031 (0.0077) | 6.7e-4 | 0.07 s |
| Ballistics | FMPE hand-rolled | midpoint × 16 | 0.0071 (0.0127) | 4.5e-5 | 374 s |
| | Diffusion | ODE Euler × 32 | 0.0071 (0.0127) | 4.4e-4 | 554 s |
| | FMPE BayesFlow | Euler × 32 | 0.0073 (0.0129) | 1.9e-4 | 518 s |
| | MDN K = 16 | 1 pass | 0.0071 (0.0127) | 1.2e-3 | 0.07 s |

Prior-only baseline: Err_post 0.242 / 0.105.

### Accuracy against NFE (Err_post)

| NFE | Diffusion ODE Euler | FMPE hand-rolled midpoint | FMPE hand-rolled Euler | FMPE BayesFlow Euler |
|---|---|---|---|---|
| 8 | 0.0048 / 0.0083 | **0.0017 / 0.0077** | 0.0086 / 0.0123 | 0.0082 / 0.0118 |
| 16 | 0.0024 / 0.0072 | | | |
| 32 | 0.0018 / 0.0071 | **0.0016 / 0.0071** | 0.0022 / 0.0074 | 0.0021 / 0.0073 |
| 64 | 0.0017 / 0.0071 | | | |

Kinematics / ballistics. The other samplers (RK4, fixed-step RK45, reverse SDE)
are in `docs/npse.md` and `docs/fmpe.md`; all are worse at equal NFE.

### Paired tests on the same 1000 y\*

Per-y\* Err_post difference A − B (negative = A better), bootstrap 95% CI of
the mean, Wilcoxon signed-rank p.

| Benchmark | A vs B | Mean diff [95% CI] | A better on | p |
|---|---|---|---|---|
| Kinematics | Diffusion Euler × 32 vs FMPE midpoint × 16 | +0.00014 [+0.00004, +0.00024] | 49% | 0.18 |
| | Diffusion Euler × 32 vs FMPE BayesFlow Euler × 32 | −0.00035 [−0.00044, −0.00026] | 61% | 5e-16 |
| | Diffusion Euler × 8 vs FMPE midpoint × 4 | +0.00310 [+0.00244, +0.00388] | 26% | 8e-77 |
| | Diffusion Euler × 32 vs MDN | −0.00134 [−0.00152, −0.00116] | 84% | 1e-102 |
| | FMPE midpoint × 16 vs MDN | −0.00147 [−0.00162, −0.00134] | 85% | 8e-108 |
| Ballistics | Diffusion Euler × 32 vs FMPE midpoint × 16 | −0.00001 [−0.00006, +0.00005] | 50% | 0.75 |
| | Diffusion Euler × 32 vs FMPE BayesFlow Euler × 32 | −0.00023 [−0.00030, −0.00016] | 58% | 8e-10 |
| | Diffusion Euler × 8 vs FMPE midpoint × 4 | +0.00058 [+0.00041, +0.00076] | 47% | 3e-4 |
| | Diffusion Euler × 32 vs MDN | −0.00002 [−0.00008, +0.00005] | 48% | 0.90 |

Inside BayesFlow, diffusion beats FMPE with the same Euler solver, but by less
than the gap between FMPE's own solvers. Against FMPE's best solver it is a tie.

## Ballistics: the v0 diagnostic

Ground-truth v0 is integer-valued (Poisson prior); every model outputs
continuous v0 (`docs/metrics.md`). Re-scored on all 1000 y\* with fresh samples
(diffusion ODE Euler × 32, FMPE midpoint × 16; MDN and prior from `docs/mdn.md`):

| Variant | MDN K = 16 | Diffusion | FMPE hand-rolled | Prior |
|---|---|---|---|---|
| Raw x (shared setting) | 0.0071 | 0.0071 | 0.0071 | 0.105 |
| v0 rounded to integers | 0.0012 | 0.00117 | 0.00116 | |
| x standardized by prior std (0.50, 0.50, 0.36, 3.87) | 0.0010 | 0.00101 | 0.00100 | 0.068 |

## Merge check (main @ df239cf)

- `import bayesflow` turns torch autograd off process-wide
  (`torch.set_grad_enabled(False)`). After the merge, pytest imports it via
  `tests/test_npse.py`, which broke `test_fmpe.py::test_learns_conditional_gaussian`;
  fixed with `@torch.enable_grad()`. The training scripts run in separate
  processes, so no reported result was affected. A script that trains the
  hand-rolled FMPE in the same process as BayesFlow needs the same guard.
- Every table row in `docs/npse.md` and `docs/fmpe.md` matches its result JSON.
- `experiments/sweep_alpha.sh` never ran (started from a worktree without the
  FMPE scripts); its runs are folded into `sweep_compare.sh`.

## Sweep results (`experiments/sweep_compare.sh`)

20 trainings and 60 evaluations, 2026-10-09 23:13 to 2026-10-10 13:04, no
failures; every evaluation ran alone on the CPU. FMPE size and seed runs use
α = 1 like the headline. Numbers are read from
`metrics/results/*.json`.

### Model size (seed 0; Err_post, CPU time)

| NFE | Diffusion 300k | Diffusion 1M | Diffusion 3M | FMPE 300k | FMPE 1M | FMPE 3M |
|---|---|---|---|---|---|---|
| Kin 8 | 0.0032, 30 s | 0.0035, 61 s | 0.0048, 124 s | **0.0017, 19 s** | 0.0016, 40 s | 0.0017, 101 s |
| Kin 16 | 0.0020, 61 s | 0.0021, 126 s | 0.0024, 249 s | 0.0015, 39 s | 0.0015, 82 s | 0.0015, 181 s |
| Kin 32 | 0.0017, 127 s | 0.0017, 257 s | 0.0018, 600 s | 0.0016, 80 s | 0.0016, 184 s | 0.0016, 414 s |
| Kin 64 | 0.0017, 259 s | 0.0017, 519 s | 0.0017, 1105 s | 0.0017, 167 s | 0.0017, 366 s | 0.0017, 728 s |
| Bal 8 | 0.0085 | 0.0084 | 0.0083 | 0.0077 | 0.0078 | 0.0077 |
| Bal 16 | 0.0073 | 0.0073 | 0.0072 | 0.0071 | 0.0071 | 0.0071 |
| Bal 32 | 0.0071 | 0.0071 | 0.0071 | 0.0071 | 0.0071 | 0.0071 |

Diffusion is evaluated with ODE Euler × NFE, FMPE with midpoint × NFE/2.
Parameters: 297,944 / 992,165 / 2,994,309 (diffusion), 297,604 / 998,716 /
2,993,932 (FMPE), kinematics.

### Seeds (3M)

Err_post per seed 0 / 1 / 2; paired Wilcoxon p of diffusion against FMPE.

| Benchmark | NFE | Diffusion ODE Euler | FMPE midpoint | FMPE better by | p per seed |
|---|---|---|---|---|---|
| Kinematics | 8 | 0.0048 / 0.0046 / 0.0043 | 0.0017 / 0.0016 / 0.0017 | 0.0026–0.0031 | 8e-77 / 8e-80 / 4e-53 |
| | 32 | 0.0018 / 0.0017 / 0.0019 | 0.0016 / 0.0016 / 0.0016 | 0.00013–0.00023 | 0.18 / 0.21 / 0.045 |
| Ballistics | 8 | 0.0083 / 0.0084 / 0.0085 | 0.0077 / 0.0078 / 0.0077 | 0.0006–0.0008 | 3e-4 / 0.017 / 2e-4 |
| | 32 | 0.0071 / 0.0072 / 0.0071 | 0.0071 / 0.0071 / 0.0071 | ≤ 0.00004 | 0.75 / 0.24 / 0.49 |

### FMPE time prior (3M, seed 0)

| Kinematics Err_post | α = 0 (uniform) | α = 1 (headline) | α = 4 |
|---|---|---|---|
| Euler × 8 | 0.0084 | 0.0086 | 0.0135 |
| Euler × 32 | 0.0021 | 0.0022 | 0.0035 |
| midpoint × 4 (8 NFE) | 0.0016 | 0.0017 | 0.0021 |
| midpoint × 16 (32 NFE) | 0.0016 | 0.0016 | 0.0022 |

Ballistics, midpoint × 16: 0.0071 / 0.0071 / 0.0072. The paper default α = 0
is safe to report.

### NFE grid and cost per NFE

- 4 NFE breaks both methods: diffusion Euler × 4 scores 0.0176, FMPE Euler × 4
  0.0332 and midpoint × 2 0.0105 (kinematics).
- FMPE's best setting is midpoint × 8 (16 NFE): 0.0015 kinematics, 0.0071
  ballistics; more steps do not help.
- Cost per NFE scales with parameters: BayesFlow diffusion 15.5–18.8 s at 3M,
  3.8–4.3 s at 300k; hand-rolled FMPE 10.4–12.9 s at 3M, 2.4–2.8 s at 300k. The
  ~1.5× gap between them is BayesFlow overhead, not the method.

## Caveats

- **Libraries differ.** Diffusion is BayesFlow, the FMPE headline is hand-rolled
  PyTorch. Inside BayesFlow, with the same Euler solver, diffusion is slightly
  ahead of FMPE; the hand-rolled FMPE's lead comes partly from the midpoint
  solver, which BayesFlow does not offer.
- **Diffusion samplers.** Only Euler, fixed-step RK45 and Euler–Maruyama were
  tried; a second-order ODE sampler (Heun / DPM-Solver) would be the fair
  counterpart of FMPE's midpoint.
- **Model size was varied at the 3M schedule** (1M pairs, 50 epochs).
- Timing: one machine, evaluations run alone, ±15% between sessions.
- Err_resim below ~2e-4 (kinematics) / 4e-4 (ballistics) is under the ground
  truth's own floor and no longer ranks posteriors.
- Headline numbers stay at the shared 3M budget for every model. The size
  sweep is a side result on inference cost: the flows' cost comes from network
  size, not the method. Open for the team: which sampler to report (midpoint × 8
  for FMPE, Euler × 32 for diffusion)?
