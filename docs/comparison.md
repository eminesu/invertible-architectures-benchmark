# Diffusion (NPSE) vs Flow Matching (FMPE) — comparison

At 3M parameters and 32 NFE, diffusion and FMPE are statistically tied on both
benchmarks. FMPE is cheaper: it reaches the converged accuracy at 8 NFE, while
diffusion needs 32. Seed 0 only; `experiments/sweep_compare.sh` adds seeds,
model size and the FMPE time prior. Method details: `docs/npse.md`, `docs/fmpe.md`.

- **Kinematics:** both beat the MDN (Err_post 0.0016–0.0018 vs 0.0031, better
  on 84–85% of test y\*). Diffusion ODE Euler × 32 vs FMPE midpoint × 16: paired
  Wilcoxon p = 0.18.
- **Ballistics:** every model, MDN included, scores 0.0071. With the integer-v0
  effect removed, all three are still tied (0.0010–0.0012), so at 3M this
  benchmark does not separate them.
- **Cost:** at 8 NFE FMPE midpoint × 4 scores 0.0017 on kinematics, diffusion
  Euler × 8 scores 0.0048. Per NFE both cost the same.
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

## Caveats

- One training seed; the paired tests measure noise over y\*, not over training.
- The three flows train with different time priors / schedules (Setup).
- All flows at 3M parameters; ~6000–8000× the MDN's inference time.
- Times from different sessions; hand-rolled FMPE is ~1.3× faster per NFE than
  BayesFlow (library overhead, not the method).
- Err_resim below ~2e-4 (kinematics) / 4e-4 (ballistics) is under the ground
  truth's own floor and no longer ranks posteriors.

## Planned sweep (`experiments/sweep_compare.sh`)

| Axis | Methods | Settings | New trainings | Evaluated at (NFE) |
|---|---|---|---|---|
| Model size | diffusion, FMPE hand-rolled | 300k, 1M params, seed 0 | 8 | 8, 16, 32, 64 |
| Seeds | diffusion, FMPE hand-rolled | seeds 1, 2 at 3M | 8 | 8, 32 |
| FMPE time prior | FMPE hand-rolled | α = 0, 4 at 3M, seed 0 | 4 | Euler and midpoint at 8, 16, 32 |
| NFE grid | existing 3M seed-0 runs | diffusion Euler × 4; FMPE Euler × 4, 16, 64, midpoint × 2, 8, 32 | 0 | 4, 16, 64 |

FMPE size and seed runs use α = 1, like the headline. Training runs on MPS
(two queues in parallel), then every evaluation runs alone on the CPU.
Estimate: ~7 h training + ~9 h evaluation.
