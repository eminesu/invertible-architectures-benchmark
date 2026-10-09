# Diffusion posterior estimation (NPSE) — decisions and findings

Second state-of-the-art method compared against the replicated baselines, the
diffusion / score-based counterpart of FMPE (`docs/fmpe.md`, branch `emine-fmpe`).
A network learns the score of the noised posterior, ∇ log p_t(x | y), by
denoising score matching; a posterior sample is drawn by starting from noise and
running the reverse process (reverse SDE or probability-flow ODE) conditioned on
y\*. Like FMPE the network has no invertibility constraint, and sampling costs one
network pass per function evaluation (NFE) of the sampler.

## Files

| Path | Purpose |
|---|---|
| `experiments/bayesflow_npse.py` | Training and evaluation with BayesFlow's `DiffusionModel`; same data / budget / schedule as the baselines; scores with `metrics.evaluate` and stores sampler, steps, counted NFE, parameter count, load average |
| `experiments/plot_npse_posterior.py` | Sanity check (build step 4): samples for a few y\* against the rejection ground truth, plus per-y\* numbers |
| `experiments/sweep_npse.sh` | Sampling-step sweep (run) and model-size sweep (not run yet) |
| `experiments/figures/npse_posterior_*.png` | Sanity plots |

```bash
.venv/bin/python experiments/bayesflow_npse.py --problem kinematics --device mps --sampler   # train only
.venv/bin/python experiments/bayesflow_npse.py --problem kinematics --evaluate-only --device cpu --sampler euler:32
.venv/bin/python experiments/plot_npse_posterior.py kin_bf_npse_s0 --sampler euler:32
```

## Build vs. reuse: BayesFlow, as the spec asks

The spec names BayesFlow the primary tool and a hand-rolled model the fallback
"only if the library blocks the fair-budget comparison". It does not block it:
the network width is a constructor argument and `bayesflow_npse.py` solves it to
≤ 3M parameters. So every number below is BayesFlow. This is also the harness of
the FMPE cross-check (`bayesflow_fmpe.py`), so diffusion vs FMPE inside BayesFlow
differs only in the method.

Library class, checked in the installed source (bayesflow 2.0.14, Keras 3.15.1
on the torch backend, torch 2.14.1): `bf.networks.DiffusionModel`, whose
docstring cites Geffner et al. (2023) and the SBI diffusion tutorial of Arruda
et al. (2025, arXiv:2512.20685). `sbi` is not installed and was not used.

## Method, checked against the papers

- **Song et al. (2021), arXiv:2011.13456:** VP SDE dx = −½β(t)x dt + √β(t) dw
  (Eq. 11), reverse SDE (Eq. 6), probability-flow ODE (Eq. 13), denoising score
  matching (Eq. 7) with λ(t) ∝ 1/E‖∇ log p_0t‖², i.e. noise prediction.
- **Geffner et al. (2023), arXiv:2209.14249:** NPSE uses discrete VP-type kernels
  N(√γ_t θ, (1 − γ_t) I) with T = 400 levels (App. B), parameterizes the score
  through the noise (App. B.1) and samples NPSE with the DDPM ancestral sampler
  of Ho et al. (App. B.3). The spec's VE form x_t = x_0 + σ(t)ε is not what
  either paper uses for SBI.
- **BayesFlow's `DiffusionModel` (library defaults, kept):** VP cosine schedule
  on log-SNR ∈ [−12, 12], network predicts the velocity (v-parameterization),
  trained with the noise loss and sigmoid weighting (Kingma et al. 2023). Same
  family as both papers (VP, noise-space loss); the schedule and weighting are
  the library's modern defaults.

## Our choices

| Choice | Value | Notes |
|---|---|---|
| Formulation | VP, cosine log-SNR schedule, v-prediction, noise loss | BayesFlow default; fixed on both benchmarks |
| Network | BayesFlow TimeMLP, 5 residual blocks × width 409, Fourier time embedding, mish | only the width changed from the default (256) to fit the budget |
| Parameters | 2,994,309 (both benchmarks) | budget ≤ 3M, as the baselines; same width as the FMPE cross-check |
| Data / schedule | 1M train, 20k val, seeds 10000+s / 20000+s; 50 epochs, batch 1000; Adam, lr 1e-3, cosine, weight decay 1e-5, clipnorm 10 | as MDN / FMPE; **team placeholder** |
| Standardization | `standardize='all'` | x and y |
| Samplers | probability-flow ODE: `euler`, fixed-step `rk45`; reverse SDE: `euler_maruyama` | NFE counted by wrapping `DiffusionModel.velocity` |
| Headline sampler | ODE Euler × 32 (32 NFE) | best accuracy per NFE among the tested settings |

Training: 53 min (kinematics) and 51 min (ballistics) on MPS. Final train / val
loss 0.190 / 0.188 (kinematics), 0.273 / 0.272 (ballistics), still decreasing
slowly.

## Sanity check (build step 4)

`experiments/figures/npse_posterior_{kin,bal}_bf_npse_s0_{eulerx32,euler_maruyamax128}.png`,
test y\* #0–2, 1000 samples each. On both benchmarks the re-simulated end points
sit on y\*, the four marginals overlap the rejection ground truth, the
per-dimension std ratio model / ground truth is 0.95–1.07, and no ballistics
sample is without a ground impact. The one visible deviation is v0 on
ballistics: the ground truth is integer-valued (Poisson prior), the model is
continuous and smooths it, exactly as FMPE and the MDN do.

## Results (seed 0, shared metrics)

All rows: `metrics.evaluate` (`emine-metrics`), the same 1000 test y\* and cached
rejection ground truth, 1000 samples per y\*, unbiased MMD² (biased in brackets).
Times are 1000 y\* × 1000 samples on the CPU of an Apple M4 (timing caveats below).
Other models' numbers are read from their committed result files, not re-run:
MDN `metrics/results/*_mdn_K16_s0.json` (main), autoencoder `emine-autoencoder`
@ 707e079, FMPE `emine-fmpe` @ 863002e.

### Headline: diffusion at 32 NFE vs. the other models

| Benchmark | Model | Params | NFE | Err_post | Err_resim | Inference |
|---|---|---|---|---|---|---|
| Kinematics | MDN K=16 | 2.995M | 1 | 0.0031 (0.0077) | 6.7e-4 | 0.07 s |
| | Autoencoder | ≤3M | 1 | 0.0053 (0.0099) | 4.1e-4 | 7.9 s |
| | FMPE, hand-rolled (midpoint × 16) | 2.994M | 32 | 0.0016 (0.0062) | 2.7e-6 | 414 s |
| | FMPE, BayesFlow (Euler × 32) | 2.994M | 32 | 0.0021 (0.0067) | 1.3e-5 | 523 s |
| | **Diffusion, BayesFlow (ODE Euler × 32)** | 2.994M | 32 | **0.0018** (0.0064) | 6.2e-5 | 600 s |
| Ballistics | MDN K=16 | 2.995M | 1 | 0.0071 (0.0127) | 1.2e-3 | 0.07 s |
| | Autoencoder | ≤3M | 1 | 0.0081 (0.0136) | 2.4e-3 | 8.0 s |
| | FMPE, hand-rolled (midpoint × 16) | 2.993M | 32 | 0.0071 (0.0127) | 4.5e-5 | 374 s |
| | FMPE, BayesFlow (Euler × 32) | 2.994M | 32 | 0.0073 | 1.9e-4 | 518 s |
| | **Diffusion, BayesFlow (ODE Euler × 32)** | 2.994M | 32 | **0.0071** (0.0127) | 4.4e-4 | 554 s |

Prior-only baseline for scale: Err_post 0.242 / 0.105, Err_resim 2.7 / 20. The
coupling-flow baselines (INN, cINN) are not trained yet, so the comparison
against them is still open.

**Answer to the spec's question.** On kinematics, score-based diffusion beats
the MDN (Err_post 0.0018 vs 0.0031) and the autoencoder (0.0053), and is on par
with FMPE (0.0016–0.0021). On ballistics every model ties at 0.0071; that value
is the floor every continuous model hits on the integer-valued v0
(`docs/metrics.md`), so this benchmark does not separate them on Err_post. The
price is sampling cost: at matched parameters, ~8000× the MDN's inference time
at 32 NFE; per NFE it costs the same as FMPE (Finding 6).

### Sampling-step sweep (diffusion, both samplers)

| Sampler | NFE | Kin Err_post | Kin Err_resim | Bal Err_post | Bal Err_resim | Time (kin / bal) |
|---|---|---|---|---|---|---|
| ODE Euler × 8 | 8 | 0.0048 | 8.1e-4 | 0.0083 | 7.7e-3 | 124 / 139 s |
| ODE rk45 × 2 (fixed) | 12 | 0.0276 | 1.2e-1 | 0.0186 | 1.0 ¹ | 216 / 218 s |
| ODE Euler × 16 | 16 | 0.0024 | 1.2e-4 | 0.0072 | 9.7e-4 | 249 / 286 s |
| ODE rk45 × 4 (fixed) | 24 | 0.0117 | 2.6e-2 | 0.0097 | 2.2e-1 | 431 / 440 s |
| ODE Euler × 32 | 32 | 0.0018 | 6.2e-5 | 0.0071 | 4.4e-4 | 600 / 554 s |
| SDE Euler–Maruyama × 32 | 32 | 0.0022 | 1.9e-2 | 0.0081 | 1.4e-1 | 578 / 582 s |
| ODE Euler × 64 | 64 | 0.0017 | 5.1e-5 | 0.0071 | 3.8e-4 | 1105 / 1152 s |
| SDE Euler–Maruyama × 128 | 128 | 0.0016 | 7.4e-4 | 0.0071 | 5.4e-3 | 2429 / 2359 s |
| SDE Euler–Maruyama × 256 | 256 | 0.0016 | 1.5e-4 | not run | | 4670 s / – |

¹ 0.55% of ballistics samples had no ground impact (left out of Err_resim).

### Diffusion vs. FMPE at equal NFE

Err_post; FMPE values from the `emine-fmpe` result files. "BayesFlow" columns
use the same library, budget, data and evaluation code for both methods.

| NFE (solver) | Kin: diffusion | Kin: FMPE BayesFlow | Kin: FMPE hand-rolled | Bal: diffusion | Bal: FMPE BayesFlow | Bal: FMPE hand-rolled |
|---|---|---|---|---|---|---|
| 8 (Euler) | **0.0048** | 0.0082 | 0.0086 | **0.0083** | 0.0118 | 0.0123 |
| 8 (FMPE midpoint × 4) | | | 0.0017 | | | 0.0077 |
| 12 (rk45 × 2) | 0.0276 | **0.0133** | | **0.0186** | 0.0287 | |
| 24 (rk45 × 4) | 0.0117 | **0.0039** | | **0.0097** | 0.0136 | |
| 32 (Euler) | **0.0018** | 0.0021 | 0.0022 | **0.0071** | 0.0073 | 0.0074 |
| 32 (FMPE midpoint × 16) | | | 0.0016 | | | 0.0071 |
| adaptive (library default) | not run | 0.0016 (tsit5, 331 NFE) | | | | |

## Findings

1. **Diffusion matches FMPE in accuracy; neither family wins outright.** With
   the same library, budget and Euler solver, diffusion is slightly ahead at 8
   and 32 NFE on both benchmarks. With fixed-step rk45 it is behind on
   kinematics and ahead on ballistics. Both converge to the same Err_post
   (kinematics 0.0016–0.0018, ballistics 0.0071) by 32 NFE.
2. **The cheapest good setting is still FMPE's.** Hand-rolled FMPE with the
   midpoint solver reaches 0.0017 on kinematics at 8 NFE; diffusion needs 32 NFE
   (ODE Euler) for 0.0018. At 8 NFE diffusion is at 0.0048.
3. **Reverse SDE: the weak spot the spec expected.** At equal NFE the SDE is
   never better than the ODE on Err_post, and its Err_resim is ~300× worse on
   both benchmarks at 32 steps: the stochastic sampler leaves noise
   on the samples, so they sit off f⁻¹(y\*). It needs 256 steps on kinematics to
   come back to the ground truth's own floor (2e-4). Use the probability-flow ODE.
4. **Higher-order fixed-step solvers lose at equal NFE**, as for FMPE (Finding
   2 of `docs/fmpe.md`): rk45 × 4 (24 NFE) has 5–7× the Err_post of Euler × 16 /
   × 32 on kinematics and ~1.4× on ballistics.
5. **Err_resim alone would rank the samplers wrongly.** SDE × 128 has the best
   kinematics Err_post (0.0016) but a 12× worse Err_resim than ODE × 32; read
   both metrics together, as for FMPE.
6. **Cost per NFE is the same as FMPE's.** One NFE over the whole test set
   (10⁶ samples) costs 15.5–18.8 s on this CPU; BayesFlow FMPE took 15.9–16.3 s
   in its session. The difference (−3% at 8 NFE, +7–15% at 32 NFE) is within
   the timing noise, so at equal NFE the two methods cost the same; the cost
   question is only how many NFE each needs.

## Timing caveats

- Times are wall-clock on the same Apple M4, CPU, one evaluation at a time,
  no training running, but with the usual desktop apps open (window server,
  browser, Claude app). Seconds per NFE varied between 15.5 and 18.8 s across
  the runs, so read times as ±15%.
- The `load_avg_before` field in the sweep result files is measured right after
  the previous evaluation finished, so it includes that run's tail; it is not a
  clean idle measurement.
- MDN, autoencoder and FMPE times come from their own sessions' result files,
  not re-measured here; comparable only roughly.

## Not done yet / open items

- **Model-size sweep (build step 7, second half).** Prepared in
  `experiments/sweep_npse.sh` (300k and 1M parameters) but not run, to focus on
  the comparison first.
- **Library-default adaptive sampler** (`two_step_adaptive`, ~290 NFE on 100 y\*)
  was stopped before finishing; not reported.
- **FMPE side may change.** `emine-fmpe` @ 863002e notes the FMPE headline runs
  were trained with p(t) ∝ t (dingo α = 1) and adds an α sweep, which was
  stopped. The FMPE numbers above are those committed result files.
- **Coupling baselines (INN, cINN)** not trained yet; the comparison against
  them is open.
- **Hand-rolled fallback not used.** The library did not block the fair budget.
- **sbi** not used (BayesFlow chosen so diffusion and FMPE share one harness);
  agree with the team.
- **Single seed (0)**, like the MDN and FMPE.
- Everything the MDN doc lists as a placeholder (data size, epochs, batch).
