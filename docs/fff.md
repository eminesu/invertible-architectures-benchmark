# Conditional Free-Form Flows (FFF / M-FFF): decisions and findings

Paul's suggestion: Free-Form Flows, and the manifold version (M-FFF) for the
periodic joint angles of the kinematics benchmark. Both are from the Köthe /
VLL group, like the reference paper.

- [FFF] Draxler et al. (2024), *Free-form Flows: Make Any Architecture a
  Normalizing Flow*, AISTATS, arXiv:2310.16624: the method, the trace-estimator
  gradient, and the conditional version for SBI.
- [M-FFF] Sorrenson et al. (2024), *Learning Distributions on Manifolds with
  Free-Form Flows*, NeurIPS, arXiv:2312.09852: projection, tangent-space
  log-det, tori. **Unconditional only.**

**The conditional manifold model is our own combination of the two papers.**
M-FFF has no conditional version, so flag that to Paul as new work.

## Files

| Path | Purpose |
|---|---|
| `models/fff.py` | `ConditionalFFF`: encoder + decoder, FFF loss, torus x R^k product manifold, exact `log_prob`, one-pass `sample(y_star, n)` |
| `experiments/train_fff.py` | Training on the same data / schedule as FMPE; per-epoch checkpoint with resume |
| `experiments/evaluate_fff.py` | `metrics.evaluate`, plus the reconstruction error on the GT samples |
| `experiments/plot_fff_posterior.py` | GT vs. FFF arms / trajectories and marginals, one row per run (the conditional-FMPE figure with rows = model; that figure's helpers are copied in, so this branch does not depend on the unmerged `emine-conditional-fmpe`) |
| `experiments/plot_fff_learning_curve.py` | Training objective, exact NLL and reconstruction error per epoch, train vs. val |
| `tests/test_fff.py` | Shapes / parameter count, embedding round trip, identity at init, density normalizes on R and on the circle (both latents), torus recovery across the ±π cut |

```bash
.venv/bin/python experiments/train_fff.py --problem kinematics --device mps --run-name kin_fff_s0
.venv/bin/python experiments/train_fff.py --problem kinematics --manifold torus --latent wrapped --beta 400 --device mps --run-name kin_mfff_wrapped_b400_s0
.venv/bin/python experiments/train_fff.py --problem kinematics --manifold torus --device cpu --run-name kin_mfff_s0
.venv/bin/python experiments/train_fff.py --problem ballistics --device mps --run-name bal_fff_s0
.venv/bin/python experiments/evaluate_fff.py kin_fff_s0 kin_mfff_s0 kin_mfff_wrapped_b400_s0 bal_fff_s0
.venv/bin/python experiments/plot_fff_posterior.py kin_fff_s0 kin_mfff_s0 kin_mfff_wrapped_b400_s0 --index 707 695 555
.venv/bin/python experiments/plot_fff_posterior.py bal_fff_s0 --index 200 346 966
.venv/bin/python experiments/plot_fff_learning_curve.py kin_fff_s0 kin_mfff_s0 kin_mfff_wrapped_b400_s0 --out experiments/figures/fff_learning_curve_kin.png
.venv/bin/python experiments/plot_fff_learning_curve.py bal_fff_s0
```

## Method

- **Networks.** The encoder f(x, y) and the decoder g(z, y) are residual MLPs:
  6 blocks of 2 linear layers each, SiLU, a global skip connection and a
  zero-initialized output layer, so both start as the identity. The
  standardized y is concatenated to each network's input; this is the FFF
  paper's SBI recipe. Width 352 puts encoder + decoder at **2.99M parameters**,
  inside the 3M budget, with half of that in each network. Only the decoder
  (1.49M) runs at sampling.
- **Loss.** Per sample (FFF Alg. 1, as in the authors' `fff.loss`):

  `-log p(z) - vᵀ f'(x) SG(g'(z) v) + β‖x - g(f(x))‖²`

  g'(z)v is a forward-mode JVP through the decoder; vᵀf'(x) is a backward-mode
  VJP through the encoder. There is one Hutchinson vector per sample, rescaled
  to norm √dim. The surrogate's **value** is ≈ tr(I), not the log-det; only its
  gradient is right. So the log also records the **exact** NLL from the
  encoder's change of variables, on val and on a fixed 20k train subset.
- **Sampling.** z ~ p(z), x = g(z, y*): **1 NFE**, with no ODE.
- **Torus (`--manifold torus`).** Each kinematics angle is embedded as
  (cos φ, sin φ), and the rail offset stays Euclidean. That makes the space
  T³ × R, embedded in R⁷. Both networks are projected onto the manifold, so
  each (cos, sin) pair is normalized. The Hutchinson vector is drawn in the
  tangent space at z. The exact log-det is that of Rᵀ f'(u) Q, where Q and R
  are orthonormal tangent bases. The embedding is an isometry of the unit
  circle, so M-FFF's metric term vanishes, as expected for the torus.
  `test_density_normalizes` checks this: the density integrates to 1.000 on
  the circle. Angles come back via atan2 in (-π, π].
- **Latent on the circles (`--latent`).** `uniform` is the M-FFF choice for
  compact manifolds. `wrapped` is a wrapped normal with σ equal to that angle's
  training std (0.5). With `wrapped`, the torus model at initialization has
  exactly the Euclidean FFF's distribution. Comparing Euclidean against
  wrapped therefore isolates the geometry, and comparing uniform against
  wrapped isolates the latent.
- **Not used:** M-FFF's extra latent-reconstruction (`--beta-z`) and
  projection-distance (`--beta-p`) regularizers. They are implemented but
  left at 0.
- **Schedule.** Identical to FMPE and MDN: 1M train / 20k val (seeds 10000 /
  20000), 50 epochs, batch 1000, Adam 1e-3, weight decay 1e-5, cosine, grad
  clip 10.
- **MPS.** PyTorch has no forward-mode AD rule for `linear` on MPS. The model's
  `Linear` therefore computes `addmm` instead, with the same parameters and
  result.

### Decision 1: y-noise in training (an FFF-specific data change; read before comparing)

Both forward processes are deterministic. With y = f(x), p(x | y) is
concentrated on a lower-dimensional set: 2-D in R⁴ for kinematics, 3-D for
ballistics. Its likelihood is then unbounded. Maximum-likelihood FFF exploits
this. The kinematics pilot below shows it at β = 100: the loss fell to −19
while reconstruction degraded. The fix used here is to train on
**y + σε, with σ the ground-truth likelihood's own σ** (0.01 kinematics /
0.02 ballistics, `metrics/ground_truth.py`), resampled every batch. The GT
defines the posterior under exactly this likelihood, so FFF fits the same
target the GT samples. `--y-noise` defaults to that σ.

**Confound:** FMPE, MDN and NPSE were trained on noiseless y. Part of FFF's
kinematics advantage below may come from fitting the σ-blurred GT exactly,
for example its Err_resim of 2e-4, which equals the GT's own 2σ². The fair
control is FMPE trained with the same y-noise. That needs a flag in
`train_fmpe.py`, which this branch does not touch (see open items).

### Decision 2: β (reconstruction weight)

Pilot: kinematics, Euclidean, 200k samples × 10 epochs (2000 steps).
Err_post / Err_resim are on the full 1000 y*; recon is val.

| β | y-noise | val recon | val exact NLL | Err_post | Err_resim |
|---|---|---|---|---|---|
| 10 | none | 2.9e-1 | -2.64 | 0.049 | 0.016 |
| 100 | none | 1.3e-2 | -3.33 (loss -19: exploiting the degenerate likelihood) | 0.100 | 0.049 |
| 1000 | none | 5.4e-6 | -2.42 | 0.111 | 0.096 |
| **100** | **σ_GT** | **1.1e-3** | **-4.79** | **0.021** | **0.009** |
| 1000 | σ_GT | 7.0e-6 | -2.52 | 0.113 | 0.098 |

β = 100 with y-noise is the only setting that both inverts and learns within
the pilot budget. β = 1000 keeps g = f⁻¹ exactly but learns much more slowly.
Over the full run, reconstruction falls to 3.7e-5 (val; 8e-5 on the GT
samples). The likelihood is valid there, but it was not near zero early in
training (1e-3 to 1e-2 in epochs 1–20).

For the torus, reconstruction is measured on the unit circle. A given angular
error therefore costs about 4× less there than in the Euclidean model, where
angles are standardized by σ = 0.5. **β = 400 on the torus matches β = 100
Euclidean.** `kin_mfff_s0` ran at β = 100, before this was noticed, so it is
effectively under-weighted. The uniform-latent run at β = 400 was stopped at
epoch 12, with val NLL between 0.7 and 3.3 and no trend toward the Euclidean
value. Whatever its β, the uniform latent did not train as well.

## Results (seed 0, 1000 y* × 1000 samples, CPU)

| Run | Space / latent | Err_post mean | median [q1, q3] | Err_resim mean | median | Recon (GT) | Time for 10⁶ samples |
|---|---|---|---|---|---|---|---|
| `kin_fff_s0` | Euclidean | **0.0002** | 0.0000 [-0.0005, 0.0007] | 2.3e-4 | 2.0e-4 | 8.4e-5 | 4.9 s |
| `kin_mfff_wrapped_b400_s0` | T³×R, wrapped normal | 0.0006 | 0.0001 [-0.0004, 0.0010] | 3.8e-4 | 2.1e-4 | 2.0e-5 | 5.1 s |
| `kin_mfff_s0` | T³×R, uniform (M-FFF), β = 100 | 0.0360 | 0.0246 [0.0185, 0.0403] | 1.0e-2 | 2.4e-3 | 1.2e-3 | 5.1 s |
| `bal_fff_s0` | Euclidean | **0.0074** | 0.0067 [0.0039, 0.0100] | 9.0e-4 | 6.3e-4 | 1.8e-4 | 5.1 s |

The same metrics for the reference models (`metrics/results`, CPU):

| | Kinematics Err_post | Kinematics Err_resim | Ballistics Err_post | Ballistics Err_resim | Time for 10⁶ samples |
|---|---|---|---|---|---|
| FMPE, midpoint × 16 (32 NFE) | 0.0016 | 2.7e-6 | 0.0071 | 4.5e-5 | ~410 s |
| NPSE, euler × 32 | 0.0018 | 6.2e-5 | 0.0071 | 4.4e-4 | ~600 s |
| **FFF (Euclidean), 1 NFE** | **0.0002** | 2.3e-4 | 0.0074 | 9.0e-4 | **~5 s** |

Ground-truth self-resimulation is 2σ² = 2e-4 for kinematics and σ² = 4e-4 for
ballistics. FFF's Err_resim sits at that level by construction (Decision 1).
FMPE's is lower because it learns the noiseless posterior. No FFF kinematics
sample has an undefined f(x); 6e-6 of the ballistics samples do.

### Finding 1: flow-quality posteriors at one decoder pass

FFF samples 10⁶ posterior draws in about 5 s on CPU, against about 410 s for
FMPE at 32 NFE: **about 80× faster**. That puts it in MDN territory (one
network pass). Ballistics accuracy is on par with FMPE and NPSE (0.0074 vs.
0.0071). On kinematics the mean Err_post is 8× lower than FMPE's and the
median is 0.0000, so the unbiased MMD² cannot tell the samples apart from the
GT. Read this with the y-noise confound above.

### Finding 2: the torus helps only with a matched latent, and not beyond Euclidean here

- **Uniform latent (as in M-FFF): clearly worse.** Err_post is 0.036, and the
  val NLL got worse after epoch 1 (+0.8 → +7.8 → +2.9 final, against −5.2 for
  Euclidean). Reconstruction stayed at 1e-3 to 1e-2, so the trace estimator's
  g ≈ f⁻¹ assumption never held well. The posterior figure (y* #707) shows
  long angle tails, only 67 % of samples in the valid region, and resim 6.6e-2.
  The likely mechanism: the encoder has to stretch each narrow conditional
  angle posterior (width ~0.1–1 rad, with a ~0.01 thin direction) over the
  whole circle. That map is steep and hard to invert. β = 400 did not fix it
  within 12 epochs.
- **Wrapped-normal latent: on par with Euclidean.** Err_post 0.0006 vs. 0.0002,
  medians 0.0001 vs. 0.0000, final val NLL −5.175 vs. −5.202. The learning
  curves overlap after epoch ~35.
- **Why the geometry cannot help here.** The kinematics prior puts the angles
  at N(0, 0.5²). The GT posteriors never leave (−2.8, 2.6) rad: only 1e-6 of
  GT samples have |φ| > 2.5, and none come near ±π. Periodicity is real in the
  forward model but never active in the data, so there is nothing for the torus
  to fix. The torus machinery itself works: `test_torus_learns_distribution_across_the_cut`
  recovers a distribution straddling ±π that a Euclidean model would split in
  two. A benchmark variant with a wide or uniform angle prior would be the
  place where M-FFF should win.

### Ballistics torus (optional step 6): not run

The launch angle is bounded to 9°–81° by the prior: a fifth of the circle,
never near the wrap. A torus would add only the latent problem of Finding 2.

## Figures

- `experiments/figures/fff_posterior_kin_fff_s0_kin_mfff_s0_kin_mfff_wrapped_b400_s0_y{707,695,555}.png`:
  kinematics y* = (0.50, 1.42) near and folded, (1.53, −0.09) mid, and
  (1.95, −0.02) far, at almost full extension. Rows: Euclidean FFF, torus
  (uniform), torus (wrapped). Same layout as the conditional-FMPE figure: GT
  gray, model blue, y* black +, valid region green. Kinematics valid region:
  rail |x0| ≤ 0.75 and |φᵢ| ≤ 1.5 (3σ of the prior); the arm panel shades the
  rail segment. No separate GT row: every row already overlays the GT.
- `experiments/figures/fff_posterior_bal_fff_s0_y{200,346,966}.png`: ballistics
  y* = 3, 8, 13, the conditional-FMPE indices, with its λ = 15 valid region.
  One row (Euclidean).
- `experiments/figures/fff_learning_curve_kin.png` and
  `fff_learning_curve_bal_fff_s0.png`, three panels:
  - The training objective, train vs. val. It is flat and noisy by
    construction (the surrogate's value), so the val spikes are Hutchinson
    noise and not divergence.
  - The exact NLL. This is the real overfitting check. Train vs. val is only
    available for the wrapped run; `train_nll` logging was added after the
    other runs.
  - The reconstruction error (log scale).

  Euclidean and wrapped decrease and converge, with no train/val gap (wrapped:
  −5.161 / −5.175). The loss values are not comparable across models, or
  across Euclidean vs. torus runs.

## Open items for the team / supervisor

- **y-noise control.** Train FMPE with the same `y + σ_GT ε` (a flag in
  `train_fmpe.py`) to separate "FFF is better" from "fits the σ-blurred GT".
  This decides whether Finding 1's kinematics gap is real.
- **Staging** was plain conditional FFF first, then torus, as recommended.
  The result says to keep Euclidean FFF as the headline and report the torus
  as an ablation.
- **Conditional M-FFF is new.** Flag to Paul: the manifold paper has no
  conditional version, and on this data its uniform latent fails. The
  wrapped-normal latent is our change.
- **Authors' code.** The torus projection, tangent sampling and log-det here
  are about 60 lines in `models/fff.py`, tested for normalization. The
  authors' released code was not used or checked for compatibility.
- **Seeds.** One seed per configuration so far. Seeds 1–2 for `kin_fff_s0`,
  `kin_mfff_wrapped_b400_s0` and `bal_fff_s0` would give error bars on the
  0.0002 vs. 0.0006 kinematics gap.
- **Train-NLL curves for the Euclidean runs.** They were trained before
  `train_nll` was logged; retrain to get the train curve in panel 2
  (about 1 h each on MPS).
