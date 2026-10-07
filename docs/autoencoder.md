# Standard autoencoder — decisions, status, hand-off

Plain-autoencoder baseline of Kruse et al. (2021), arXiv:2101.10763, Sec. 2
(Eqs. 3, 7–8). Ordinary PyTorch, no FrEIA.

**Status: hand-off point.** Build-order steps 1–4 are done (data, networks,
losses, sanity plot). Steps 5–7 (full training on both benchmarks, evaluation,
sweeps) are for Berker. Every open choice is marked `TODO(berker)` in the code:

```bash
grep -rn "TODO(berker)" models/ experiments/
```

> ⚠️ **Read first — two findings from the first shared-metrics evaluation**
> (details in [First evaluation with the shared metrics](#first-evaluation-with-the-shared-metrics-kinematics-10-epochs)):
>
> 1. **Accuracy is already better than the paper's autoencoder, but worse than our
>    MDN.** Err_post is ~2.6× lower than the paper's AE, but the mean is ~2.6× the
>    median: a few hard y\* in low-density regions carry most of the error.
> 2. **On CPU the autoencoder is NOT the fastest model: it is ~110× slower than
>    the MDN** (8.2 s vs 75 ms). This is expected from the architecture, not a bug.
>    **Inference time must be measured on a GPU**, with every model timed on the
>    same machine and in the same way, before any speed claim is made.

## Files

| Path | Purpose |
|---|---|
| `models/autoencoder.py` | Encoder / decoder MLPs, `(ŷ, z)` split, the three losses, `sample(y_star, n)` |
| `metrics/mmd.py` → `mmd_torch` | Differentiable twin of the shared `mmd`; same kernel object, tested equal (`tests/test_metrics.py`) |
| `experiments/train_autoencoder.py` | Training; writes `experiments/runs/<run>/` (git-ignored), like `train_mdn.py` |
| `experiments/plot_autoencoder_sanity.py` | Step 4: arm plot for fixed y\*, plus encoder-z and reconstruction diagnostics |
| `tests/test_autoencoder.py` | Shapes / split, parameter budget, gradient routing of L_z, `sample` = D(y\*, z) |
| `data/toy_data.py`, `experiments/plot_data.py` | Step 1, shared with the MDN (already on `main`) |

## Model

```
encoder E: x (4) -> (ŷ, z)      dim ŷ = dim y,  dim z = 4 - dim y
decoder D: (y, z) (4) -> x (4)
test time: z ~ N(0, I),  x = D(y*, z)        (decoder only)
```

| Benchmark | dim x | dim y | dim z |
|---|---|---|---|
| Kinematics | 4 | 2 | 2 |
| Ballistics | 4 | 1 | 3 |

The code is exactly as wide as x (no bottleneck), so at zero reconstruction
loss D is the exact inverse of E. In practice reconstruction never reaches
zero ("soft" invertibility); the sanity plot shows the residual per coordinate.

## Losses

`L = L_recon + a·L_y + b·L_z`, all in standardized units (training-set mean / std):

| Term | Formula | Notes |
|---|---|---|
| L_recon | `‖x − D(ŷ, z)‖²` | literal D(E(x)); decoding from the true y instead is a `TODO(berker)` |
| L_y | `‖ŷ − y‖²` | makes the y-slot carry the observation |
| L_z | `MMD([ŷ.detach(), z], [y, ε])`, ε ~ N(0, I) | Eq. 3 |

**L_z is on the joint `(y, z)`, not on z alone.** Eq. 3 needs z ~ N(0, I) *and*
z independent of y, otherwise sampling z from the prior at a fixed y\* is wrong.
A marginal MMD on z only checks the first. The joint version follows the INN
code of Ardizzone et al. (2019). ŷ is detached so the MMD only shapes z
(`test_joint_latent_mmd_does_not_train_y_hat`). `--latent-mmd marginal` is kept
for comparison.

**Kernel: shared with the metrics.** `mmd_torch` takes the same `imq_kernel`
(IMQ, h ∈ {0.05, 0.2, 0.9}) and the same unbiased estimator as Err_post, and a
test checks it equals `metrics.mmd.mmd` to 1e-10. One thing to know: here the
kernel acts on standardized (y, z), where typical distances are about 2–3, while
the bandwidths were chosen for raw x units. The largest bandwidth (0.9) is
still informative, and the probe below shows the term works. Whether wider
bandwidths train better is a `TODO(berker)`, but the metric itself must not
change.

## Choices (the paper does not specify these)

| Choice | Value | Status |
|---|---|---|
| Latent split | dim z = 4 − dim y (code size = dim x) | decided by the "no bottleneck" argument; larger z is a `TODO(berker)` |
| Networks | 2 MLPs, 4 hidden layers each, ReLU, width 705 | width solved for ≤ 3M total → **2,999,078** params; depth / activation `TODO(berker)` |
| Normalization | x and y standardized with training-set statistics | same as MDN |
| a (L_y weight) | 1 | placeholder, `TODO(berker)` |
| b (L_z weight) | 100 | from the coarse probe below, `TODO(berker)` |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5, cosine schedule, grad-clip 10 | copied from MDN, `TODO(berker)` |
| Data / schedule | 1M train, 20k val, batch 1000, 50 epochs; seeds 10 000+s / 20 000+s | **same placeholders and seeds as the MDN**, pending the team's shared schedule |

## Probe of b (not a sweep)

Kinematics, 300k training samples, 5 epochs, a = 1. Val = held-out losses;
resim = mean ‖f(D(y, z)) − y‖² over 200 val y × 100 samples (provisional, not
the shared metric).

| b | val recon | val L_y | val L_z | resim |
|---|---|---|---|---|
| 0 | 0.0012 | 0.0020 | 0.0454 | 0.0927 |
| 10 | 0.0019 | 0.0047 | 0.0001 | 0.0223 |
| **100** | 0.0031 | 0.0071 | −0.0002 | **0.0091** |
| 1000 | 0.0266 | 0.0189 | −0.0002 | 0.0139 |

With b = 0 the model reconstructs best and samples worst: z drifts off the
prior, which is the failure mode the spec predicted. Too large a b costs
reconstruction and L_y. b is a trade-off, so it needs a real sweep with the
shared metrics on the full schedule.

## Sanity check (step 4)

Run `kin_ae_sanity_s0`: kinematics, 1M samples, **10 epochs only** (short on
purpose), defaults otherwise. 7 min on an M4 CPU.

```bash
.venv/bin/python experiments/train_autoencoder.py --problem kinematics --epochs 10 --run-name kin_ae_sanity_s0
.venv/bin/python experiments/plot_autoencoder_sanity.py kin_ae_sanity_s0
```

![sanity](../experiments/figures/autoencoder_sanity_kin_ae_sanity_s0.png)

Final val: recon 0.0041, L_y 0.0016, L_z ≈ 0 (−0.0003, at the noise floor),
provisional resim 0.0033 (still dropping each epoch).

- **End points sit on y\*, and the arms spread over many poses** (top row).
  The posterior does not collapse to one pose.
- **Encoder z matches the prior on held-out x**: mean 0.002 / 0.000, std
  0.998 / 0.993, corr(z1, z2) = 0.009. Test-time sampling from N(0, I) is valid.
- **Reconstruction is not exact** (soft invertibility): RMS 0.012 in raw x
  units, with per-coordinate medians at 0, so there is no visible systematic bias.
- **Weak spot: low-density y\*.** At y\* = (1.5, 0), resim is 0.035 and the end
  points are visibly scattered. This is the same point where the MDN struggles
  (`docs/mdn.md`), and it is worse here.

Quick Err_post check (shared `metrics.mmd.mmd`, unbiased, cached ground truth),
first 100 of the 1000 test y\*, 1000 samples each. **Not the official number:**
that comes from `metrics.evaluate` on all 1000 after full training.

| | test y\* #0 | mean, 100 y\* | median, 100 y\* |
|---|---|---|---|
| Autoencoder, 10 epochs | 0.0020 | 0.0138 | 0.0048 |
| MDN K=16, 50 epochs (`kin_K16_s0`) | 0.0022 | 0.0027 | 0.0016 |

On a typical y\* the autoencoder is close to the MDN, but it has a heavy tail
(mean ≈ 3× median). The paper has the same ordering (AE 0.037 vs MDN 0.007).
The full 50-epoch schedule and the b sweep are the first things to try on the tail.

## First evaluation with the shared metrics (kinematics, 10 epochs)

Official `metrics.evaluate`: all 1000 test y\* × 1000 samples, unbiased MMD²,
cached ground truth. Model: the 10-epoch sanity run `kin_ae_sanity_s0`, so
these are **not final numbers** (the full schedule is 50 epochs). Ballistics has
not been trained yet. Output: `metrics/results/kinematics_ae_sanity10ep_s0.{json,npz}`.

```python
from metrics.evaluate import evaluate
evaluate(model.sample, 'kinematics', name='ae_sanity10ep_s0')
```

| Model | Err_post (median [q1, q3]) | Err_resim (median [q1, q3]) | Inference, 1000 y\* × 1000 samples |
|---|---|---|---|
| **Autoencoder, 10 epochs** | **0.0141** (0.0054 [0.0024, 0.0132]) | **0.0034** (0.0019 [0.0004, 0.0036]) | **8.2 s** (CPU) |
| Our MDN K=16, 50 epochs | 0.0031 | 0.0007 | 75 ms (CPU) |
| Paper autoencoder (Table 1) | 0.037 | 0.012 | < 1 ms (1080 Ti) |
| Paper MDN (Table 1) | 0.007 | 0.012 | 601 ms (1080 Ti) |

Biased-estimator Err_post: 0.0187. Hardware: Apple M4, CPU.

### ⚠️ Finding 1: accuracy beats the paper's AE, but there is a heavy tail

- The 10-epoch model is already **~2.6× better than the paper's autoencoder on
  Err_post and ~3.5× better on Err_resim**.
- It is **behind our MDN** (0.0141 vs 0.0031). The paper has the same ordering
  (AE 0.037 vs MDN 0.007).
- **Mean ≈ 2.6× median** (max 0.18): most of the error comes from a small
  number of hard y\* in low-density regions of p(y), e.g. y\* = (1.5, 0) in the
  sanity plot. **This tail is the first target** for the full 50-epoch schedule
  and the b sweep. Report median and quartiles next to the mean.

### ⚠️ Finding 2: on CPU the autoencoder is ~110× SLOWER than the MDN

The spec expected the autoencoder to be the fastest model in the study. On our
CPU measurement it is the opposite: **8.2 s vs 75 ms for the MDN**.

**Why (architecture, not a bug):** the decoder pushes *every* sample through the
full 3M-parameter network: 1000 y\* × 1000 samples = 10⁶ forward passes, about
3·10¹² FLOPs. The MDN runs its big network **once per y\*** (1000 passes), and
drawing samples from the resulting Gaussian mixture is cheap. The more samples
per y\*, the bigger the gap.

The paper's "< 1 ms" was measured on a GTX 1080 Ti, and the paper does not say
exactly what was timed (per sample? per y\*? which batch size?). A batched MLP
forward pass is exactly what a GPU is good at, so the CPU number is not
representative.

**Decision: measure inference time on a GPU.** Rules for a fair comparison:
- time every model (AE, MDN, INN, cINN, …) **on the same GPU** with the same
  `metrics.evaluate.timed_sampling` call (1000 y\* × 1000 samples, after warm-up);
- report the GPU model and the batch layout next to the number;
- **do not compare our times with the paper's ms figures**, only with each other;
- keep the CPU numbers above as a secondary data point and say why they differ.

## Hand-off to Berker: what is left

1. **Full training** on both benchmarks with the team's agreed schedule
   (`--n-train/--epochs/--batch-size`; currently the MDN placeholders):
   ```bash
   .venv/bin/python experiments/train_autoencoder.py --problem kinematics --run-name kin_ae_s0
   .venv/bin/python experiments/train_autoencoder.py --problem ballistics --run-name bal_ae_s0
   ```
   About 4.5 s per 100 steps on an M4 CPU, so 50 epochs × 1000 steps ≈ 40 min.
2. **Evaluation** with the shared metrics. Do not reimplement them. The model
   already has the interface:
   ```python
   from metrics.evaluate import evaluate
   evaluate(model.sample, 'kinematics', name='ae_s0')   # model loaded as in plot_autoencoder_sanity.load()
   ```
   Report Err_post, Err_resim and the inference time. **Measure inference time
   on a GPU**: on CPU the autoencoder is ~110× slower than the MDN (see
   Finding 2 above). Time every model on the same GPU in the same way, then
   check whether the "fastest model / speed baseline" claim actually holds.
3. **Sweeps**, logging every run: b, a, `--latent-mmd joint|marginal`, depth /
   activation at a fixed 3M budget, larger dim z, decoding from true y in L_recon.
4. **Ballistics specifics** (see `docs/mdn.md`, "Caveats"): v0 is
   integer-valued (Poisson prior) and dominates raw-unit Err_post, and x₂ has
   a point mass at 0. Use whatever the team decides there, identically for all
   models.
5. **Final table** against the paper (Tables 1–2, autoencoder rows):

   | Benchmark | Err_post | Err_resim | Inference (paper, 1080 Ti) |
   |---|---|---|---|
   | Kinematics | 0.037 | 0.012 | < 1 ms |
   | Ballistics | 0.049 | confirm from Table 2 + boxplots (clamped outliers, Sec. 4.2) | < 1 ms |

   Our MDN beat the paper's MDN by a wide margin (`docs/mdn.md`), so compare
   the autoencoder against our MDN as well as against the paper's numbers.

## Known behaviour to watch

- **Residual reconstruction error → posterior bias.** Check whether the
  per-coordinate error in the sanity plot has a nonzero median. A systematic
  offset would show up as bias in the samples.
- **Bad numbers? Check z first.** Compare the encoder-z histograms and the
  `val_z` column of `log.csv` before touching anything else (see the b = 0 row
  above).
