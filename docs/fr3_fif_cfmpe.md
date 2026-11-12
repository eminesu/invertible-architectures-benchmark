# FIF and CFMPE on the FR3 benchmark

Implementation and experiment protocol. No FR3 FIF/CFMPE training, evaluation,
ground-truth generation, or results are included in this change.

FR3-specific adaptations live in `experiments/fr3_fif_cfmpe_adapters.py`.
The shared files `data/toy_data.py`, `robot_ik/tasks.py`,
`metrics/ground_truth.py`, and `models/conditional_fmpe.py` are unchanged.
The adapter creates a separate prior registry and a local CFMPE subclass;
it does not mutate shared registries or patch shared classes at runtime.

## Benchmark and the reviewed colleague work

The primary problem is `fr3_6d`: seven joint angles in radians, conditioned on
the flange position and first two rotation-matrix columns (nine numbers).
The pose has six independent constraints, so the regular posterior is a
one-dimensional set. `fr3_5d` is also supported, with six encoded numbers,
five independent constraints, and a two-dimensional posterior.

The prior, flange, joint limits, conditions, exact posterior tracer, and metrics
are the professor's existing setup in [robot_ik.md](robot_ik.md).
Samples are neither clamped to joint limits nor refined by numerical IK.

Reviewed reference commits:

| Author | Commit | Relevant precedent |
|---|---|---|
| Emine | `f443004` | FR3 FMPE: ~3M parameters, 1M train / 20k validation, 50 epochs; validation sampler selection, then held-out test reporting |
| Emine | `2caa0e3`, `a7e0cb6` | Validation split, FR3 marginal plots, explicit counting of optional non-finite redraws |
| Emine | `412467f`, `512c478` | Existing CFMPE is conditioned on a family of priors, in addition to the observation |
| Umut | `e21300a` | Repeatable ILNN command/variant sweeps, parameter budgets, saved configs and logs; the artifacts in this commit are for the planar task |

The experiment design follows [IKFlow](https://arxiv.org/abs/2111.08933) in
assessing pose accuracy, diverse joint solutions, and speed. The scaling sweep
uses the empirical approach of [Kaplan et al.](https://arxiv.org/abs/2001.08361);
language-model exponents are not assumed for either method here.

## Method adaptations

### FIF

- Existing `models/fif.py`, including both `--arch mlp` and `--arch paper`.
- FR3 dimensions use an experiment-local constraint map: `(x, y, z) = (7, 9, 1)` for 6-D,
  `(7, 6, 2)` for 5-D. Encoded pose width is not the number of constraints.
- Default observation noise is zero because the reference is the exact
  deterministic posterior. Sampling then conditions directly on the target;
  the toy tasks retain their existing noisy-observation KDE bridge.
- Training retains the exact trace surrogate, beta 100, and checkpoint
  selection by validation re-simulation error. Both architectures use the
  actual seven-dimensional parameter count when resolving the width. No
  metadata is added to the shared robot task classes.
- Validation now re-simulates seven-variable samples. Logs also include
  update count and elapsed seconds.

The one-dimensional latent space describes the local posterior dimension.
A continuous decoder may struggle to cover disconnected solution components;
measure MMD, coverage, and calibration alongside pose accuracy.

### CFMPE: a proposed FR3 prior family

The existing method estimates `p(q | y, c)`, where `c` describes the prior.
For FR3, define seven width scales `c_j` independently in `[0.5, 1.0]`:

```
mid_j = (lower_j + upper_j) / 2
q_j | c_j ~ Uniform(mid_j - c_j * (upper_j - lower_j) / 2,
                   mid_j + c_j * (upper_j - lower_j) / 2)
```

Thus `c = (1, ..., 1)` is exactly the unchanged benchmark prior. The default
nominal-prior sampler reproduces its original draws for the same RNG seed.
Training uses 25% nominal-prior rows, with the other rows drawn from the width
family. This anchor directly represents the seven-dimensional corner where
the benchmark is evaluated. `--nominal-fraction` controls the mixture.

The seven prior inputs are appended to the pose, retaining the existing gated
velocity field, flow-matching loss, normalization, sigma_min 1e-4, and alpha 1.
The 3M budget includes the additional prior inputs. No observation noise is
added. This prior-family and anchoring choice is an implementation proposal;
its quality has not yet been measured.

Evaluation is restricted to all widths equal to one, using the professor's
exact cache. Narrower priors raise an error because their posterior reference
is not implemented. The local evaluator adapter converts nominal widths to
`prior_params=None` before calling the unchanged shared evaluator. The other
priors' training data do not modify the benchmark's uniform prior. A small
normalization floor is implemented only in the FR3-specific CFMPE subclass.

CFMPE learns across a family while FIF learns only the fixed prior. Report
that difference when comparing them with Emine's fixed-prior FMPE baseline.

## Headline budget and planning

Defaults: 3M maximum parameters, 1M train / 20k validation rows, 50 epochs,
batch 1000, seed 0, Adam 1e-3 with weight decay 1e-5, cosine learning rate,
gradient clipping 10. FIF defaults to the original MLP architecture. The
configuration records the resolved width and actual parameter count.

Use the project environment in PowerShell:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py experiments/run_fr3_fif_cfmpe.py --stage train
& $py experiments/run_fr3_fif_cfmpe.py --stage gt
& $py experiments/run_fr3_fif_cfmpe.py --stage val
```

These commands print plans only. Append `--execute` when intentionally ready
to perform a stage. The `gt` stage generates the shared validation and test
caches through the existing script; use the same files for every model and
check the regenerated oracle before comparing distributional errors.

Run names encode task, method, architecture (FIF), budget, training-data size,
epochs, and seed. An optional `--tag` distinguishes other protocol changes.
`--problem fr3_5d` selects the second task; it is not run as part of 6-D.
CFMPE resumes from the newest complete `checkpoint.pt` or `checkpoint.pt.tmp`.
Its local file-I/O helper retries transient Windows sharing failures before
giving up, preserving the old checkpoint and the complete temporary file.
On resume, any logged epochs beyond the selected checkpoint are removed after
backing up the original log, so a retrained epoch is not counted twice.
FIF refuses
an existing training folder. Evaluation outputs are preserved on repeated runs.

The underlying trainers can also be used directly:

```powershell
& $py experiments/train_fif.py --problem fr3_6d --run fr3_6d_fif_s0 --arch mlp --device cuda
& $py experiments/train_conditional_fmpe.py --problem fr3_6d --run-name fr3_6d_cfmpe_s0 --device cuda --alpha 1 --nominal-fraction 0.25
```

## Validation, test, and calibration

FIF can be scored with `evaluate_fif.py <run> --split val --device cuda`.
Both FR3 methods keep training artifacts (`model.pt`, `config.json`,
`log.csv`, and `summary.json`) in `experiments/runs/<run>/`, and evaluation
metrics in `metrics/results/`, matching Emine's FMPE layout. For FIF, examples
are `fr3_6d_fr3_6d_fif_s0_val.json` (validation) and
`fr3_6d_fr3_6d_fif_s0.json` (test), with matching `.npz` files for per-pose
metrics. The same naming applies to `fr3_5d`. CFMPE filenames also include
the prior tag and solver. Existing result files are preserved; use `--name`
or `--out` for a separate evaluation. Toy FIF runs retain their existing
run-local `evaluation[_N]/` layout.

The CFMPE validation grid matches Emine's FR3 grid:

```powershell
& $py experiments/evaluate_conditional_fmpe.py fr3_6d_cfmpe_s0 --split val --solver midpoint:8 midpoint:16 midpoint:32 euler:32 euler:64 --device cuda
```

Choose the solver on validation: keep Err_post values within 5e-4 of the best,
then prefer higher success at 1 mm / 1 degree, lower pose-error medians, and
lower NFE. The 5e-4 tolerance is the existing oracle's approximate noise scale;
check it after regenerating the reference. Record the selection before testing.

The suite's test stage requires one explicitly supplied CFMPE solver. For
example, **only if validation selects it**, use `--stage test --solver euler:64`.
FIF has no ODE solver sweep. `evaluate_fif.py --split test` reports its chosen
checkpoint. The shared metrics include MMD squared, encoding re-simulation
error, position and rotation errors, both success thresholds, joint-limit
violations, coverage, and sampling time.

CFMPE defaults to 8192 ODE states per chunk to bound GPU memory. Optional
`--redraw-nonfinite` replaces diverged draws and records their number, as in
Emine's evaluator. It is off by default; no valid out-of-limit samples are
rejected. Use the same redraw policy and device for compared timings.

The local calibration entry point supplies the nominal prior and calls the
unchanged shared calibration functions:

```powershell
& $py experiments/evaluate_fr3_fif_cfmpe_calibration.py fif fr3_6d_fif_s0 --device cuda
& $py experiments/evaluate_fr3_fif_cfmpe_calibration.py cfmpe fr3_6d_cfmpe_s0 --solver euler:64 --device cuda
```

Again, substitute the validation-selected CFMPE solver. Calibration uses the
held-out test conditions. The local CFMPE sampler accepts `--chunk` (8192
by default) to bound ODE memory.

For a small integration check, evaluator flags `--conditions`, `--samples`,
and `--gt-samples` explicitly produce pilot-scoped results. A different subset
or reference sample count may require a separate ground-truth cache.

## Plots and preliminary parameter scaling

The FR3 plotter supports both methods' saved learning curves. FIF's surrogate
may be negative and is not a log density; it uses a linear vertical axis in
that case. Different methods have separate objective panels.

```powershell
& $py experiments/plot_fr3_fif_cfmpe.py fr3_6d_fif_s0 fr3_6d_cfmpe_s0 --out experiments/figures/shared/comparison/fr3_learning_curves.png
& $py experiments/plot_fr3_fif_cfmpe.py fr3_6d_fif_s0 --posterior --split val --out experiments/figures/fif/posterior/fr3_6d_val.png
```

Learning curves read files only. Posterior plots sample the model and use the
exact reference, with seven joint marginals and the URDF limits marked.

Plan a size sweep while keeping dataset, training schedule, and evaluation
protocol fixed:

```powershell
& $py experiments/run_fr3_fif_cfmpe.py --stage train --budgets 300000 1000000 3000000 --seeds 0 1 2
& $py experiments/run_fr3_fif_cfmpe.py --stage val --budgets 300000 1000000 3000000 --seeds 0 1 2
```

For a parameter-only scaling fit, use the same CFMPE sampler at every size,
for example by passing a single `--solver` on the validation stage. The
analyzer separates solver settings and training protocols. It uses actual
parameter counts, requires at least three distinct sizes and the same seeds
at every size, and rejects duplicate size/seed evaluations.

```powershell
& $py experiments/analyze_fr3_scaling.py experiments/runs metrics/results --split val --metric pos_err_mm.median --out experiments/figures/shared/comparison/fr3_parameter_scaling.json
```

The output fits `E = A (N / 1e6)^b` and records sources, log-space R-squared,
and whether there is only one training seed. Nonpositive MMD estimates cannot
be fitted in log space. Provide individual JSON paths when a folder contains
multiple evaluations of the same run. A fit from three sizes is exploratory;
these are fixed-data/epoch trends, with no estimated irreducible-error floor
or compute-optimal exponent. Training objectives are not compared across
families. No scaling law has been estimated by this implementation change.

## Verification status

Regression tests were added for intrinsic FIF dimensions and gradients,
seven-variable validation, prior-family bounds and nominal equivalence,
exact-reference routing, CFMPE checkpoint/evaluator integration, plan-only
behavior, unchanged shared registries/classes, toy-data delegation, and
scaling-fit protocol checks. They have not been executed, in
accordance with the request to implement without running experiments.

The checkpoint file-I/O regression suite was subsequently executed while fixing
a Windows checkpoint-replacement failure: 14 tests passed. It simulates locks,
partial saves, temporary-file recovery, and log alignment without training or
GPU work. The model-integration and experiment tests above remain unexecuted.
