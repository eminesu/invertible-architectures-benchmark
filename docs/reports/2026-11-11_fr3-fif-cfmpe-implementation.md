# FIF and CFMPE on FR3 5DOF and 6DOF

Updated 12 November 2026. Training, validation evaluation, and test evaluation
are complete for four seed-0 FIF/CFMPE runs at approximately 3M parameters,
1M training samples, and 50 epochs. Protocol and reproduction commands are in
[FIF and CFMPE on FR3](../fr3_fif_cfmpe.md).

CFMPE has lower Err_post and Err_resim than FIF on both test tasks, with much
better pose accuracy and coverage. FIF generates samples much faster. Strict
IK precision remains limited: FIF produced no test samples meeting the
1 mm / 1 degree success criterion, and CFMPE's success fractions were
0.01427 on 6DOF and 0.04103 on 5DOF. These are results for one training seed
and one parameter budget, rather than a scaling study.

## Setup and evaluated checkpoints

All runs used seed 0, 1,000,000 training rows, 20,000 training-validation rows,
batch size 1000, Adam with learning rate 0.00100 and weight decay 0.00001,
cosine scheduling, gradient clipping at 10, and 50 epochs. Recorded software
and hardware: PyTorch 2.14.1+cu126, Windows, NVIDIA GeForce GTX 1650 Ti.
The project environment used to check the artifacts is Python 3.14.4.

| Run | Parameters | Hidden width | Completed epochs | Evaluated checkpoint epoch | Recorded training time (s) |
|---|---:|---:|---:|---:|---:|
| `fr3_6d_fif_s0` | 2,992,038 | 691 | 50 | 27 | 2786.56250 |
| `fr3_6d_cfmpe_s0` | 2,988,817 | 495 | 50 | 50 | 1047.94962 |
| `fr3_5d_fif_s0` | 2,994,069 | 695 | 50 | 27 | 3413.64872 |
| `fr3_5d_cfmpe_s0` | 2,999,319 | 496 | 50 | 50 | 1032.27855 |

FIF used the original MLP architecture, four hidden layers, beta 100, zero
observation noise, and latent dimension 1 for 6DOF or 2 for 5DOF. Its
`model.pt` was selected by re-simulation error during training validation;
`last.pt` retains epoch 50. CFMPE used six gated blocks, alpha 1, and
sigma_min 0.00010; its `model.pt` is the final epoch-50 model. Training times
are the trainers' recorded summary fields; their setup and provisional-metric
work differ, so they are not pure optimizer-time measurements.

| Run | Epoch-50 train loss | Epoch-50 val loss | Evaluated-checkpoint train loss | Evaluated-checkpoint val loss |
|---|---:|---:|---:|---:|
| `fr3_6d_fif_s0` | 52.66537 | 55.91591 | 68.58171 | 71.18073 |
| `fr3_6d_cfmpe_s0` | 2.52288 | 2.52177 | 2.52288 | 2.52177 |
| `fr3_5d_fif_s0` | 13.88241 | 16.02029 | 18.01548 | 18.84780 |
| `fr3_5d_cfmpe_s0` | 3.89082 | 3.90371 | 3.89082 | 3.90371 |

FIF's surrogate and CFMPE's flow-matching objective have different definitions;
their numeric losses cannot rank the methods. The 20,000 training-validation
rows are distinct from the committed benchmark validation poses. Training-time
provisional re-simulation numbers also use different sampling protocols and
are not substituted for the shared metrics below.

CFMPE estimates `p(q | y, c)` for seven prior-width scales in
`[0.50000, 1.00000]`, with 25% nominal-prior training rows. FIF learns only
the fixed benchmark prior. Both evaluations use the nominal full-joint-limit
prior; off-nominal accuracy has not been measured.

## Validation and solver selection

Both methods were evaluated on the same 200 committed validation poses, with
1000 model samples and 1000 reference samples per pose. The following are
the FIF and selected-CFMPE validation means:

| Task | Method | Err_post mean | Err_resim mean | Sampling time (s) |
|---|---|---:|---:|---:|
| 6DOF | FIF | 0.14995 | 1.90710 | 0.61916 |
| 6DOF | CFMPE Euler 64 | 0.02597 | 0.00573 | 61.89709 |
| 5DOF | FIF | 0.03014 | 0.44806 | 0.59784 |
| 5DOF | CFMPE Euler 64 | 0.00379 | 0.00104 | 58.74780 |

Five CFMPE sampling settings were compared using the same trained checkpoint.
The selection rule follows Emine's FMPE protocol: retain Err_post values no
more than 0.00050 above the best, then choose the highest success fraction
at 1 mm / 1 degree, then lower pose-error medians, then fewer network
evaluations (NFE). Success requires staying inside all joint limits.
Selection used full-precision values before test evaluation. The tolerance
was adopted from the existing protocol; its interpretation as an oracle
noise scale has not been revalidated on the regenerated local caches.

### CFMPE 6DOF validation sweep

| Setting | NFE | Err_post mean | Err_resim mean | Success at 1 mm / 1 degree | Position median (mm) | Retained |
|---|---:|---:|---:|---:|---:|---|
| Midpoint 8 | 16 | 0.02633 | 0.00676 | 0.00215 | 8.67135 | No |
| Midpoint 16 | 32 | 0.02597 | 0.00569 | 0.00628 | 6.91483 | Yes |
| Midpoint 32 | 64 | 0.02580 | 0.00613 | 0.00811 | 7.05520 | Yes |
| Euler 32 | 32 | 0.02640 | 0.00624 | 0.01520 | 5.57651 | No |
| **Euler 64** | 64 | **0.02597** | **0.00573** | **0.01765** | **5.58481** | **Yes, selected** |

### CFMPE 5DOF validation sweep

| Setting | NFE | Err_post mean | Err_resim mean | Success at 1 mm / 1 degree | Position median (mm) | Retained |
|---|---:|---:|---:|---:|---:|---|
| Midpoint 8 | 16 | 0.00375 | 0.00149 | 0.00698 | 6.80326 | Yes |
| Midpoint 16 | 32 | 0.00373 | 0.00105 | 0.01623 | 5.08302 | Yes |
| Midpoint 32 | 64 | 0.00373 | 0.00115 | 0.02426 | 4.84918 | Yes |
| Euler 32 | 32 | 0.00392 | 0.00122 | 0.03544 | 4.08502 | Yes |
| **Euler 64** | 64 | **0.00379** | **0.00104** | **0.04485** | **3.84538** | **Yes, selected** |

Midpoint 32 has the lowest Err_post on both tasks; Euler 64 is inside the
retained window and has the highest pose success among those settings.
CFMPE is a separate model from Emine's fixed-prior FMPE: solver selection
here compares only the CFMPE rows, not the FMPE results.

## Test results

Only the validation-selected CFMPE setting, Euler 64, was evaluated on test.
Each task uses 1000 committed test poses and 1000 model/reference samples per
pose. Err_post and Err_resim are arithmetic means across poses. Err_post is
unbiased MMD squared; Err_resim is the squared forward-encoding residual,
not a position error in millimetres. Position and angle columns show the
median and 95th percentile across model samples. Lim Viol and Cov are
fractions; Cov is the mean across poses. Time measures sampling of
1,000,000 model outputs only, excluding GT generation and metric computation.
All displayed numeric results use five decimal places; raw artifacts retain
full precision, and a displayed 0.00000 can be a rounded nonzero value.

### 6DOF test

| Model | Err_post mean | Err_resim mean | Pos (mm) Med/P95 | Rot (deg) Med/P95 | Lim Viol | Cov | Time (s) |
|---|---:|---:|---|---|---:|---:|---:|
| [FIF](../../metrics/results/fr3_6d_fr3_6d_fif_s0.json) | 0.15562 | 1.97056 | 193.13302 / 858.10392 | 45.63359 / 162.11481 | 0.07408 | 0.00157 | 3.00913 |
| [CFMPE Euler 64](../../metrics/results/fr3_6d_fr3_6d_cfmpe_s0_width1-1-1-1-1-1-1_eulerx64.json) | 0.02741 | 0.00698 | 5.93151 / 23.67971 | 1.29697 / 5.37915 | 0.09116 | 0.93327 | 310.45147 |

### 5DOF test

| Model | Err_post mean | Err_resim mean | Pos (mm) Med/P95 | Axis (deg) Med/P95 | Lim Viol | Cov | Time (s) |
|---|---:|---:|---|---|---:|---:|---:|
| [FIF](../../metrics/results/fr3_5d_fr3_5d_fif_s0.json) | 0.03207 | 0.45652 | 90.33020 / 419.70598 | 14.45037 / 99.60692 | 0.34753 | 0.67236 | 2.83667 |
| [CFMPE Euler 64](../../metrics/results/fr3_5d_fr3_5d_cfmpe_s0_width1-1-1-1-1-1-1_eulerx64.json) | 0.00385 | 0.00121 | 4.03223 / 15.87672 | 0.59496 / 2.61333 | 0.06258 | 0.97118 | 297.89331 |

The 5DOF angle is the error between flange axes, not a full-rotation error.
Coverage uses a joint-space radius of 0.10000 rad for 6DOF and 0.50000 rad
for 5DOF, so coverage values from the two tasks are not directly comparable.

| Task | Method | Success at 1 mm / 1 degree | Success at 10 mm / 5 degrees | Non-finite redraws |
|---|---|---:|---:|---:|
| 6DOF | FIF | 0.00000 | 0.00011 | Not used |
| 6DOF | CFMPE Euler 64 | 0.01427 | 0.69925 | 0 |
| 5DOF | FIF | 0.00000 | 0.00501 | Not used |
| 5DOF | CFMPE Euler 64 | 0.04103 | 0.83542 | 0 |

CFMPE test commands enabled `--redraw-nonfinite`, but neither test needed a
replacement. Samples were not clamped to joint limits or refined with IK.
No forward-simulation failures were recorded in these four test evaluations.

## Reference caches and comparison limits

The four local caches were generated once per task/split and reused by FIF
and CFMPE. They have the expected sample-array shapes and report all traced
loops closed, with 4 to 8 loops per pose. Their SHA256 hashes differ from
Emine's files even though the filenames and target poses match.

| File in `metrics/cache/` | Split | Sample-array shape | Local SHA256 |
|---|---|---|---|
| `gt_fr3_6d_n1000_s777_3297457ebd.npz` | val | `(200, 1000, 7)` | `7b0c776ac5cc2fd769882ef779aebdd892be8925aaa2e46413e9498e85d3876e` |
| `gt_fr3_6d_n1000_s777_fb7720ec0b.npz` | test | `(1000, 1000, 7)` | `993bcff703c059885855f1b1c0140ec14844c4c3c49f0df5f1056f8037209f8e` |
| `gt_fr3_5d_n1000_s777_068e186e5c.npz` | val | `(200, 1000, 7)` | `c79a62d9a9acaedd5283bbb4b086f7545c64ef94fb2aca09aaeb054df7dc72b2` |
| `gt_fr3_5d_n1000_s777_f0b43cd65f.npz` | test | `(1000, 1000, 7)` | `42bd4203f5ee2f0e0b96cd2f985fd03112a96af7b9f9b1026c1e83a5d305697f` |

Emine's original hashes are recorded in her
[FMPE report](2026-11-11_fr3-fmpe.md#ground-truth). Direct comparisons with her
FMPE/FFF/M-FFF numbers therefore use different reference draws. Her RTX 3060
Laptop timings also cannot be ranked directly against the local GTX 1650 Ti
timings. The locally regenerated caches lack the `x_true_on_fiber` diagnostic;
an oracle recheck, missed-component check, and ground-truth calibration on
these files have not been completed. The original committed oracle results
have not been refreshed for them.

Model calibration/SBC for these FR3 FIF/CFMPE runs is also outstanding.
No accuracy claim is made for narrower CFMPE priors, other training seeds,
different budgets, or longer training. The cause of FIF's low pose accuracy
has not been isolated; the present results do not establish that the method
cannot work on this problem.

## Artifacts and verification

Training artifacts are in `experiments/runs/<run>/`: `config.json`,
`log.csv`, `summary.json`, and `model.pt`, plus FIF `last.pt` and CFMPE
`checkpoint.pt`. All four logs contain 50 completed epochs. The source
metric files are in `metrics/results/`, each with a matching `.npz`:

- FIF: `<problem>_<problem>_fif_s0_val.json` and
  `<problem>_<problem>_fif_s0.json`.
- CFMPE validation:
  `<problem>_<problem>_cfmpe_s0_width1-1-1-1-1-1-1_<method>x<steps>_val.json`,
  for the five settings listed above.
- CFMPE test:
  `<problem>_<problem>_cfmpe_s0_width1-1-1-1-1-1-1_eulerx64.json`.

Here `<problem>` is `fr3_6d` or `fr3_5d`. All 16 summaries (12 validation,
4 test) were checked against their saved per-pose arrays: dimensions, finite
counts, and Err_post/Err_resim means agree. Each has 1000 reference samples
per pose and full evaluation scope. These artifact checks did not rerun
training, sampling, or GT generation.

During implementation, 14 checkpoint-I/O tests, 8 queue-controller tests,
and 4 mocked FIF output-preservation cases passed. A full regression-suite
run is not reported here. FR3 adaptations remain in experiment-local
adapters and scripts; the shared data/prior registry, robot task classes,
ground-truth module, and `ConditionalFMPE` class were not changed by this work.
Existing colleagues' result artifacts were preserved.
