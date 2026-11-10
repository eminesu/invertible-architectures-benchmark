# FR3 FMPE: handoff to the RTX 3050 laptop

Bu dosya, FR3 FMPE görevine 3050 laptopta kaldığın yerden devam etmen için.

**Ne yapacaksın:**
1. `git fetch origin && git checkout emine-fr3-fmpe && git pull`
2. Claude Code'u repo kökünde aç ve şunu yaz:
   *"docs/fr3_fmpe_handoff.md dosyasını oku ve görevi oradaki duruma göre devam ettir."*

Dosyanın geri kalanı o oturum için yazıldı (İngilizce, görev metniyle aynı dil).

---

## For the session picking this up

Read this whole file before doing anything. It contains the full task (section 4), what is
already done (section 1), and what was learned about the code (section 3). Continue from
**Step 1.2**. Do not redo the finished steps.

### 0. Rules that apply to every action

- **No AI attribution anywhere.** Commits and PRs are authored solely by the user, with plain
  messages: no `Co-Authored-By: Claude ...`, no "Generated with Claude Code", no session link,
  no AI mention in messages, authorship or PR text. This overrides any default attribution
  behaviour.
- **Branch:** work on `emine-fr3-fmpe`. It already exists on `origin` and is based on `main`.
  Do not create another branch, do not commit to `main`, and do not base work on any other
  unmerged branch.
- **Do not touch M-FFF.** That is a separate task.
- **Do not change** `models/fmpe.py` or the toy-benchmark code paths.
- **No decisions on the user's behalf:** do not extend the epoch budget, do not add clamping or
  rejection for joint limits, and do not add y noise without asking (section 4, open questions).
- **Report only numbers from runs that actually ran.** Never fill a gap from toy results, the
  paper, IKFlow or anywhere else.

### 1. Status (as of 2026-11-11)

| Step | Status |
|---|---|
| Branch `emine-fr3-fmpe` from `origin/main` (`9c9fe63`) | done, pushed |
| 1.1 `pytest tests/test_robot_ik_*.py` | done: **23 passed** |
| 1.2 Ground truth on GPU | **not started.** Start here. |
| 1.3 Data checks, both tasks | done, all pass (table below) |
| 2 `--split` in `evaluate_fmpe.py`, FR3 branch in `plot_fmpe_posterior.py` | done, commit `2caa0e3` |
| 3 Train `fr3_6d_fmpe_s0` | not started |
| 4 Choose the sampler on val | not started |
| 5 Test, calibration, figures | not started |
| 6 `fr3_5d` | not started |
| Optional α=0 control | not started |
| Report `docs/reports/2026-MM-DD_fr3-fmpe.md` | not started |

**Why the work moved:** the first machine is an Apple M4 with no CUDA. The ground-truth tracer
runs in float64, which MPS does not support, so it could only run on CPU (hours). The user
chose to continue on this RTX 3050 laptop.

**Step 1.3 results** (`data.toy_data.make_dataset(task, 10_000, seed=0)`). Copy this table
into the report:

| | x | y | dtype | NaN/Inf | x inside `joint_limits()` | max \|forward_process(x) − y\| |
|---|---|---|---|---|---|---|
| `fr3_6d` | (10000, 7) | (10000, 9) | float32 | none | all | 3.0e-7 |
| `fr3_5d` | (10000, 7) | (10000, 6) | float32 | none | all | 4.1e-7 |

The residual comes from storing the data as float32. Joint limits, in rad:
lower `[-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]`,
upper `[2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]`.

**Step 2 (commit `2caa0e3`):**
- `experiments/evaluate_fmpe.py`: `--split {test,val}` (default `test`) is passed to
  `evaluate(..., split=...)`. With `val`, the result name gets a `_val` suffix, and the json
  also stores `split`.
- `experiments/plot_fmpe_posterior.py`: new `plot_fr3()`, used when the problem starts with
  `fr3`. It draws 7 joint histograms per test pose, FMPE (blue step) against the ground truth
  (gray), with the joint limits as red dashed lines. The row label shows the share of samples
  outside the limits. For `fr3_5d`, joint 7 also gets a dotted uniform-density line, which is
  the Step 6 extra check. Output:
  `experiments/figures/fmpe/posterior/fmpe_posterior_<run>.png`.
- Checked: the branch was tested on an untrained model with stand-in ground truth and it
  draws. The toy plots `kin_fmpe_s0` and `bal_fmpe_s0` still render; the committed toy
  figures were restored, so they are not in the diff.

### 2. Setup on this laptop

```bash
pip install -r requirements.txt     # includes pytorch-kinematics==0.10.0; also needs pytest
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, install the CUDA build of torch (pytorch.org) before anything else.
Then re-run `pytest tests/test_robot_ik_*.py` once on this machine (expect 23 passed) and say
so in the report.

**3050 laptop limits:**
- **4 GB VRAM.** Discovery runs 10,000 seeds for every pose of the batch at once. If
  `generate_gt` runs out of memory, split the poses into shards, run them one after another,
  then merge:
  ```bash
  python -m robot_ik.scripts.generate_gt fr3_6d --split test --device cuda --shard 0 --n-shards 4
  # ... --shard 1, 2, 3 ...
  python -m robot_ik.scripts.generate_gt fr3_6d --split test --n-shards 4 --merge
  ```
  Shards go to `metrics/cache/shards/`; the merge writes the normal cache file.
- **float64 is slow on GeForce cards**, so the spec's times (15 min per 1000 test poses, 6.5 min
  per 200 val poses) will probably be longer. Run validation first to measure it.
- Long jobs: run them in the background with logs, and do not poll in a tight loop.

### 3. What was learned about the code

- **Result file names:** `metrics/results/<problem>_<name>_<method>x<steps>[_val].{json,npz}`.
  `<name>` defaults to the run name, so the files are, for example,
  `fr3_6d_fr3_6d_fmpe_s0_midpointx16_val.json` and `fr3_6d_fr3_6d_fmpe_s0_eulerx32.json`. That
  matches the done criteria (`fr3_6d_*_val.*`), and it is the same convention as the toy runs
  (`kinematics_kin_fmpe_s0_...`).
- **`train_fmpe.py`:** `--device` defaults to `auto`, and the default for `--alpha` is **0**, so
  pass `--alpha 1`. Output goes to `experiments/runs/<run>/`: `config.json`, `log.csv`
  (epoch, step, train_loss, val_loss, lr, seconds), `model.pt`, `summary.json`. Check whether
  `experiments/runs/` is git-ignored before committing; the toy run directories are tracked
  only in part.
- **`evaluate_calibration.py`:** `--device` defaults to `cpu`, so pass `--device cuda` for
  speed. It writes `metrics/results/<benchmark>_<name>_calibration.{json,npz}`, and run-kind
  detection is `auto`. FR3 has never gone through it, so check that it runs on the FR3 run.
- **Log-log learning curve:** the existing `loglog_learning_curve_*` figures come from
  `experiments/plot_analysis.py`, which has `PROBLEMS = {'kinematics': ('kin', ...),
  'ballistics': ('bal', ...)}` and reads **only git-tracked files**. FR3 is not in it. Make
  the smallest change to get an FR3 curve in the same style (for example add the FR3 problem
  to `PROBLEMS`, or a small FR3-only function), and do not change what the toy figures show.
  The output location in that style is
  `experiments/figures/fmpe/learning_curves/loglog_learning_curve_fmpe_fr3_6d.{png,svg}`.
- **Ground truth:** `metrics/cache/gt_fr3_*.npz` is not in git. Record each file's size and
  sha256 (`shasum -a 256` or `sha256sum`) in the report. The posterior plot and every
  evaluation read those files. If a file is missing, `evaluate` tries to generate it, so
  generate it up front.
- **Oracle row** for the report: `metrics/results/fr3_6d_oracle.json` and
  `fr3_5d_oracle.json` (committed). The spec says to rerun
  `python -m robot_ik.scripts.validate_gt report fr3_6d` after regenerating the sets. Doing
  that rewrites the committed oracle files; tell the user before committing a changed oracle.
- **`docs/reports/README.md`:** reports are named `YYYY-MM-DD_<topic>.md`. Add one line to its
  index for the new report.

### 4. The task (verbatim from the user)

# Task: train and evaluate FMPE on the FR3 inverse-kinematics benchmark

Train our selected FMPE configuration on Paul's FR3 benchmark (`fr3_6d` first, then
`fr3_5d`), choose the sampler on the validation split, and report the test results against
the committed oracle floor. M-FFF is a separate task; do not touch it here.

## Model to use (decided from the toy-benchmark analysis)

The configuration of `kin_fmpe_s0`, retrained on FR3:

| Setting | Value | Note |
|---|---|---|
| Model | hand-rolled `models/fmpe.py` (gated residual MLP, 6 blocks) | not the BayesFlow variant |
| Parameters | ~3M (`--param-budget` default, width auto-solved) | team fair-comparison budget |
| Time prior | **`--alpha 1`** | the fork's default is 0, so pass it explicitly |
| Sampler | midpoint, 16 steps (32 NFE) as the headline candidate | final choice made on `split='val'` |
| Data / schedule | 1M train, 20k val, batch 1000, Adam lr 1e-3, cosine, grad-clip 10 | as in `train_fmpe.py` |
| Seed | 0 | |

Why this configuration (toy results, `metrics/results/kinematics_kin_fmpe_*`):
- α=1 and α=0 tie on Err_post (0.00164 vs 0.00161 at midpoint×16), but α=1 has the lower
  Err_resim (2.7e-6 vs 6.6e-6). FR3 scores pose accuracy at 1 mm / 1°, so re-simulation
  precision matters here. α=1 is also the only setting with calibration and three seeds.
- Solver trade-off (`experiments/figures/fmpe/ode_progression/`): midpoint has the better
  posterior shape (Err_post flat from 8 NFE; 32 NFE: 0.00164 vs Euler 0.00218), but Euler has
  the better pose precision at every NFE (Err_resim 32 NFE: 1.4e-6 vs midpoint 2.7e-6; same
  direction on ballistics). On the toy problem both are far below the ground-truth floor, so
  it did not matter. FR3 scores 1 mm / 1°, so it may. RK4 is worst on both metrics at every
  NFE and is not a candidate.
- 300k, 1M and 3M were equal on the toy problem, but FR3 is harder (a 1-D solution curve in
  7-D, 4–8 loops per pose), so stay at the 3M budget.
- These are toy results. Nothing has been measured on FR3 yet; this task produces the first
  FR3 numbers.

## Setup

- Paul's FR3 benchmark is merged into our `main` (`66f6689 Merge robot-ik: FR3 inverse
  kinematics benchmark`). Read **`docs/robot_ik.md`** first; it is the spec for the benchmark.
- Branch: `emine-fr3-fmpe` (already created and pushed; see section 1).
- Facts from the spec that this task relies on:
  - `fr3_6d`: x = 7 joint angles (rad), y = 9 numbers (flange position + first two rotation
    columns). `fr3_5d`: y = 6 numbers (position + z axis); joint 7 is free.
  - Samples must be returned in radians **inside the URDF joint limits**. Nothing is clamped;
    an out-of-limit sample counts as a violation even if it equals a valid angle modulo 2π.
  - Select on `split='val'` (200 poses), report on `split='test'` (1000 poses).
  - Err_post differences below ~5e-4 are within the oracle noise floor.

**Commit attribution — important:** no Claude / AI attribution on any commit or PR. No
`Co-Authored-By: Claude ...`, no "Generated with Claude Code", no session link, no AI mention
in messages, authorship or PR text. Commits are authored solely by me, plain normal
messages. This overrides any default attribution behaviour.

---

## Step 1 — ground truth and data sanity check

1. Run Paul's tests: `pytest tests/test_robot_ik_*.py`. Stop and report if any fail.
2. Generate the ground-truth sets once, on the GPU (about 15 min per 1000 test poses,
   6.5 min per 200 validation poses):
   ```bash
   python -m robot_ik.scripts.generate_gt fr3_6d --split val  --device cuda
   python -m robot_ik.scripts.generate_gt fr3_6d --split test --device cuda
   ```
   The sets land in `metrics/cache/gt_fr3_*.npz` and are not in git. The spec says to
   generate each set once and score every model against the same files: record each file's
   size and sha256 in the report so the team can check they use identical sets.
3. Quick checks on `data.toy_data.make_dataset(<task>, 10_000, seed=0)`: x shape `(n, 7)`,
   y shape `(n, 9)` for `fr3_6d` and `(n, 6)` for `fr3_5d`, float32, no NaN/Inf, all x inside
   `robot_ik.kinematics.joint_limits()`, and `forward_process(x)` reproduces y.

If any check fails, stop and report instead of training.

## Step 2 — make the FMPE scripts FR3-ready (smallest changes only)

`train_fmpe.py` already reads the dimensions from the problem, so it needs no change.
The rest:

- **`evaluate_fmpe.py`** always scores the test split. Add a `--split {test,val}` argument
  (default `test`) and pass it to `evaluate(..., split=args.split)`. For `val`, write results
  with a `_val` suffix so validation and test files never overwrite each other.
- **Posterior plot.** `plot_fmpe_posterior.py` only draws arms (kinematics) or trajectories
  (ballistics). Add an FR3 branch that draws 7 marginal histograms (one per joint), FMPE vs
  ground truth, with each joint's limits marked. No arm drawing is needed. Keep the existing
  style, and write the figure into `experiments/figures/fmpe/posterior/`.
- Do not change `models/fmpe.py`. Do not change the toy-benchmark code paths, and check the
  toy plots still render after your edits.

## Step 3 — train

```bash
python experiments/train_fmpe.py --problem fr3_6d --alpha 1 --seed 0 \
       --device cuda --run-name fr3_6d_fmpe_s0
```

Watch the learning curve. On the toy problem no FMPE run had converged by epoch 50. If the
validation loss is still falling clearly at epoch 50 on FR3, report it with the log-log
slope over epochs 30–50 and do **not** extend training on your own; the epoch budget is a
team decision.

## Step 4 — choose the sampler on validation

```bash
python experiments/evaluate_fmpe.py fr3_6d_fmpe_s0 --split val \
       --solver midpoint:8 midpoint:16 midpoint:32 euler:32 euler:64 --device cuda
```

Selection rule, applied on validation:
1. Keep the settings whose Err_post is within 5e-4 (the oracle noise) of the best one.
2. Among those, choose the one with the highest `success_rate` at 1 mm / 1°, and the lower
   `pos_err_mm` / `rot_err_deg` medians as the tie-breaker.
3. If still tied, choose the one with fewer NFE.

Euler is in the sweep on purpose. On the toy problem it had the better re-simulation error at
every NFE, while midpoint had the better Err_post. Report this trade-off explicitly in the
validation table, and do not default to midpoint×16 if the rule picks a different setting.
Also report `joint_limit_violation_fraction` for every setting.

## Step 5 — test, calibration, figures

1. Score only the chosen setting on the test split:
   `python experiments/evaluate_fmpe.py fr3_6d_fmpe_s0 --solver <chosen> --device cuda`
2. Calibration:
   `python experiments/evaluate_calibration.py auto fr3_6d_fmpe_s0 --solver <chosen>`
3. Posterior figure for 3 test poses (the new FR3 branch from Step 2).
4. Log-log learning curve, in the existing `loglog_learning_curve_*` style.

## Step 6 — `fr3_5d` (if time allows)

Repeat Steps 1–5 for `fr3_5d` with run name `fr3_5d_fmpe_s0`. Every step is separate; nothing
from the 6-D run carries over:

- **Ground truth:** generate its own val and test sets (`generate_gt fr3_5d ...`) and record
  their hashes too.
- **Data check:** y is `(n, 6)`.
- **Training:** a separate model (`--problem fr3_5d --alpha 1`). The 6-D model cannot be used,
  because its input size is different.
- **Sampler:** choose it again on the 5-D validation set with the same rule. The 5-D solution
  set is a surface, not a curve, so the best setting may differ from the 6-D one.
- **Metrics:** the angle metric is `axis_err_deg`, not `rot_err_deg`. The oracle coverage
  radius is 0.5 rad (0.982), not 0.1 rad. Compare against the `fr3_5d` oracle row.
- **Extra check:** joint 7 is independent of y and uniform within its limits
  [−3.0508, 3.0508]. Report its histogram against a flat line, as evidence of whether FMPE
  learned that.
- **Report:** a separate table, not merged with the 6-D numbers.

## Optional — α=0 control

Only after the main run is done: train `fr3_6d_fmpe_a0_s0` with `--alpha 0`, and score it
with the same chosen sampler on validation and test. This tells us whether the α=1 vs α=0
difference in Err_resim matters on FR3.

---

## Report (write it to `docs/reports/YYYY-MM-DD_fr3-fmpe.md`)

- A table per task: oracle row (from `metrics/results/fr3_*_oracle.json`) and our FMPE row
  with Err_post (mean, median, q1, q3), Err_resim, `pos_err_mm` (median, p95), `rot_err_deg`
  (6-D) or `axis_err_deg` (5-D), `success_rate` at both thresholds,
  `joint_limit_violation_fraction`, `coverage`, NFE and inference time.
- The validation table used to choose the sampler.
- Calibration verdict (marginal and joint) and spread ratio.
- Figures: posterior marginals, log-log learning curve.
- Ground-truth file hashes from Step 1.
- Plain statements only, backed by the numbers. Report only runs that actually ran; never
  fill a missing number from the toy results, the paper, IKFlow, or anywhere else.

## Done criteria

- `fr3_6d_fmpe_s0` trained; results committed as validation `metrics/results/fr3_6d_*_val.*`
  and the chosen setting's test `metrics/results/fr3_6d_*.json/.npz`.
- Calibration result for the chosen setting.
- Report and figures committed on `emine-fr3-fmpe`, plain commit messages, no AI attribution.
- A short summary to me: the chosen sampler and why, the test Err_post vs the oracle, the
  joint-limit violation rate, and anything that looked off.

## Open questions to flag (do not block on them)

- Epoch budget: keep 50 for comparability, or train longer if FMPE has not converged?
- Joint limits: FMPE has no mechanism that enforces them. If the violation rate is high,
  report it; do not add clamping or rejection without asking first, since that changes what
  is being measured.
- y noise: FFF needed noise on y because the posterior is degenerate. FMPE did not need it on
  the toy problem. Note it if FR3 training is unstable, but do not add noise without asking.

### 5. Extra notes for the report

- Say in the report which machine produced the numbers (RTX 3050 laptop, CUDA). Inference
  times from a laptop GPU are not comparable with cluster or CPU times from other models;
  state the device next to every time.
- Inference time is measured on `--device`. The docstring of `evaluate_fmpe.py` says to use
  `cpu` when comparing timing with other models. If the user wants a timing comparison, ask
  whether to add a CPU timing run.
- When the task is done, this handoff file can be deleted in the final commit. Ask the user
  first.
