# IKFlow

Ames, Morgan & Konidaris, *IKFlow: Generating Diverse Inverse Kinematics Solutions*, IEEE RA-L 2022,
[arXiv:2111.08933](https://arxiv.org/abs/2111.08933). Code: `models/ikflow.py`,
`experiments/train_ikflow.py`, `experiments/evaluate_ikflow.py`, `tests/test_ikflow.py`.

## What it is

A conditional normalizing flow over the joint angles, conditioned on the end-effector pose, trained
by maximum likelihood. It is the same GLOW-style conditional flow as the cINN in `models/inn.py`
(`models.inn.Flow`) plus four things:

| | paper | here |
|---|---|---|
| block | GLOW coupling, subnets 3 x 1024, LeakyReLU, conditioning in every subnet | `models.inn.Flow`, `--hidden`, `--layers 3`, `--activation leaky_relu` |
| width | flow width >= joints, joints zero-padded (Panda: 7 joints, width 9, 12 blocks) | `--width` (default joints + 2), `--blocks 12`; padding dropped when sampling |
| SoftFlow | c ~ U(0,1), noise c * 1e-3 added to the data, c fed to every subnet, c = 0 at test | `--softflow-scale 1e-3`, noise in raw joint units (rad) |
| sampling | z scaled by 0.25 | `--latent-scale 0.25` (training-time check); `evaluate_ikflow.py --latent-scale` |
| optimisation | Ranger, lr 5e-4, x0.979 every 39000 batches, batch 128, 2.5M points | `--optimizer ranger` (RAdam + Lookahead k=6, alpha=0.5), same lr schedule, `--batch-size 128`, `--paper-size` |

Check against the paper: with conditioning size `dim(y) + 1` (7-D pose + c), the closed-form
parameter count of `Flow` reproduces all six rows of the paper's Table I to within 0.02% (they look
truncated, not rounded): width 9 / 12 blocks gives 5.0934e7 against the reported 5.093e7. Without
the extra c channel the counts come out about 0.04% too low. This is `test_table1_parameter_counts`.

## Not in the paper, chosen here

* Soft clamp 2.0 on the log-scales and identity initialisation (the repo's `Flow`, as FrEIA).
* Noise parameterisation: the paper writes `v ~ N(0, c)`; this follows SoftFlow, std = c * scale.
* Lookahead k = 6, alpha = 0.5 (Ranger's defaults); gradient clipping at 10 (`--grad-clip 0` disables).
* Default size: fitted to `--param-budget` (3M, like the other models) instead of ~5e7. `--paper-size`
  gives 1024-wide subnets (51.0M parameters on the FR3 tasks) and 2.5M points. The number of epochs
  is not stated in the paper ("until convergence").
* The paper's conditioning is the 7-D pose (position + quaternion); here it is the task's own
  encoding (`fr3_6d`: 9 numbers, `fr3_5d`: 6), standardized.
* The paper's Section V evaluation (ground truth from TRAC-IK seeds, 250 samples per pose) is not
  replicated; `metrics.evaluate` scores against the exact posterior (`robot_ik`).

## Commands

```bash
python experiments/train_ikflow.py --problem fr3_6d --run-name fr3_6d_ikflow            # 3M parameters, 1M points, 50 epochs
python experiments/train_ikflow.py --problem fr3_6d --paper-size --run-name fr3_6d_ikflow_paper
python experiments/evaluate_ikflow.py fr3_6d_ikflow --split val                         # latent scale 1.0
python experiments/evaluate_ikflow.py fr3_6d_ikflow --split val --latent-scale 0.25     # the paper's setting
```

For scaling curves the knobs are `--param-budget` (model size) and `--n-train` (data); the number of
optimizer steps is `n_train // batch_size * epochs`, so keep it fixed or vary `--epochs` with it on
purpose. Scaling-law fits: Kaplan et al., arXiv:2001.08361.
