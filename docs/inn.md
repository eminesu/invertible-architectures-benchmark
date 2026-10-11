# INN and conditional INN (Kruse et al. 2021, arXiv:2101.10763)

Code: `models/inn.py`, `experiments/train_inn.py`, `experiments/evaluate_inn.py`, `tests/test_inn.py`.

| Paper row | Command |
|---|---|
| INN (Eq. 4, max. likelihood) | `python experiments/train_inn.py --problem kinematics --model inn --loss ml` |
| INN (L2 + MMD, Eq. 3) | `python experiments/train_inn.py --problem kinematics --model inn --loss l2mmd` |
| cINN (Eq. 5) | `python experiments/train_inn.py --problem kinematics --model cinn` |

Use `--problem ballistics` for the second benchmark. Training writes `experiments/runs/<run>/`; score a run with
`python experiments/evaluate_inn.py <run>` (shared `metrics.evaluate`, writes `metrics/results/`).
`experiments/summarize_fmpe.py` (branch `emine-fmpe`) tables those results next to the other models.
Note `experiments/summarize_runs.py` is MDN-only (reads `components`) and will not accept these runs.

Reused from the repo: `data.toy_data`, `metrics.{benchmarks,evaluate,mmd}`, `models.mdn.count_parameters`,
`experiments.train_mdn.pick_device`, and the training protocol/seeds of `train_mdn.py`.
Copied from `emine-autoencoder` (same signatures; import them from there once it is merged):
`mlp`, `evaluate_losses`, `provisional_resim`.

## Not specified by the paper (all placeholders, all flags)
subnet depth/width (`--layers 3`, width fitted to `--param-budget 3e6`), `--blocks 8`, `--clamp 2.0`,
`--sigma-y 0.05` (Eq. 4), `--w-y 1`, `--w-z 100` (alpha, Eq. 3), optimizer and schedule (same as MDN/AE).
`--latent-mmd joint` (as `models/autoencoder.py`) vs `marginal` (literal Eq. 3).
`--dequantize-v0` (ballistics, off by default) is a team decision, see `docs/metrics.md`.

## Known behaviour
Reduced-scale kinematics runs: INN-ML and cINN train stably. INN L2+MMD was unstable early
(tiny MMD value but heavy-tailed z gives wild inverse samples); expect to tune `--w-z`,
`--latent-mmd` or the kernel. No paper-comparable numbers exist yet; run the full protocol.
