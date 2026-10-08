#!/usr/bin/env bash
# FMPE time-prior sweep: alpha in {0, 4}. kin_fmpe_s0 / bal_fmpe_s0 are effectively alpha = 1 (docs/fmpe.md).
# p(t) ∝ t^alpha, so alpha > 0 trains more on the posterior end t = 1 (docs/fmpe.md).
#   experiments/sweep_alpha.sh
# Training: kinematics on MPS and ballistics on CPU in parallel. Evaluation:
# one run at a time on the CPU, keep the machine otherwise idle.
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
mkdir -p experiments/runs/logs

train() {  # problem prefix device
  for a in 0 4; do
    [ -f "experiments/runs/${2}_fmpe_a${a}_s0/model.pt" ] || \
      $py experiments/train_fmpe.py --problem $1 --alpha $a --device $3 --run-name ${2}_fmpe_a${a}_s0 \
        > experiments/runs/logs/${2}_fmpe_a${a}_s0.out 2>&1
  done
}
train kinematics kin mps &
train ballistics bal cpu &
wait

for r in kin bal; do
  for a in 0 4; do
    $py experiments/evaluate_fmpe.py ${r}_fmpe_a${a}_s0 --solver euler:8 midpoint:4 midpoint:16 rk4:8 rk4:16
  done
done
