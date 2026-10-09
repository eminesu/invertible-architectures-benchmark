#!/usr/bin/env bash
# FMPE build-order steps 5-7: score the trained FMPE runs over an NFE sweep and
# train + score the BayesFlow cross-check.
#   experiments/sweep_fmpe.sh
# Expects experiments/runs/{kin,bal}_fmpe_s0 from train_fmpe.py. Evaluations run
# one at a time on the CPU so the inference times are comparable with each other
# and with the other models; keep the machine otherwise idle while this runs.
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python

# BayesFlow training (MPS, timing irrelevant).
for p in kinematics ballistics; do
  [ -f "experiments/runs/${p:0:3}_bf_fmpe_s0/approximator.keras" ] || \
    $py experiments/bayesflow_fmpe.py --problem $p --device mps --solver
done

while pgrep -f train_fmpe.py > /dev/null; do sleep 60; done

# Same ODE budgets for every solver: 8, 16, 32, 64 NFE per sample.
for r in kin bal; do
  $py experiments/evaluate_fmpe.py ${r}_fmpe_s0 --solver euler:8 euler:32 midpoint:4 midpoint:16 rk4:2 rk4:4 rk4:8 rk4:16
done
# BayesFlow's default solver (adaptive tsit5) took 331 NFE and 1.7 h of CPU on
# kinematics, so it is run there only.
$py experiments/bayesflow_fmpe.py --problem kinematics --evaluate-only --solver euler:8 euler:32 rk45:2 rk45:4 tsit5:adaptive
$py experiments/bayesflow_fmpe.py --problem ballistics --evaluate-only --solver euler:8 euler:32 rk45:2 rk45:4
# Reference: a fine RK4 grid (256 NFE) on MPS, for accuracy only (its time is not comparable).
for r in kin bal; do
  $py experiments/evaluate_fmpe.py ${r}_fmpe_s0 --solver rk4:64 --device mps --name ${r}_fmpe_s0_mps
done
