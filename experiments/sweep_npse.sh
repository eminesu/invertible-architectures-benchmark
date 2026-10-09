#!/usr/bin/env bash
# NPSE build-order step 7: sampling-step sweep, then model-size sweep.
#   experiments/sweep_npse.sh
# Expects experiments/runs/{kin,bal}_bf_npse_s0 from bayesflow_npse.py (3M).
# Comparable with the FMPE sweep (experiments/sweep_fmpe.sh on emine-fmpe): same
# metrics.evaluate path, CPU, 100 y* per sampling call, seed 0, and the same
# budgets of 8-64 network evaluations per sample. Evaluations run one at a time
# on the CPU and no training runs while anything is timed; keep the machine
# otherwise idle. Every result json records the load average before it started
# (measured right after the previous evaluation, so it includes that run's tail).
#
# Status: only part 1 without the adaptive run has been executed. The adaptive
# two_step_adaptive run and the model-size sweep (part 2) were stopped / not
# started, to focus on the comparison first (docs/npse.md, open items).
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
ev() { $py experiments/bayesflow_npse.py --evaluate-only --device cpu "$@"; }
idle() { while pgrep -f "train_|bayesflow_.*--device mps" > /dev/null; do sleep 60; done; }

# 1. Sampling steps (3M models). ODE: euler (1 NFE/step), fixed-step rk45 (6 NFE/step);
#    reverse SDE: euler_maruyama (1 NFE/step). euler:32 and euler_maruyama:128 are done.
idle
ev --problem kinematics --sampler euler:8 euler:16 euler:64 rk45:2 rk45:4 euler_maruyama:32 euler_maruyama:256
ev --problem ballistics --sampler euler:8 euler:16 euler:64 rk45:2 rk45:4 euler_maruyama:32
# Library default sampler (adaptive two_step_adaptive SDE), kinematics only, as for FMPE's tsit5.
ev --problem kinematics --sampler two_step_adaptive:adaptive

# 2. Model size: same recipe at smaller parameter budgets (training on MPS, nothing timed meanwhile),
#    then scored at the headline sampler, euler:32.
for p in kinematics ballistics; do
  for b in 300000 1000000; do
    r=${p:0:3}_bf_npse_p${b}_s0
    [ -f experiments/runs/$r/approximator.keras ] || \
      $py experiments/bayesflow_npse.py --problem $p --device mps --param-budget $b --run-name $r --sampler
  done
done
for p in kinematics ballistics; do
  for b in 300000 1000000; do
    ev --problem $p --run-name ${p:0:3}_bf_npse_p${b}_s0 --sampler euler:32
  done
done
