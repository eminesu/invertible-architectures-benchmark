#!/usr/bin/env bash
# Diffusion (NPSE) vs FMPE sweep, docs/comparison.md "Planned sweep":
#   1. model size  300k / 1M params, seed 0     (diffusion + hand-rolled FMPE)
#   2. seeds       1, 2 at 3M                    (diffusion + hand-rolled FMPE)
#   3. time prior  alpha = 0, 4 at 3M, seed 0    (hand-rolled FMPE; the headline runs are alpha = 1)
#   4. NFE grid    4 / 16 / 64 NFE for the existing 3M seed-0 runs
#
#   nohup experiments/sweep_compare.sh > experiments/runs/logs/sweep_compare.out 2>&1 &
#
# Phase A trains everything on MPS (diffusion and FMPE queues in parallel).
# Phase B evaluates one run at a time on the CPU, after all training has ended,
# so the inference times compare with each other; keep the machine otherwise idle.
# FMPE size / seed runs use alpha = 1 like the headline kin_fmpe_s0 / bal_fmpe_s0.
# Resumable: finished trainings and existing result files are skipped. A failed
# step prints "FAILED ..." and the sweep goes on; its log is in experiments/runs/logs.
set -euo pipefail
cd "$(dirname "$0")/.."
py=.venv/bin/python
logs=experiments/runs/logs
mkdir -p "$logs"

PROBLEMS="kinematics ballistics"
BUDGETS="300000 1000000"
SEEDS="1 2"
ALPHAS="0 4"

# ---- Phase A: training ------------------------------------------------------

train_npse() {  # problem run seed budget
  [ -f "experiments/runs/$2/approximator.keras" ] && return
  echo "$(date '+%F %T') train $2"
  $py experiments/bayesflow_npse.py --problem "$1" --device mps --seed "$3" --param-budget "$4" \
    --run-name "$2" --sampler > "$logs/$2.out" 2>&1 || echo "FAILED train $2"
}

train_fmpe() {  # problem run seed budget alpha
  [ -f "experiments/runs/$2/model.pt" ] && return
  echo "$(date '+%F %T') train $2"
  $py experiments/train_fmpe.py --problem "$1" --device mps --seed "$3" --param-budget "$4" \
    --alpha "$5" --run-name "$2" > "$logs/$2.out" 2>&1 || echo "FAILED train $2"
}

npse_queue() {
  for p in $PROBLEMS; do
    for b in $BUDGETS; do train_npse "$p" "${p:0:3}_bf_npse_p${b}_s0" 0 "$b"; done
    for s in $SEEDS; do train_npse "$p" "${p:0:3}_bf_npse_s$s" "$s" 3000000; done
  done
}

fmpe_queue() {
  for p in $PROBLEMS; do
    for b in $BUDGETS; do train_fmpe "$p" "${p:0:3}_fmpe_p${b}_s0" 0 "$b" 1; done
    for s in $SEEDS; do train_fmpe "$p" "${p:0:3}_fmpe_s$s" "$s" 3000000 1; done
    for a in $ALPHAS; do train_fmpe "$p" "${p:0:3}_fmpe_a${a}_s0" 0 3000000 "$a"; done
  done
}

npse_queue & fmpe_queue & wait
echo "$(date '+%F %T') training done"

# ---- Phase B: evaluation, one at a time on the CPU --------------------------

todo() {  # benchmark run spec... -> the specs without a result file
  local bench=$1 run=$2; shift 2
  for spec in "$@"; do
    [ -f "metrics/results/${bench}_${run}_${spec/:/x}.json" ] || echo "$spec"
  done
}

eval_npse() {  # problem run seed spec...
  local p=$1 r=$2 s=$3; shift 3
  local specs; specs=$(todo "$p" "$r" "$@")
  [ -z "$specs" ] && return
  echo "$(date '+%F %T') eval $r:" $specs
  $py experiments/bayesflow_npse.py --problem "$p" --evaluate-only --device cpu --seed "$s" --run-name "$r" \
    --sampler $specs >> "$logs/${r}_eval.out" 2>&1 || echo "FAILED eval $r"
}

eval_fmpe() {  # problem run spec...
  local p=$1 r=$2; shift 2
  local specs; specs=$(todo "$p" "$r" "$@")
  [ -z "$specs" ] && return
  echo "$(date '+%F %T') eval $r:" $specs
  $py experiments/evaluate_fmpe.py "$r" --device cpu --solver $specs >> "$logs/${r}_eval.out" 2>&1 || echo "FAILED eval $r"
}

for p in $PROBLEMS; do
  q=${p:0:3}
  # 4. NFE grid of the existing 3M seed-0 runs (cheap, first).
  eval_npse "$p" ${q}_bf_npse_s0 0 euler:4
  eval_fmpe "$p" ${q}_fmpe_s0 euler:4 euler:16 euler:64 midpoint:2 midpoint:8 midpoint:32
  # 2. Seeds: headline samplers at 8 and 32 NFE.
  for s in $SEEDS; do
    eval_npse "$p" ${q}_bf_npse_s$s "$s" euler:8 euler:32
    eval_fmpe "$p" ${q}_fmpe_s$s midpoint:4 midpoint:16
  done
  # 3. FMPE time prior.
  for a in $ALPHAS; do
    eval_fmpe "$p" ${q}_fmpe_a${a}_s0 euler:8 euler:16 euler:32 midpoint:4 midpoint:8 midpoint:16
  done
  # 1. Model size: 8 / 16 / 32 / 64 NFE.
  for b in $BUDGETS; do
    eval_npse "$p" ${q}_bf_npse_p${b}_s0 0 euler:8 euler:16 euler:32 euler:64
    eval_fmpe "$p" ${q}_fmpe_p${b}_s0 midpoint:4 midpoint:8 midpoint:16 midpoint:32
  done
done
echo "$(date '+%F %T') sweep done"
