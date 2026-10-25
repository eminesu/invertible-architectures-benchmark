#!/usr/bin/env bash
# Compare ILNN and cINN across different parameter budgets.
#
# Usage:
#   bash experiments/compare_ilnn_cinn_budgets.sh
#   DRY_RUN=1 bash experiments/compare_ilnn_cinn_budgets.sh  # print commands without running
#   EVAL=end bash experiments/compare_ilnn_cinn_budgets.sh   # train, then evaluate all

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}
EVAL=${EVAL:-0}
DRY_RUN=${DRY_RUN:-0}
SEEDS=${SEEDS:-0}
BUDGETS=${BUDGETS:-"30000 100000 300000"}

RUNS=()

run_model() {
  local name=$1
  local script=$2
  shift 2
  RUNS+=("kinematics:$name")
  
  if [ -f "experiments/runs/$name/summary.json" ]; then 
    echo "skip $name (done)"
    return
  fi
  
  local cmd="$PY $script --problem kinematics --run-name $name $*"
  echo ">>> $cmd"
  
  if [ "$DRY_RUN" = 0 ]; then
    $PY $script --problem kinematics --run-name "$name" "$@" || { echo "FAILED: $name"; return; }
    
    if [ "$EVAL" = 1 ]; then
      local eval_script="experiments/evaluate_${script##*_}"
      echo ">>> $PY $eval_script $name --device $EVAL_DEVICE"
      $PY "$eval_script" "$name" --device "$EVAL_DEVICE" || echo "EVAL FAILED: $name"
    fi
  fi
}

for seed in $SEEDS; do
  for budget in $BUDGETS; do
    tag="param_curve_b${budget}"
    
    run_model "kin_ilnn_ml_s${seed}_${tag}" "experiments/train_ilnn.py" \
      --loss ml \
      --seed "$seed" \
      --device "$DEVICE" \
      --param-budget "$budget"

    run_model "kin_cinn_s${seed}_${tag}" "experiments/train_inn.py" \
      --model cinn \
      --seed "$seed" \
      --device "$DEVICE" \
      --param-budget "$budget"
  done
done

echo "training done"

if [ "$EVAL" = end ] && [ "$DRY_RUN" = 0 ]; then
  for entry in "${RUNS[@]}"; do
    prob=${entry%%:*}
    name=${entry#*:}
    
    if [ -f "metrics/results/${prob}_${name}.json" ]; then 
      echo "skip eval $name (result exists)"
      continue
    fi
    
    eval_script="experiments/evaluate_inn.py"
    if [[ "$name" == *"ilnn"* ]]; then
        eval_script="experiments/evaluate_ilnn.py"
    fi
    
    echo ">>> $PY $eval_script $name --device $EVAL_DEVICE"
    $PY "$eval_script" "$name" --device "$EVAL_DEVICE" || echo "EVAL FAILED: $name"
  done
fi

echo "sweep done"