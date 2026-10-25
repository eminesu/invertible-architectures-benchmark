#!/usr/bin/env bash
# Test, train and evaluate the ILNN on both benchmark problems.
#
#   bash experiments/run_ilnn.sh                          # seed 0, both problems, both losses (mmd = paper, ml = likelihood)
#   LOSSES=ml SEEDS="1 2" bash experiments/run_ilnn.sh    # the cheap way to get more seeds (~1.8 h per seed)
#   SMOKE=1 bash experiments/run_ilnn.sh                  # quick check (20k samples, 2 epochs)
#   PROBLEMS="kinematics" bash experiments/run_ilnn.sh
#   DEVICE=cuda EVAL_DEVICE=cpu bash experiments/run_ilnn.sh
#   TAG=u2 LOSSES=ml bash experiments/run_ilnn.sh --unfolds 2   # ablation: TAG keeps it from overwriting kin_ilnn_ml_s0
#
# Extra train_ilnn.py flags can be appended after the script name. ALWAYS set TAG when you do, because the run
# name does not contain the flags: without a TAG the run reuses (and overwrites) the plain run's folder.
# Run names: kin_ilnn_<loss>_s<seed>[_<TAG>]. A run whose experiments/runs/<name>/summary.json exists is
# skipped (FORCE=1 redoes it), and a failed run does not stop the others.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SEEDS="${SEEDS:-0}"
PROBLEMS="${PROBLEMS:-kinematics ballistics}"
LOSSES="${LOSSES:-mmd ml}"
DEVICE="${DEVICE:-auto}"            # training device
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"   # cpu keeps inference time comparable with the other models
TAG="${TAG:-}"
FORCE="${FORCE:-0}"
EXTRA=("$@")
FAILED=()

TRAIN_FLAGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--n-train 20000 --n-val 2000 --epochs 2)
  echo ">> SMOKE mode: 20k train samples, 2 epochs (not a real result)"
fi
if [[ ${#EXTRA[@]} -gt 0 && -z "$TAG" ]]; then
  echo "!! extra flags given but TAG is empty: the run would overwrite the plain run's folder. Set TAG=<name>." >&2
  exit 1
fi

echo ">> unit tests"
"$PY" -m pytest tests/test_ilnn.py -q || { echo "unit tests failed, stopping"; exit 1; }

for seed in $SEEDS; do
  for problem in $PROBLEMS; do
    case "$problem" in
      kinematics) short=kin ;;
      ballistics) short=bal ;;
      *) short="$problem" ;;
    esac
    for loss in $LOSSES; do
      run="${short}_ilnn_${loss}_s${seed}${TAG:+_$TAG}"
      [[ "${SMOKE:-0}" == "1" ]] && run="${run}_smoke"

      if [[ -f "experiments/runs/$run/summary.json" && "$FORCE" != "1" ]]; then
        echo ">> skip   $run (trained; FORCE=1 to redo)"
      else
        echo ">> train  $run"
        "$PY" experiments/train_ilnn.py --problem "$problem" --loss "$loss" --seed "$seed" \
          --device "$DEVICE" --run-name "$run" "${TRAIN_FLAGS[@]+"${TRAIN_FLAGS[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}" \
          || { echo "!! train failed: $run"; FAILED+=("$run"); continue; }
      fi

      if [[ -f "metrics/results/${problem}_${run}.json" && "$FORCE" != "1" ]]; then
        echo ">> skip   eval $run (result exists)"
      else
        echo ">> eval   $run"
        "$PY" experiments/evaluate_ilnn.py "$run" --device "$EVAL_DEVICE" --seed "$seed" \
          || { echo "!! eval failed: $run"; FAILED+=("$run (eval)"); }
      fi
    done
  done
done

echo ">> done. Results: metrics/results/*_ilnn_*.json"
[[ ${#FAILED[@]} -gt 0 ]] && echo "!! failed: ${FAILED[*]}"
exit 0