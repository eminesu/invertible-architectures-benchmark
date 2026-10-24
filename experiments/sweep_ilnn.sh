#!/usr/bin/env bash
# Test, train and evaluate the ILNN on both benchmark problems.
#
#   bash experiments/run_ilnn.sh                  # full run, seed 0, both problems
#   SMOKE=1 bash experiments/run_ilnn.sh          # quick check (20k samples, 2 epochs)
#   SEEDS="0 1 2" bash experiments/run_ilnn.sh    # several seeds
#   PROBLEMS="kinematics" bash experiments/run_ilnn.sh
#   DEVICE=cuda EVAL_DEVICE=cpu bash experiments/run_ilnn.sh
#
# Extra train_ilnn.py flags can be appended: bash experiments/run_ilnn.sh --unfolds 6 --dt 0.2
# Run names: kin_ilnn_s<seed> / bal_ilnn_s<seed>; results land in metrics/results/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SEEDS="${SEEDS:-0}"
PROBLEMS="${PROBLEMS:-kinematics ballistics}"
DEVICE="${DEVICE:-auto}"            # training device
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"   # cpu keeps inference time comparable with the other models
EXTRA=("$@")

TRAIN_FLAGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  TRAIN_FLAGS+=(--n-train 20000 --n-val 2000 --epochs 2)
  echo ">> SMOKE mode: 20k train samples, 2 epochs (not a real result)"
fi

echo ">> unit tests"
"$PY" -m pytest tests/test_ilnn.py -q

for seed in $SEEDS; do
  for problem in $PROBLEMS; do
    case "$problem" in
      kinematics) short=kin ;;
      ballistics) short=bal ;;
      *) short="$problem" ;;
    esac
    run="${short}_ilnn_s${seed}"
    [[ "${SMOKE:-0}" == "1" ]] && run="${run}_smoke"

    echo ">> train  $run"
    "$PY" experiments/train_ilnn.py --problem "$problem" --seed "$seed" \
      --device "$DEVICE" --run-name "$run" "${TRAIN_FLAGS[@]}" "${EXTRA[@]}"

    echo ">> eval   $run"
    "$PY" experiments/evaluate_ilnn.py "$run" --device "$EVAL_DEVICE" --seed "$seed"
  done
done

echo ">> done. Results: metrics/results/*_ilnn_*.json"