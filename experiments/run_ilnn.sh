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
# Variant sweep (new). SWEEP=1 runs the named variants below on top of the plain run; by default it
# only does kinematics with the ml loss (override with PROBLEMS / LOSSES):
#   SWEEP=1 bash experiments/run_ilnn.sh                              # all variants, seed 0
#   SWEEP=1 SEEDS="1 2" VARIANTS="base mix" bash experiments/run_ilnn.sh
#   VARIANTS="h96b16all" SEEDS="0 1 2" bash experiments/run_ilnn.sh   # VARIANTS alone also works
#   DRYRUN=1 SWEEP=1 bash experiments/run_ilnn.sh                     # print the commands, run nothing
#
#   variant     what it changes (vs. the plain run)
#   base        nothing: the plain run, same name as before (kin_ilnn_ml_s0), reused if it exists
#   mix         --perm-mode mixing                      every inter-block permutation mixes the two halves
#   lam         --lam-init-std 0.5 --no-wd-cell-scalars non-zero Lambda at init, no weight decay on Lambda/tau
#   h96b16      --hidden 96 --blocks 16                 narrower cells, twice the blocks (~0.33M params instead of 3M)
#   h96b16all   h96b16 + mix + lam
#   h427all     the 3M-parameter model + mix + lam
# Sweep runs also log mean |Lambda| and tau (--log-cell-stats).
#
# Extra train_ilnn.py flags can be appended after the script name. ALWAYS set TAG when you do, because the run
# name does not contain the flags: without a TAG the run reuses (and overwrites) the plain run's folder.
# Run names: <kin|bal>_ilnn_<loss>_s<seed>[_<variant>][_<TAG>]. A run whose experiments/runs/<name>/summary.json
# exists is skipped (FORCE=1 redoes it), and a failed run does not stop the others.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
SEEDS="${SEEDS:-0}"
DEVICE="${DEVICE:-auto}"            # training device
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"   # cpu keeps inference time comparable with the other models
TAG="${TAG:-}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"
SWEEP="${SWEEP:-0}"
EXTRA=("$@")
FAILED=()
RAN=()

# Variants. SWEEP=1 selects all of them; an explicit VARIANTS wins; neither = the plain run only (old behaviour).
ALL_VARIANTS="base mix lam h96b16 h96b16all h427all"
if [[ -n "${VARIANTS:-}" ]]; then
  SWEEP_MODE=1
elif [[ "$SWEEP" == "1" ]]; then
  VARIANTS="$ALL_VARIANTS"; SWEEP_MODE=1
else
  VARIANTS="base"; SWEEP_MODE=0
fi

# Problems / losses: the sweep was designed on kinematics with the likelihood loss.
if [[ "$SWEEP_MODE" == "1" ]]; then
  PROBLEMS="${PROBLEMS:-kinematics}"
  LOSSES="${LOSSES:-ml}"
else
  PROBLEMS="${PROBLEMS:-kinematics ballistics}"
  LOSSES="${LOSSES:-mmd ml}"
fi

# Flags of one variant (printed on one line; empty for base).
variant_flags() {
  case "$1" in
    base)      echo "" ;;
    mix)       echo "--perm-mode mixing" ;;
    lam)       echo "--lam-init-std 0.5 --no-wd-cell-scalars" ;;
    h96b16)    echo "--hidden 96 --blocks 16" ;;
    h96b16all) echo "--hidden 96 --blocks 16 --perm-mode mixing --lam-init-std 0.5 --no-wd-cell-scalars" ;;
    h427all)   echo "--perm-mode mixing --lam-init-std 0.5 --no-wd-cell-scalars" ;;
    *)         return 1 ;;
  esac
}
for v in $VARIANTS; do
  variant_flags "$v" > /dev/null || { echo "!! unknown variant '$v' (known: $ALL_VARIANTS)" >&2; exit 1; }
done

run() {  # DRYRUN=1 prints instead of executing
  if [[ "$DRYRUN" == "1" ]]; then echo "   [dry] $*"; else "$@"; fi
}

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
if [[ "$DRYRUN" == "1" ]]; then
  echo "   [dry] $PY -m pytest tests/test_ilnn.py -q"
else
  "$PY" -m pytest tests/test_ilnn.py -q || { echo "unit tests failed, stopping"; exit 1; }
fi

for seed in $SEEDS; do
  for problem in $PROBLEMS; do
    case "$problem" in
      kinematics) short=kin ;;
      ballistics) short=bal ;;
      *) short="$problem" ;;
    esac
    for loss in $LOSSES; do
      for variant in $VARIANTS; do
        read -r -a vflags <<< "$(variant_flags "$variant")"
        run="${short}_ilnn_${loss}_s${seed}"
        [[ "$variant" != "base" ]] && run="${run}_${variant}"
        run="${run}${TAG:+_$TAG}"
        [[ "${SMOKE:-0}" == "1" ]] && run="${run}_smoke"
        sweep_flags=()
        [[ "$SWEEP_MODE" == "1" ]] && sweep_flags+=(--log-cell-stats)

        if [[ -f "experiments/runs/$run/summary.json" && "$FORCE" != "1" ]]; then
          echo ">> skip   $run (trained; FORCE=1 to redo)"
        else
          echo ">> train  $run"
          run "$PY" experiments/train_ilnn.py --problem "$problem" --loss "$loss" --seed "$seed" \
            --device "$DEVICE" --run-name "$run" \
            "${sweep_flags[@]+"${sweep_flags[@]}"}" "${vflags[@]+"${vflags[@]}"}" \
            "${TRAIN_FLAGS[@]+"${TRAIN_FLAGS[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}" \
            || { echo "!! train failed: $run"; FAILED+=("$run"); continue; }
        fi

        if [[ -f "metrics/results/${problem}_${run}.json" && "$FORCE" != "1" ]]; then
          echo ">> skip   eval $run (result exists)"
        else
          echo ">> eval   $run"
          run "$PY" experiments/evaluate_ilnn.py "$run" --device "$EVAL_DEVICE" --seed "$seed" \
            || { echo "!! eval failed: $run"; FAILED+=("$run (eval)"); continue; }
        fi
        RAN+=("$run")
      done
    done
  done
done

# Recap of the sweep from the run summaries (provisional re-simulation error; the final numbers are the
# metrics/results/*.json from evaluate_ilnn.py).
if [[ "$SWEEP_MODE" == "1" && "$DRYRUN" != "1" && ${#RAN[@]} -gt 0 ]]; then
  echo ">> recap (provisional val re-simulation error, mean |Lambda|, train time)"
  "$PY" - "${RAN[@]}" <<'PYEOF'
import json, os, sys
for run in sys.argv[1:]:
    f = os.path.join('experiments', 'runs', run, 'summary.json')
    if not os.path.exists(f):
        continue
    s = json.load(open(f))
    print(f"   {run:34s} params {s['n_params']:>9,}  resim {s['err_resim_provisional']:.3e}"
          f"  |lam| {s.get('lam_abs_mean', float('nan')):.3f}  {s['train_seconds'] / 3600:.2f} h")
PYEOF
fi

echo ">> done. Results: metrics/results/*_ilnn_*.json"
[[ ${#FAILED[@]} -gt 0 ]] && echo "!! failed: ${FAILED[*]}"
exit 0