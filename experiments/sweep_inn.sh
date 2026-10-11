#!/usr/bin/env bash
# Train (and optionally score) every INN / cINN case.
#
#   bash experiments/sweep_inn.sh                  # core: 2 problems x {INN ml, INN l2mmd, cINN}
#   bash experiments/sweep_inn.sh ablate           # core + sigma / alpha / latent-MMD / block-count variants (7 extra runs per problem)
#   bash experiments/sweep_inn.sh l2mmd            # core + only the INN (L2+MMD) variants: w_z 10/1000, marginal MMD (3 extra runs per problem)
#   bash experiments/sweep_inn.sh l2mmd_wide       # core + 8 INN (L2+MMD) variants: clamp 0.5/1.0 (alone and with w_z 10), w_z 10, w_z 1, marginal MMD, lr 5e-4
#   PROBLEMS=kinematics EVAL=end bash experiments/sweep_inn.sh   # train the 3 models, THEN score all of them
#   SEEDS="0 1 2" EVAL=1 bash experiments/sweep_inn.sh          # score each run right after its training
#   DRY_RUN=1 bash experiments/sweep_inn.sh ablate # print the commands only
#
# Environment variables (defaults in brackets):
#   PY [python]      interpreter           DEVICE [cuda]   cuda | cpu
#   SEEDS [0]        seeds to run          EVAL [0]        1 = score after each run, end = score all at the end
#   EVAL_DEVICE [cpu]      device for evaluate_inn.py; cpu keeps inference times comparable with the other
#                          models and the existing kin_cinn_s0 result (training stays on DEVICE)
#   PROBLEMS [kinematics ballistics]       DEQUANT [0]     1 = --dequantize-v0 on ballistics
#   EXTRA []         extra train_inn.py args for every run (e.g. "--epochs 15 --n-train 200000")
#   TAG []           suffix added to every run name (e.g. TAG=pilot), so cheap pilot runs never
#                    collide with, or get skipped because of, full-protocol runs
#   DRY_RUN [0]      1 = print, do not run
# A run whose experiments/runs/<name>/summary.json exists is skipped, so an interrupted
# sweep can simply be started again.

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}; DEVICE=${DEVICE:-cuda}; SEEDS=${SEEDS:-0}; EVAL=${EVAL:-0}
PROBLEMS=${PROBLEMS:-"kinematics ballistics"}; DEQUANT=${DEQUANT:-0}; EXTRA=${EXTRA:-}; DRY_RUN=${DRY_RUN:-0}
TAG=${TAG:-}; EVAL_DEVICE=${EVAL_DEVICE:-cpu}; RUNS=()
MODE=${1:-core}

run() {  # run <name> <train_inn.py args...>
  local name=$1${TAG:+_$TAG}; shift
  RUNS+=("$P:$name")
  if [ -f "experiments/runs/$name/summary.json" ]; then echo "skip $name (done)"; return; fi
  local cmd="$PY experiments/train_inn.py --device $DEVICE --run-name $name $* $EXTRA"
  echo ">>> $cmd"
  [ "$DRY_RUN" = 1 ] && return
  $PY experiments/train_inn.py --device "$DEVICE" --run-name "$name" "$@" $EXTRA || { echo "FAILED: $name"; return; }
  if [ "$EVAL" = 1 ]; then
    echo ">>> $PY experiments/evaluate_inn.py $name --device $EVAL_DEVICE"
    $PY experiments/evaluate_inn.py "$name" --device "$EVAL_DEVICE" || echo "EVAL FAILED: $name"
  fi
}

for P in $PROBLEMS; do
  T=${P:0:3}; DQ=""
  [ "$P" = ballistics ] && [ "$DEQUANT" = 1 ] && DQ="--dequantize-v0"
  for S in $SEEDS; do
    # --- the three paper rows ---
    run ${T}_inn_ml_s$S     --problem $P --model inn  --loss ml    --seed $S $DQ
    run ${T}_inn_l2mmd_s$S  --problem $P --model inn  --loss l2mmd --seed $S $DQ
    run ${T}_cinn_s$S       --problem $P --model cinn              --seed $S $DQ
   
    if [ "$MODE" = l2mmd_best ]; then 
      # The best candidates to fix the inverse blow-up while matching the paper
      run ${T}_inn_l2mmd_wz10_s$S --problem $P --model inn --loss l2mmd --w-z 10 --seed $S $DQ
      run ${T}_inn_l2mmd_clamp1.0_wz10_s$S --problem $P --model inn --loss l2mmd --clamp 1.0 --w-z 10 --seed $S $DQ
      continue
    fi
    
    if [ "$MODE" = l2mmd_wide ]; then   # only the INN (L2+MMD) row; the failing case is kinematics
      # Diagnosis (experiments/diagnose_inn_latent.py on kin_inn_l2mmd_s0): the latent is fine, but the inverse blows
      # up on some z draws (standardised |x| up to 3.6e5, re-simulation error up to 8e9), and the y fit is weaker than
      # the ML INN's (val y 0.0018 vs 0.00025). The clamp caps the per-transformation scale at e^clamp (default 2.0);
      # a lower w_z strengthens the y loss. Only the ratio w_z / w_y matters (Adam is scale-invariant).
      for C in 0.5 1.0; do
        run ${T}_inn_l2mmd_clamp${C}_s$S         --problem $P --model inn --loss l2mmd --clamp $C --seed $S $DQ
        run ${T}_inn_l2mmd_clamp${C}_wz10_s$S    --problem $P --model inn --loss l2mmd --clamp $C --w-z 10 --seed $S $DQ
      done
      run ${T}_inn_l2mmd_wz10_s$S  --problem $P --model inn --loss l2mmd --w-z 10 --seed $S $DQ
      run ${T}_inn_l2mmd_wz1_s$S   --problem $P --model inn --loss l2mmd --w-z 1 --seed $S $DQ
      run ${T}_inn_l2mmd_marg_s$S  --problem $P --model inn --loss l2mmd --latent-mmd marginal --seed $S $DQ
      run ${T}_inn_l2mmd_lr5e-4_s$S --problem $P --model inn --loss l2mmd --lr 5e-4 --seed $S $DQ
      continue
    fi
    [ "$MODE" = ablate ] || [ "$MODE" = l2mmd ] || continue
    # --- INN (Eq. 4): std of y around y_gt ---
    if [ "$MODE" = ablate ]; then
      for SIG in 0.01 0.1; do
        run ${T}_inn_ml_sig${SIG}_s$S --problem $P --model inn --loss ml --sigma-y $SIG --seed $S $DQ
      done
    fi
    # --- INN (Eq. 3): MMD weight alpha, and joint [y,z] vs marginal z MMD ---
    
    for WZ in 10 1000; do
      run ${T}_inn_l2mmd_wz${WZ}_s$S --problem $P --model inn --loss l2mmd --w-z $WZ --seed $S $DQ
    done
    run ${T}_inn_l2mmd_marg_s$S --problem $P --model inn --loss l2mmd --latent-mmd marginal --seed $S $DQ
    # --- architecture: depth of the flow (same ~3M budget, width re-fitted) ---
    if [ "$MODE" = ablate ]; then
      for B in 6 12; do
        run ${T}_cinn_b${B}_s$S --problem $P --model cinn --blocks $B --seed $S $DQ
      done
    fi
  done
done
echo "training done"

if [ "$EVAL" = end ]; then
  for entry in "${RUNS[@]}"; do
    prob=${entry%%:*}; name=${entry#*:}
    if [ -f "metrics/results/${prob}_${name}.json" ]; then echo "skip eval $name (result exists)"; continue; fi
    echo ">>> $PY experiments/evaluate_inn.py $name --device $EVAL_DEVICE"
    [ "$DRY_RUN" = 1 ] || $PY experiments/evaluate_inn.py "$name" --device "$EVAL_DEVICE" || echo "EVAL FAILED: $name"
  done
fi

if [ "$EVAL" != 0 ] && [ "$DRY_RUN" != 1 ]; then
  $PY - "${RUNS[@]}" <<'PYEOF'
import json, re, sys
from pathlib import Path
# Paper, Tables 1-2: (Err_post, Err_resim) per row. Ballistics resim is compared on the clamped mean.
PAPER = {'kinematics': {'inn_ml': (0.025, 0.015), 'inn_l2mmd': (0.017, 0.086), 'cinn': (0.015, 0.008)},
         'ballistics': {'inn_ml': (0.047, 0.019), 'inn_l2mmd': (0.060, 3.668), 'cinn': (0.047, 0.437)}}
def fmt(v):
    return f'{v:.2e}' if 0 < abs(v) < 1e-3 else f'{v:,.4f}' if abs(v) >= 1000 else f'{v:.4f}'
print(f"\n{'run':<30}{'post':>9}{'(biased)':>10}{'paper':>8}{'resim':>14}{'paper':>8}{'ms/y*':>9}  vs paper (lower is better)")
for line in sys.argv[1:]:
    prob, name = line.split(':', 1)
    f = Path('metrics/results') / f'{prob}_{name}.json'
    if not f.exists():
        print(f'{name:<30} (no result)'); continue
    r = json.loads(f.read_text())
    key = 'inn_l2mmd' if 'l2mmd' in name else 'cinn' if 'cinn' in name else 'inn_ml'
    pp, pr = PAPER[prob][key]
    post, post_b = r['err_post']['mean'], r['err_post_biased']['mean']
    resim = r['err_resim']['mean_clamped'] if prob == 'ballistics' else r['err_resim']['mean']
    verdict = ('post ' + ('better' if post < pp else 'worse') + ', resim ' + ('better' if resim < pr else 'worse'))
    print(f"{name:<30}{fmt(post):>9}{fmt(post_b):>10}{pp:>8.3f}{fmt(resim):>14}{pr:>8.3f}{r['inference_ms_per_condition']:>9.1f}  {verdict}")
print('\nNote: paper Err_post may be a biased estimator (compare the "(biased)" column); inference times are not')
print('comparable with the paper (different hardware) - compare models run on the same device only.')
PYEOF
fi
echo "sweep done"