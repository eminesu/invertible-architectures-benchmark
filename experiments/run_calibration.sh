#!/usr/bin/env bash
# Overall confidence (calibration) test on every trained run that has a model.pt:
# SBC ranks, central-interval coverage, joint HPD coverage and, with GT=1, the
# spread-vs-ground-truth check. Model kinds: INN/cINN, ILNN, FMPE, conditional FMPE,
# MDN, autoencoder/cVAE/InvAuto, FFF/mFFF, FIF (BayesFlow NPSE runs are skipped).
#
#   bash experiments/run_calibration.sh                    # everything, with ground truth
#   GT=0 bash experiments/run_calibration.sh               # fast: no ground-truth check
#   ONLY=ilnn bash experiments/run_calibration.sh          # only runs whose name contains "ilnn"
#   ONLY="kin_" bash experiments/run_calibration.sh        # only kinematics runs
#   SOLVER=euler:32 bash experiments/run_calibration.sh    # FMPE / cFMPE solver, method:steps
#   DEVICE=cuda bash experiments/run_calibration.sh        # sample on GPU (mps on a Mac)
#   FORCE=1 bash experiments/run_calibration.sh            # redo runs that already have a result
#   DRY_RUN=1 bash experiments/run_calibration.sh          # list what would run
#
# Env vars [default]: PY [python] DEVICE [cpu] SOLVER [midpoint:16] GT [1] ONLY [] FORCE [0] DRY_RUN [0]
# Output: metrics/results/<benchmark>_<name>_calibration.{json,npz}, logs in calibration_logs/,
#         summary table on screen + metrics/results/calibration_summary.csv,
#         plots in metrics/results/calibration_plots/ (signed-error bars, rank histograms, coverage curves).
# FMPE / cFMPE results are named <run>_<method>x<steps>. A failed run never stops the others.

cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}; DEVICE=${DEVICE:-cpu}; SOLVER=${SOLVER:-midpoint:16}
GT=${GT:-1}; ONLY=${ONLY:-}; FORCE=${FORCE:-0}; DRY_RUN=${DRY_RUN:-0}
LOGDIR=calibration_logs; mkdir -p "$LOGDIR"
GTFLAG=""; [ "$GT" = 0 ] && GTFLAG="--no-gt"

# ---- 1. jobs: run <tab> kind <tab> problem, from experiments/evaluate_calibration.py list -------
mapfile -t LINES < <($PY experiments/evaluate_calibration.py list 2>"$LOGDIR/skipped.txt" | tr -d '\r')
JOBS=()
for l in "${LINES[@]}"; do
  IFS=$'\t' read -r r kind prob <<< "$l"
  [ -n "$ONLY" ] && [[ "$r" != *$ONLY* ]] && continue
  name=$r; extra=""
  if [ "$kind" = fmpe ] || [ "$kind" = cfmpe ]; then name="${r}_${SOLVER/:/x}"; extra="--solver $SOLVER"; fi
  JOBS+=("$kind|$r|$name|$prob|$extra")
done
echo "${#JOBS[@]} run(s) to score  (skipped ones are listed in $LOGDIR/skipped.txt)"
[ "${#JOBS[@]}" -eq 0 ] && { echo "nothing found: are the model.pt files in experiments/runs/?"; exit 1; }

# ---- 2. warm the ground-truth cache once per benchmark (slow the first time, cached after) -------
if [ "$GT" != 0 ] && [ "$DRY_RUN" != 1 ]; then
  for prob in $(for j in "${JOBS[@]}"; do IFS='|' read -r _ _ _ p _ <<< "$j"; echo "$p"; done | sort -u); do
    echo ">>> ground truth for $prob"
    $PY - "$prob" <<'PYEOF' || { echo "GROUND TRUTH FAILED for $prob; continuing without it"; GTFLAG="--no-gt"; }
import sys
from metrics.benchmarks import load_test_conditions
from metrics.ground_truth import ground_truth_posterior
y, _ = load_test_conditions(sys.argv[1])
ground_truth_posterior(sys.argv[1], y, verbose=True)
PYEOF
  done
fi

# ---- 3. score every run ------------------------------------------------------------------------
n=0; fail=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r kind r name prob extra <<< "$j"
  n=$((n+1))
  res="metrics/results/${prob}_${name}_calibration.json"
  if [ -f "$res" ] && [ "$FORCE" != 1 ]; then echo "[$n/${#JOBS[@]}] skip $name (done)"; continue; fi
  cmd="$PY experiments/evaluate_calibration.py $kind $r --name $name --device $DEVICE $extra $GTFLAG"
  echo "[$n/${#JOBS[@]}] >>> $cmd"
  [ "$DRY_RUN" = 1 ] && continue
  t0=$(date +%s)
  $cmd 2>&1 | tee "$LOGDIR/$name.log"
  [ -f "$res" ] || { echo "FAILED: $name (see $LOGDIR/$name.log)"; fail=$((fail+1)); }
  echo "    took $(( $(date +%s) - t0 )) s"
done
[ "$DRY_RUN" = 1 ] && exit 0
echo "calibration done: $((n)) run(s), $fail failed"

# ---- 4. summary table + CSV ----------------------------------------------------------------------
$PY - <<'PYEOF'
import csv, glob, json
rows = []
for f in sorted(glob.glob('metrics/results/*_calibration.json')):
    r = json.load(open(f)); m, j = r['marginal'], r['joint']; s = r.get('spread_vs_ground_truth', {})
    row = {'benchmark': r['benchmark'], 'model': r['model'],
           'marg_err': m['signed_error'], 'marg_verdict': m['verdict'], 'tails': m['tail_mass'],
           'joint_err': j['signed_error'], 'joint_verdict': j['verdict']}
    for p in m['per_dim']:
        row[f"x{p['dim']}_err"] = p['signed_error']
    row.update(spread_ratio=s.get('median_ratio', ''), spread_verdict=s.get('verdict', ''),
               frac_narrow=s.get('frac_too_narrow', ''), frac_wide=s.get('frac_too_wide', ''),
               nonfinite_frac=r.get('nonfinite_condition_fraction', 0.0))
    rows.append(row)
if not rows:
    raise SystemExit('no calibration results found')
cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open('metrics/results/calibration_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
g = lambda v, fmt: format(v, fmt) if isinstance(v, (int, float)) else '-'
print(f"\n{'bench':<11}{'model':<36}{'marg':>7} {'verdict':<16}{'joint':>7} {'verdict':<16}{'tails':>6}{'spread/GT':>10}{'narrow':>7}{'wide':>6}{'nonfin':>7}")
for r in rows:
    print(f"{r['benchmark']:<11}{r['model']:<36}{r['marg_err']:>+7.3f} {r['marg_verdict']:<16}{r['joint_err']:>+7.3f} "
          f"{r['joint_verdict']:<16}{r['tails']:>6.3f}{g(r['spread_ratio'], '.2f'):>10}"
          f"{g(r['frac_narrow'], '.0%'):>7}{g(r['frac_wide'], '.0%'):>6}{g(r['nonfinite_frac'], '.0%'):>7}")
print('\nmarg/joint = signed coverage error (<-0.03 over-confident, >+0.03 under-confident); tails ideal = 0.100;')
print('spread/GT = median model width / ground-truth width (<1 too narrow); narrow/wide = share of y* off by >25%;')
print('nonfin = share of y* dropped because a sample was NaN/inf. Saved metrics/results/calibration_summary.csv')
PYEOF

# ---- 5. plots of every finished run ----------------------------------------------------------------
echo ">>> plots"
$PY experiments/evaluate_calibration.py plots || echo "PLOTTING FAILED (the results are still in metrics/results/)"
echo "done. Look at metrics/results/calibration_plots/ and metrics/results/calibration_summary.csv"