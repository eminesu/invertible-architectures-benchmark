#!/usr/bin/env bash
# Score every trained FR3 run on the TEST split: 1000 poses x 1000 samples, the setting of the 2-D
# benchmarks (the tables so far are the 200-pose validation split), then draw the four-panel table.
#
#   bash experiments/run_fr3_test_eval.sh
#   EVAL_DEVICE=cuda bash experiments/run_fr3_test_eval.sh
#
# The first evaluation of each task generates the exact solution sets of the 1000 test poses (not in git;
# about 5x the 200-pose validation set, so roughly 45 minutes per task on the machine that did the
# validation one; it is cached afterwards). Runs are scored one after another, so the Time column is
# comparable between models. Finished results are skipped, so the script can be restarted.
# Every model is scored once; nothing here selects anything on test.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
DEVICE="${EVAL_DEVICE:-cpu}"
RUNS=experiments/runs
RESULTS=metrics/results

# score <task> <run> <evaluate script> <result-name suffix> [extra evaluate arguments...]
score() {
    local task=$1 run=$2 script=$3 suffix=$4; shift 4
    if [[ ! -f "$RUNS/$run/model.pt" ]]; then echo "[skip] $run: no trained model in $RUNS/$run"; return 0; fi
    if [[ -f "$RESULTS/${task}_${run}${suffix}.json" ]]; then echo "[skip] ${run}${suffix}: already scored on test"; return 0; fi
    echo "[test ] ${run}${suffix}"
    "$PY" "$script" "$run" --device "$DEVICE" "$@"
}

for t in fr3_5d fr3_6d; do
    score "$t" "${t}_cinn"   experiments/evaluate_inn.py     ""
    score "$t" "${t}_cilnn"  experiments/evaluate_ilnn.py    ""
    score "$t" "${t}_ikflow" experiments/evaluate_ikflow.py  ""                       # latent scale 1.0
    for ls in 0.75 0.5 0.25; do
        score "$t" "${t}_ikflow" experiments/evaluate_ikflow.py "_ls${ls}" --latent-scale "$ls"
    done
done
# unconditional ILNN: only fr3_5d can be built; the padded 6-D run did not train, so it is not scored
score fr3_5d fr3_5d_base experiments/evaluate_ilnn.py ""
score fr3_5d fr3_5d_pad3 experiments/evaluate_ilnn.py ""

"$PY" experiments/make_fr3_table_png.py --split test --out fr3_test_tables.png