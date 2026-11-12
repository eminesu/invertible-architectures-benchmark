#!/usr/bin/env bash
# A staged search for a strong IKFlow baseline on the FR3 tasks, then scaling to larger models and
# data, then the final runs with several seeds.
#
#   bash experiments/run_ikflow_baseline.sh                  # search scale final table
#   bash experiments/run_ikflow_baseline.sh search table
#   QUICK=1 bash experiments/run_ikflow_baseline.sh          # tiny budgets: check that the pipeline runs end to end
#   DRY=1 bash experiments/run_ikflow_baseline.sh search     # only print the commands
#   bash experiments/run_ikflow_baseline.sh test             # once, at the very end: score the final runs on test
#
# Stages
#   search  coordinate search on $TASK (default fr3_6d, the harder one) at a small budget. Groups, in order:
#           lr, optimizer, schedule, EMA, SoftFlow noise, clamp, activation, grad-clip, batch size, blocks,
#           width, layers. Every group trains one run per candidate value around the current best
#           configuration (about 32 trials per pass), scores them on val with the benchmark metrics and keeps
#           the best one if it wins by more than $MIN_GAIN (relative), otherwise keeps the old value.
#           A 12-factor full grid would be thousands of runs; this covers every factor once per pass,
#           PASSES=2 makes a second pass that revisits them around the new best. Interactions beyond that
#           are not searched.
#   scale   the best configuration at several model sizes ($SCALE_BUDGETS parameters; 51M is the paper's
#           size) with more data and steps; picks the best size on val.
#   final   trains the chosen configuration for every task in $FINAL_TASKS and seed in $SEEDS with the
#           largest budget ($FINAL_NTRAIN points, $FINAL_STEPS steps), scores them on val at the latent
#           scales $LATENT_SCALES (1.0 = the model's own distribution, 0.25 = the paper's sub-sampling).
#   test    scores the final runs on the test split at latent scales 1.0 and 0.25. Run it once, last.
#   table   comparison of what has been scored.
#
# Everything is resumable: a trial whose experiments/runs/<run>/summary.json and metrics/results/*_val.json
# exist is skipped, and the state of the search is in experiments/ikflow_search/<task>/.
# Select on val with SELECT=err_post (default; the benchmark's metric), coverage, pos or resim. The search
# never reads the test split. A choice made on one seed at a small budget can be wrong for the final
# budget: the scale stage re-checks the model size, the final stage reports the spread over seeds.

set -euo pipefail
cd "$(dirname "$0")/.."

QUICK="${QUICK:-0}"
if [[ "$QUICK" == "1" ]]; then
    : "${SCREEN_NTRAIN:=20000}" "${SCREEN_STEPS:=300}" "${SCALE_NTRAIN:=20000}" "${SCALE_STEPS:=300}"
    : "${FINAL_NTRAIN:=20000}" "${FINAL_STEPS:=300}" "${SEEDS:=0}" "${GROUP_LIMIT:=1}" "${SCALE_BUDGETS:=300000 1000000}"
fi

PY="${PY:-python}"
DRY="${DRY:-0}"
TASK="${TASK:-fr3_6d}"
FINAL_TASKS="${FINAL_TASKS:-fr3_5d fr3_6d}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
SELECT="${SELECT:-err_post}"
MIN_GAIN="${MIN_GAIN:-0.03}"
PASSES="${PASSES:-1}"
GROUP_LIMIT="${GROUP_LIMIT:-0}"          # 0 = every candidate of a group
SCREEN_NTRAIN="${SCREEN_NTRAIN:-200000}"
SCREEN_STEPS="${SCREEN_STEPS:-30000}"
SCREEN_BUDGET="${SCREEN_BUDGET:-3000000}"
SCALE_BUDGETS="${SCALE_BUDGETS:-3000000 10000000 30000000 51000000}"
SCALE_NTRAIN="${SCALE_NTRAIN:-1000000}"
SCALE_STEPS="${SCALE_STEPS:-200000}"
FINAL_NTRAIN="${FINAL_NTRAIN:-2500000}"
FINAL_STEPS="${FINAL_STEPS:-600000}"
SEEDS="${SEEDS:-0 1 2}"
LATENT_SCALES="${LATENT_SCALES:-0.25 0.5 0.75 1.0}"
RUNS=experiments/runs
RESULTS=metrics/results

run_cmd() { if [[ "$DRY" == "1" ]]; then echo "[dry] $*"; else "$@"; fi; }

# ls_suffix <scale>: the result-name suffix evaluate_ikflow.py uses ("" for 1.0, _ls0.25, ...)
ls_suffix() { if [[ "$1" == "1" || "$1" == "1.0" ]]; then echo ""; else echo "_ls$1"; fi; }

# train_if_needed <task> <run> <train arguments...>
train_if_needed() {
    local task=$1 run=$2; shift 2
    if [[ -f "$RUNS/$run/summary.json" ]]; then echo "[skip train] $run"; return 0; fi
    echo "[train] $run"
    # shellcheck disable=SC2068
    run_cmd "$PY" experiments/train_ikflow.py --problem "$task" --run-name "$run" $@
}

# eval_if_needed <task> <run> <split> <latent scale>
eval_if_needed() {
    local task=$1 run=$2 split=$3 ls=$4 suffix=""
    if [[ "$split" != "test" ]]; then suffix="_$split"; fi
    if [[ -f "$RESULTS/${task}_${run}$(ls_suffix "$ls")${suffix}.json" ]]; then echo "[skip eval] $run ($split, scale $ls)"; return 0; fi
    echo "[eval ] $run ($split, latent scale $ls)"
    run_cmd "$PY" experiments/evaluate_ikflow.py "$run" --split "$split" --latent-scale "$ls" --device "$EVAL_DEVICE"
}

trial() {   # trial <task> <run> <train arguments...>: train, then score on val at latent scale 1.0
    local task=$1 run=$2; shift 2
    train_if_needed "$task" "$run" "$@"
    eval_if_needed "$task" "$run" val 1.0
}

state_dir() { echo "experiments/ikflow_search/$TASK"; }
load_state() {
    local sd; sd=$(state_dir)
    BEST_FLAGS=""; REF_RUN=""
    if [[ -f "$sd/best_flags" ]]; then BEST_FLAGS=$(cat "$sd/best_flags"); fi
    if [[ -f "$sd/ref_run" ]]; then REF_RUN=$(cat "$sd/ref_run"); fi
    return 0
}
save_state() {
    local sd; sd=$(state_dir)
    mkdir -p "$sd"
    printf '%s' "$BEST_FLAGS" > "$sd/best_flags"
    printf '%s' "$REF_RUN" > "$sd/ref_run"
}

pick() {   # pick <reference run> <candidate runs...>: the winner on val
    local ref=$1; shift
    if [[ "$DRY" == "1" ]]; then echo "$ref"; return 0; fi
    "$PY" experiments/ikflow_pick.py pick --task "$TASK" --select "$SELECT" --min-gain "$MIN_GAIN" --ref "$ref" "$@"
}

# group <name> <tag|flags>...  (PASS is set by the caller)
group() {
    local name=$1; shift
    local sd; sd=$(state_dir); mkdir -p "$sd"
    local marker="$sd/done_p${PASS}_${name}"
    if [[ -f "$marker" ]]; then echo "[skip group] $name (pass $PASS)"; return 0; fi
    load_state
    local screen="--n-train $SCREEN_NTRAIN --steps $SCREEN_STEPS --param-budget $SCREEN_BUDGET --n-val 5000"
    if [[ -z "$REF_RUN" ]]; then
        REF_RUN="${TASK}_ikf_ref"
        # shellcheck disable=SC2086
        trial "$TASK" "$REF_RUN" $screen $BEST_FLAGS
        save_state
    fi
    local cands=("$REF_RUN") cflags=("") n=0 spec tag flags run
    for spec in "$@"; do
        tag=${spec%%|*}; flags=${spec#*|}
        if [[ " $BEST_FLAGS " == *" $flags "* ]]; then continue; fi    # already the current setting
        if [[ "$GROUP_LIMIT" != "0" && $n -ge $GROUP_LIMIT ]]; then break; fi
        n=$((n + 1))
        run="${TASK}_ikf_p${PASS}_${name}_${tag}"
        # shellcheck disable=SC2086
        trial "$TASK" "$run" $screen $BEST_FLAGS $flags
        cands+=("$run"); cflags+=("$flags")
    done
    local win i
    win=$(pick "$REF_RUN" "${cands[@]}")
    if [[ "$win" != "$REF_RUN" ]]; then
        for i in "${!cands[@]}"; do
            if [[ "${cands[$i]}" == "$win" ]]; then BEST_FLAGS="$BEST_FLAGS ${cflags[$i]}"; fi
        done
        REF_RUN=$win
        echo "[pick] $name -> $win; configuration: $BEST_FLAGS"
    else
        echo "[pick] $name -> keep the current configuration"
    fi
    save_state
    touch "$marker"
}

stage_search() {
    local PASS
    for PASS in $(seq 1 "$PASSES"); do
        group lr       "lr1e-4|--lr 1e-4" "lr2e-4|--lr 2e-4" "lr1e-3|--lr 1e-3" "lr2e-3|--lr 2e-3"
        group optim    "radam|--optimizer radam" "adam|--optimizer adam"
        group sched    "cos|--schedule cosine"
        group ema      "ema999|--ema 0.999" "ema9999|--ema 0.9999"
        group softflow "sf0|--softflow-scale 0" "sf1e-4|--softflow-scale 1e-4" "sf1e-2|--softflow-scale 1e-2" "sf3e-2|--softflow-scale 3e-2"
        group clamp    "cl1|--clamp 1" "cl3|--clamp 3" "cl5|--clamp 5"
        group act      "relu|--activation relu" "silu|--activation silu"
        group gclip    "gc0|--grad-clip 0" "gc1|--grad-clip 1"
        group batch    "bs64|--batch-size 64" "bs256|--batch-size 256" "bs512|--batch-size 512"
        group blocks   "b8|--blocks 8" "b16|--blocks 16" "b24|--blocks 24"
        group width    "w12|--width 12" "w16|--width 16" "w24|--width 24"
        group layers   "l2|--layers 2" "l4|--layers 4"
    done
    load_state
    echo "best configuration on $TASK: ${BEST_FLAGS:-<the defaults>}"
}

stage_scale() {
    load_state
    local sd; sd=$(state_dir)
    local cands=() b run first=""
    for b in $SCALE_BUDGETS; do
        run="${TASK}_ikf_scale_${b}"
        # shellcheck disable=SC2086
        trial "$TASK" "$run" $BEST_FLAGS --n-train "$SCALE_NTRAIN" --steps "$SCALE_STEPS" --param-budget "$b" --n-val 5000
        cands+=("$run")
        if [[ -z "$first" ]]; then first=$run; fi
    done
    local win; win=$(pick "$first" "${cands[@]}")
    b=${win##*_}
    if [[ "$DRY" == "1" ]]; then b=${SCALE_BUDGETS%% *}; fi
    printf '%s' "--param-budget $b" > "$sd/scale_flags"
    echo "[pick] model size -> $b parameters ($win)"
}

final_flags() {
    local sd; sd=$(state_dir)
    load_state
    local sf="--param-budget $SCREEN_BUDGET"
    if [[ -f "$sd/scale_flags" ]]; then sf=$(cat "$sd/scale_flags"); fi
    echo "$BEST_FLAGS $sf --n-train $FINAL_NTRAIN --steps $FINAL_STEPS"
}

stage_final() {
    local flags t s run ls; flags=$(final_flags)
    for t in $FINAL_TASKS; do
        for s in $SEEDS; do
            run="${t}_ikflow_best_s${s}"
            # shellcheck disable=SC2086
            train_if_needed "$t" "$run" $flags --seed "$s"
            for ls in $LATENT_SCALES; do eval_if_needed "$t" "$run" val "$ls"; done
        done
    done
}

stage_test() {
    echo "scoring the final runs on test (the ground truth of the test poses is generated on first use: slow without a cache)"
    local t s run
    for t in $FINAL_TASKS; do
        for s in $SEEDS; do
            run="${t}_ikflow_best_s${s}"
            eval_if_needed "$t" "$run" test 1.0
            eval_if_needed "$t" "$run" test 0.25
        done
    done
    "$PY" experiments/ikflow_pick.py table --split test --match ikflow_best --tasks "$FINAL_TASKS"
}

stage_table() {
    echo "-- search and scale runs on $TASK (val) --"
    "$PY" experiments/ikflow_pick.py table --split val --match ikf_ --tasks "$TASK"
    echo "-- final runs (val; _ls<scale> = latent scale, none = 1.0) --"
    "$PY" experiments/ikflow_pick.py table --split val --match ikflow_best --tasks "$FINAL_TASKS"
}

stages=("$@")
if [[ ${#stages[@]} -eq 0 ]]; then stages=(search scale final table); fi
for st in "${stages[@]}"; do
    case "$st" in
        search|scale|final|test|table) "stage_$st" ;;
        *) echo "unknown stage '$st' (search scale final test table)" >&2; exit 1 ;;
    esac
done
