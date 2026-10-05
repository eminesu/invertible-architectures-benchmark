#!/usr/bin/env bash
# Build-order step 9: sweep the number of mixture components K on one problem.
#   experiments/sweep_K.sh kinematics [extra train_mdn.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
problem=$1; shift
for K in 4 8 16 32; do
  .venv/bin/python experiments/train_mdn.py --problem "$problem" --components "$K" \
    --run-name "${problem}_K${K}_sweep" "$@"
done
.venv/bin/python experiments/summarize_runs.py
