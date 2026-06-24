#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/ydh/桌面/zhangyihao}"
CODE="${CODE:-$BASE/deeponet_macro_operator_fresh}"
DATA_ROOT="${DATA_ROOT:-$BASE/outputs/query_point_v3_css8_standard_operator/multi_case_min10}"
COMPACT_LIST="${COMPACT_LIST:-$DATA_ROOT/v3_css8_standard_operator_compact_list.txt}"
OUT_DIR="${OUT_DIR:-$BASE/run_logs_query_point_v3/v3_le_primary_formal_long_$(date +%Y%m%d_%H%M%S)}"
PRESET="${PRESET:-formal-long}"
FILTER_CASE="${FILTER_CASE:-}"
FILTER_FRAME="${FILTER_FRAME:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260624}"
USE_PLUS_SAMPLES="${USE_PLUS_SAMPLES:-1}"
MODEL_STYLE="${MODEL_STYLE:-fe-state-linear-residual}"
STATE_BASELINE_RANK="${STATE_BASELINE_RANK:-}"
RESIDUAL_JACOBIAN_WEIGHT="${RESIDUAL_JACOBIAN_WEIGHT:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
NPROC="${NPROC:-3}"
EXTRA_ARGS=("$@")

if [[ ! -f "$COMPACT_LIST" ]]; then
  echo "Missing COMPACT_LIST: $COMPACT_LIST" >&2
  exit 2
fi

export PYTHONPATH="$CODE/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "$OUT_DIR"
cd "$CODE"

if [[ "$NPROC" -gt 1 ]]; then
  CMD=(
    python3 -m torch.distributed.run
    --nproc_per_node "$NPROC"
    --standalone
    scripts/train_v3_le_primary_full_direction_smoke.py
    --ddp
  )
else
  CMD=(
    python3
    scripts/train_v3_le_primary_full_direction_smoke.py
  )
fi

CMD+=(
  --compact-list "$COMPACT_LIST"
  --out-root "$OUT_DIR"
  --preset "$PRESET"
  --model-style "$MODEL_STYLE"
  --device "$DEVICE"
  --seed "$SEED"
  --save-checkpoints
)

if [[ "$USE_PLUS_SAMPLES" != "0" ]]; then
  CMD+=(--use-plus-samples)
fi

if [[ -n "$FILTER_CASE" ]]; then
  CMD+=(--filter-case "$FILTER_CASE")
fi

if [[ -n "$FILTER_FRAME" ]]; then
  CMD+=(--filter-frame "$FILTER_FRAME")
fi

if [[ -n "$STATE_BASELINE_RANK" ]]; then
  CMD+=(--state-baseline-rank "$STATE_BASELINE_RANK")
fi

if [[ -n "$RESIDUAL_JACOBIAN_WEIGHT" ]]; then
  CMD+=(--residual-jacobian-weight "$RESIDUAL_JACOBIAN_WEIGHT")
fi

if [[ -n "$EVAL_EVERY" ]]; then
  CMD+=(--eval-every "$EVAL_EVERY")
fi

CMD+=("${EXTRA_ARGS[@]}")

printf "%q " "${CMD[@]}" > "$OUT_DIR/command.txt"
printf "\n" >> "$OUT_DIR/command.txt"
{
  echo "OUT_DIR=$OUT_DIR"
  echo "CODE=$CODE"
  echo "COMPACT_LIST=$COMPACT_LIST"
  echo "PRESET=$PRESET"
  echo "MODEL_STYLE=$MODEL_STYLE"
  echo "USE_PLUS_SAMPLES=$USE_PLUS_SAMPLES"
  echo "FILTER_CASE=$FILTER_CASE"
  echo "FILTER_FRAME=$FILTER_FRAME"
  echo "STATE_BASELINE_RANK=$STATE_BASELINE_RANK"
  echo "RESIDUAL_JACOBIAN_WEIGHT=$RESIDUAL_JACOBIAN_WEIGHT"
  echo "EVAL_EVERY=$EVAL_EVERY"
  echo "NPROC=$NPROC"
  echo "DEVICE=$DEVICE"
  echo "SEED=$SEED"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
} > "$OUT_DIR/run_env.txt"

"${CMD[@]}" 2>&1 | tee "$OUT_DIR/launch.log"
