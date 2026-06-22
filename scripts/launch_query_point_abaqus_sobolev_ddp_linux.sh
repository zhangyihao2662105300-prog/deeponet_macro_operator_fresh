#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/ydh/桌面/zhangyihao}"
CODE="${CODE:-$BASE/deeponet_macro_operator_fresh}"
OUT_DIR="${OUT_DIR:-$BASE/run_logs_query_point_abaqus/query_point_abaqus_sobolev_v1_1}"
COMPACT_LIST="${COMPACT_LIST:-}"
POINT_FEATURE_SOURCE="${POINT_FEATURE_SOURCE:-data}"
MODEL_STYLE="${MODEL_STYLE:-query-fe-linear-residual}"
LE_NORMALIZATION="${LE_NORMALIZATION:-global-component}"
TRAIN_POINT_SAMPLE_COUNT="${TRAIN_POINT_SAMPLE_COUNT:-64}"
BRANCH_FEATURE_MODE="${BRANCH_FEATURE_MODE:-xkeep-qraw}"
INCLUDE_ID_FEATURES="${INCLUDE_ID_FEATURES:-0}"
SPLIT_MODE="${SPLIT_MODE:-case}"
VAL_CASES="${VAL_CASES:-}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SCALE_MODE="${SCALE_MODE:-normalized}"
B_LABEL_COORDINATE="${B_LABEL_COORDINATE:-auto}"
DETJ_SCALE_DIM="${DETJ_SCALE_DIM:-3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
JAC_COLS_PER_GPU="${JAC_COLS_PER_GPU:-16}"
JACOBIAN_METHOD="${JACOBIAN_METHOD:-forward}"
EVAL_COLUMNS="${EVAL_COLUMNS:-0,1,2,3,4,5,6,7,8,9,10,11}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-512}"
BASIS_DIM="${BASIS_DIM:-96}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
BRANCH_DEPTH="${BRANCH_DEPTH:-5}"
TRUNK_DEPTH="${TRUNK_DEPTH:-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-1}"
LR="${LR:-8e-5}"
LR_DECAY="${LR_DECAY:-0.9995}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
SEED="${SEED:-20260620}"
NPROC="${NPROC:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
EXTRA_ARGS=("$@")

if [[ -z "$COMPACT_LIST" ]]; then
  echo "COMPACT_LIST must point to complete Abaqus compact paths for the query-point route." >&2
  exit 2
fi

if [[ "$SPLIT_MODE" == "case" && -z "$VAL_CASES" && "${VAL_FRACTION}" == "0" ]]; then
  echo "Formal query-point route training needs VAL_CASES or VAL_FRACTION>0 for case/geometry split." >&2
  exit 2
fi

export OMP_NUM_THREADS
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export PYTHONPATH="$CODE/src:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"
cd "$CODE"

TARGET_IPS="$(seq -s, 0 127)"

CMD=(
  python3 -m torch.distributed.run
  --nproc_per_node "$NPROC"
  --standalone
  -m macro_deeponet.train_true176_generic_sobolev
  --compact-list "$COMPACT_LIST"
  --out-dir "$OUT_DIR"
  --target-ips "$TARGET_IPS"
  --branch-feature-mode "$BRANCH_FEATURE_MODE"
  --point-feature-source "$POINT_FEATURE_SOURCE"
  --model-style "$MODEL_STYLE"
  --le-normalization "$LE_NORMALIZATION"
  --train-point-sample-count "$TRAIN_POINT_SAMPLE_COUNT"
  --split-mode "$SPLIT_MODE"
  --val-fraction "$VAL_FRACTION"
  --scale-mode "$SCALE_MODE"
  --b-label-coordinate "$B_LABEL_COORDINATE"
  --detj-scale-dim "$DETJ_SCALE_DIM"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --max-eval-frames "$MAX_EVAL_FRAMES"
  --basis-dim "$BASIS_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --branch-depth "$BRANCH_DEPTH"
  --trunk-depth "$TRUNK_DEPTH"
  --activation tanh
  --baseline-jacobian-weight 1.0
  --baseline-j-loss-mode norm-plus-physical
  --jacobian-columns all
  --jacobian-columns-per-batch "$JAC_COLS_PER_GPU"
  --jacobian-method "$JACOBIAN_METHOD"
  --eval-columns "$EVAL_COLUMNS"
  --j-loss-mode norm-plus-physical
  --physical-j-aux-weight 0.5
  --physical-j-abs-weight 1.0
  --physical-j-rel-weight 0.02
  --physical-j-action-weight 0.05
  --physical-j-rel-eps-scale 0.02
  --physical-j-action-directions 4
  --initial-jacobian-weight 1.0
  --lr "$LR"
  --lr-decay "$LR_DECAY"
  --grad-clip "$GRAD_CLIP"
  --eval-every "$EVAL_EVERY"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
  --cuda
  --ddp
)

if [[ -n "$VAL_CASES" ]]; then
  CMD+=(--val-cases "$VAL_CASES")
fi

if [[ "$INCLUDE_ID_FEATURES" != "0" ]]; then
  CMD+=(--include-id-features)
fi

CMD+=("${EXTRA_ARGS[@]}")

printf "%q " "${CMD[@]}" > "$OUT_DIR/command.txt"
printf "\n" >> "$OUT_DIR/command.txt"
{
  echo "OUT_DIR=$OUT_DIR"
  echo "COMPACT_LIST=$COMPACT_LIST"
  echo "POINT_FEATURE_SOURCE=$POINT_FEATURE_SOURCE"
  echo "MODEL_STYLE=$MODEL_STYLE"
  echo "LE_NORMALIZATION=$LE_NORMALIZATION"
  echo "TRAIN_POINT_SAMPLE_COUNT=$TRAIN_POINT_SAMPLE_COUNT"
  echo "BRANCH_FEATURE_MODE=$BRANCH_FEATURE_MODE"
  echo "INCLUDE_ID_FEATURES=$INCLUDE_ID_FEATURES"
  echo "SPLIT_MODE=$SPLIT_MODE"
  echo "VAL_CASES=$VAL_CASES"
  echo "VAL_FRACTION=$VAL_FRACTION"
  echo "NPROC=$NPROC"
} > "$OUT_DIR/run_env.txt"

"${CMD[@]}"
