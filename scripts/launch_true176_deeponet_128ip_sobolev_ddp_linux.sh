#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/ydh/桌面/zhangyihao}"
CODE="${CODE:-$BASE/deeponet_macro_operator_fresh}"
DATA="${DATA:-$BASE/true176_shape4_qraw_128ip_training_data_20260620}"
OUT_DIR="${OUT_DIR:-$BASE/run_logs_128ip_fullframe_20260620/deeponet_true176_128ip_sobolev_ddp_v1}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
JAC_COLS_PER_GPU="${JAC_COLS_PER_GPU:-8}"
JACOBIAN_METHOD="${JACOBIAN_METHOD:-forward}"
EVAL_COLUMNS="${EVAL_COLUMNS:-0,1,2,3,4,5,6,7,8,9,10,11}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-512}"
MAX_FRAMES_PER_COMPACT="${MAX_FRAMES_PER_COMPACT:-0}"
BASIS_DIM="${BASIS_DIM:-96}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
BRANCH_DEPTH="${BRANCH_DEPTH:-5}"
TRUNK_DEPTH="${TRUNK_DEPTH:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-1}"
LR="${LR:-8e-5}"
LR_DECAY="${LR_DECAY:-0.9995}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
PHYSICAL_J_ACTION_DIRECTIONS="${PHYSICAL_J_ACTION_DIRECTIONS:-4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
SEED="${SEED:-20260620}"
NPROC="${NPROC:-3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
EXTRA_ARGS=("$@")

export OMP_NUM_THREADS
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export PYTHONPATH="$CODE/src:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"
cd "$CODE"

COMPACT_LIST="$DATA/compact_paths_linux.txt"
TARGET_IPS="$(seq -s, 0 127)"

CMD=(
  python3 -m torch.distributed.run
  --nproc_per_node "$NPROC"
  --standalone
  -m macro_deeponet.train_true176_deeponet_sobolev
  --compact-list "$COMPACT_LIST"
  --out-dir "$OUT_DIR"
  --target-ips "$TARGET_IPS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --max-eval-frames "$MAX_EVAL_FRAMES"
  --basis-dim "$BASIS_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --branch-depth "$BRANCH_DEPTH"
  --trunk-depth "$TRUNK_DEPTH"
  --activation tanh
  --include-id-features
  --jacobian-columns all
  --jacobian-columns-per-batch "$JAC_COLS_PER_GPU"
  --jacobian-method "$JACOBIAN_METHOD"
  --eval-columns "$EVAL_COLUMNS"
  --j-loss-mode norm-plus-physical
  --physical-j-aux-weight 0.05
  --physical-j-abs-weight 1.0
  --physical-j-rel-weight 0.02
  --physical-j-action-weight 0.05
  --physical-j-rel-eps-scale 0.02
  --physical-j-action-directions "$PHYSICAL_J_ACTION_DIRECTIONS"
  --tangent-directions 0
  --initial-jacobian-weight 1.0
  --initial-tangent-weight 0.0
  --lr "$LR"
  --lr-decay "$LR_DECAY"
  --grad-clip "$GRAD_CLIP"
  --val-fraction 0.0
  --eval-every "$EVAL_EVERY"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
  --cuda
  --ddp
)

if [[ "$MAX_FRAMES_PER_COMPACT" != "0" ]]; then
  CMD+=(--max-frames-per-compact "$MAX_FRAMES_PER_COMPACT")
fi

if [[ -n "$INIT_CHECKPOINT" ]]; then
  CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

CMD+=("${EXTRA_ARGS[@]}")

printf "%q " "${CMD[@]}" > "$OUT_DIR/command.txt"
printf "\n" >> "$OUT_DIR/command.txt"
{
  echo "OUT_DIR=$OUT_DIR"
  echo "EPOCHS=$EPOCHS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
  echo "JAC_COLS_PER_GPU=$JAC_COLS_PER_GPU"
  echo "JACOBIAN_METHOD=$JACOBIAN_METHOD"
  echo "MAX_EVAL_FRAMES=$MAX_EVAL_FRAMES"
  echo "MAX_FRAMES_PER_COMPACT=$MAX_FRAMES_PER_COMPACT"
  echo "BASIS_DIM=$BASIS_DIM"
  echo "HIDDEN_DIM=$HIDDEN_DIM"
  echo "BRANCH_DEPTH=$BRANCH_DEPTH"
  echo "TRUNK_DEPTH=$TRUNK_DEPTH"
  echo "LR=$LR"
  echo "LR_DECAY=$LR_DECAY"
  echo "GRAD_CLIP=$GRAD_CLIP"
  echo "PHYSICAL_J_ACTION_DIRECTIONS=$PHYSICAL_J_ACTION_DIRECTIONS"
  echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
  echo "NPROC=$NPROC"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
} > "$OUT_DIR/run_env.txt"

"${CMD[@]}"
