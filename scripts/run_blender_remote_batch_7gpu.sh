#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/blender_remote}"

RUN_SPLIT="${RUN_SPLIT:-auto}"
RUN_MLP="${RUN_MLP:-1}"
RUN_SIAMESE="${RUN_SIAMESE:-1}"
RUN_XGBOOST="${RUN_XGBOOST:-0}"
USE_DDP="${USE_DDP:-1}"
USE_AMP="${USE_AMP:-1}"

BLENDER_ALL="${BLENDER_ALL:-$ROOT_DIR/datasets/Blender - synthetic/normalized_dataset.csv}"
BLENDER_TRAIN="${BLENDER_TRAIN:-$ROOT_DIR/datasets/Blender - synthetic/normalized_dataset_TRAIN.csv}"
BLENDER_VALID="${BLENDER_VALID:-$ROOT_DIR/datasets/Blender - synthetic/normalized_dataset_VALID.csv}"
BLENDER_TEST="${BLENDER_TEST:-$ROOT_DIR/datasets/Blender - synthetic/normalized_dataset_TEST.csv}"

SPLIT_MODE="${SPLIT_MODE:-ratio}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_CHUNKSIZE="${SPLIT_CHUNKSIZE:-200000}"

MLP_GPU_SET="${MLP_GPU_SET:-0,1,2,3}"
SIAMESE_GPU_SET="${SIAMESE_GPU_SET:-4,5,6}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-2048}"
SIAMESE_BATCH_SIZE="${SIAMESE_BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MLP_MASTER_PORT="${MLP_MASTER_PORT:-29501}"
SIAMESE_MASTER_PORT="${SIAMESE_MASTER_PORT:-29502}"

mkdir -p "$LOG_DIR"

count_visible_gpus() {
  local gpu_csv="$1"
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
  echo "${#gpu_ids[@]}"
}

need_split=0
for split_file in "$BLENDER_TRAIN" "$BLENDER_VALID" "$BLENDER_TEST"; do
  if [[ ! -f "$split_file" ]]; then
    need_split=1
  fi
done

if [[ "$RUN_SPLIT" == "1" || ( "$RUN_SPLIT" == "auto" && "$need_split" == "1" ) ]]; then
  echo "Creating subject-wise Blender splits..."
  "$PYTHON_BIN" "$ROOT_DIR/Split_Dataset_subjectwise.py" \
    --input-csv "$BLENDER_ALL" \
    --train-out "$BLENDER_TRAIN" \
    --valid-out "$BLENDER_VALID" \
    --test-out "$BLENDER_TEST" \
    --split-mode "$SPLIT_MODE" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SPLIT_SEED" \
    --chunksize "$SPLIT_CHUNKSIZE"
fi

if [[ ! -f "$BLENDER_TRAIN" || ! -f "$BLENDER_VALID" ]]; then
  echo "Missing Blender TRAIN/VALID splits. Aborting."
  exit 1
fi

mlp_pid=""
siamese_pid=""

if [[ "$RUN_MLP" == "1" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="$MLP_GPU_SET"
    mlp_gpu_count="$(count_visible_gpus "$MLP_GPU_SET")"
    if [[ "$USE_DDP" == "1" && "$mlp_gpu_count" -gt 1 ]]; then
      mlp_train_cmd=(
        "$PYTHON_BIN" -m torch.distributed.run
        --standalone
        --nproc_per_node "$mlp_gpu_count"
        --master-port "$MLP_MASTER_PORT"
        "$ROOT_DIR/Train_MLP.py"
        --dataset blender
        --device cuda
        --distributed
        --batch-size "$MLP_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --model-save-path "$ROOT_DIR/models/blender_MLP.pth"
      )
    else
      mlp_train_cmd=(
        "$PYTHON_BIN" "$ROOT_DIR/Train_MLP.py"
        --dataset blender
        --device cuda
        --batch-size "$MLP_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --model-save-path "$ROOT_DIR/models/blender_MLP.pth"
      )
      if [[ "$mlp_gpu_count" -gt 1 ]]; then
        mlp_train_cmd+=(--multi-gpu)
      fi
    fi
    if [[ "$USE_AMP" == "1" ]]; then
      mlp_train_cmd+=(--amp)
    fi

    "${mlp_train_cmd[@]}" > "$LOG_DIR/blender_mlp_train.log" 2>&1

    "$PYTHON_BIN" "$ROOT_DIR/Run_CrossDomain_MLP.py" \
      --model-path "$ROOT_DIR/models/blender_MLP.pth" \
      --source-dataset blender \
      --device cuda \
      --output-dir "$ROOT_DIR/models/stats/cross_domain/blender_mlp" \
      --tag blender_mlp \
      > "$LOG_DIR/blender_mlp_eval.log" 2>&1
  ) &
  mlp_pid=$!
  echo "Started MLP pipeline on GPUs [$MLP_GPU_SET] (pid=$mlp_pid)"
fi

if [[ "$RUN_SIAMESE" == "1" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="$SIAMESE_GPU_SET"
    siamese_gpu_count="$(count_visible_gpus "$SIAMESE_GPU_SET")"
    if [[ "$USE_DDP" == "1" && "$siamese_gpu_count" -gt 1 ]]; then
      siamese_train_cmd=(
        "$PYTHON_BIN" -m torch.distributed.run
        --standalone
        --nproc_per_node "$siamese_gpu_count"
        --master-port "$SIAMESE_MASTER_PORT"
        "$ROOT_DIR/Train_siameseMLP.py"
        --dataset blender
        --device cuda
        --distributed
        --batch-size "$SIAMESE_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --model-save-path "$ROOT_DIR/models/blender_siameseMLP.pth"
      )
    else
      siamese_train_cmd=(
        "$PYTHON_BIN" "$ROOT_DIR/Train_siameseMLP.py"
        --dataset blender
        --device cuda
        --batch-size "$SIAMESE_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --model-save-path "$ROOT_DIR/models/blender_siameseMLP.pth"
      )
      if [[ "$siamese_gpu_count" -gt 1 ]]; then
        siamese_train_cmd+=(--multi-gpu)
      fi
    fi
    if [[ "$USE_AMP" == "1" ]]; then
      siamese_train_cmd+=(--amp)
    fi

    "${siamese_train_cmd[@]}" > "$LOG_DIR/blender_siamese_train.log" 2>&1

    "$PYTHON_BIN" "$ROOT_DIR/Run_CrossDomain_Siamese.py" \
      --model-path "$ROOT_DIR/models/blender_siameseMLP.pth" \
      --source-dataset blender \
      --device cuda \
      --output-dir "$ROOT_DIR/models/stats/cross_domain/blender_siamese" \
      --tag blender_siamese \
      > "$LOG_DIR/blender_siamese_eval.log" 2>&1
  ) &
  siamese_pid=$!
  echo "Started Siamese pipeline on GPUs [$SIAMESE_GPU_SET] (pid=$siamese_pid)"
fi

if [[ -n "$mlp_pid" ]]; then
  wait "$mlp_pid"
fi
if [[ -n "$siamese_pid" ]]; then
  wait "$siamese_pid"
fi

if [[ "$RUN_XGBOOST" == "1" ]]; then
  echo "Running optional XGBoost pipeline..."
  "$PYTHON_BIN" "$ROOT_DIR/Train_XGBoost.py" \
    --dataset blender \
    --model-dir "$ROOT_DIR/models" \
    --stats-dir "$ROOT_DIR/models/stats" \
    > "$LOG_DIR/blender_xgboost_train.log" 2>&1

  "$PYTHON_BIN" "$ROOT_DIR/Run_CrossDomain_XGBoost.py" \
    --model-path "$ROOT_DIR/models/blender_xgboost.pkl" \
    --source-dataset blender \
    --device cpu \
    --output-dir "$ROOT_DIR/models/stats/cross_domain/blender_xgboost" \
    --tag blender_xgboost \
    > "$LOG_DIR/blender_xgboost_eval.log" 2>&1
fi

echo "All requested Blender pipelines completed."
echo "Logs: $LOG_DIR"
