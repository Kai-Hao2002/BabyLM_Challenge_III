#!/bin/bash
#SBATCH --job-name=eval_gpt2_10M_final
#SBATCH --output=logs/eval_gpt2_10M_final_%j.out
#SBATCH --error=logs/eval_gpt2_10M_final_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set +e

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III

mkdir -p logs

# ==============================
# Basic paths
# ==============================

ROOT_DIR="outputs/baseline_pack_wrapped_lr_1e-4_gpt2_10M"
EVAL_ROOT="outputs/evaluations/baseline_pack_wrapped_lr_1e-4_gpt2_10M/final"

TASKS=(
  babybabellm_eng
  babybabellm_nld
  babybabellm_zho
)

mkdir -p "$EVAL_ROOT"

# ==============================
# HuggingFace cache
# ==============================

export HF_HOME=~/hf_cache_babylm_eval
export HF_DATASETS_CACHE=~/hf_cache_babylm_eval/datasets
export TRANSFORMERS_CACHE=~/hf_cache_babylm_eval/transformers

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ==============================
# Run information
# ==============================

echo "========================================"
echo "BabyLM final model evaluation"
echo "Root dir: $ROOT_DIR"
echo "Eval root: $EVAL_ROOT"
echo "Tasks: ${TASKS[*]}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ==============================
# Main loop
# ==============================

for MODEL_DIR in "$ROOT_DIR"/*; do
  if [ ! -d "$MODEL_DIR" ]; then
    continue
  fi

  MODEL_NAME=$(basename "$MODEL_DIR")
  MODEL_PATH="$MODEL_DIR/checkpoints/final_model"
  MODEL_EVAL_DIR="$EVAL_ROOT/$MODEL_NAME"

  echo ""
  echo "========================================"
  echo "Processing model: $MODEL_NAME"
  echo "Model path: $MODEL_PATH"
  echo "Eval dir: $MODEL_EVAL_DIR"
  echo "========================================"

  mkdir -p "$MODEL_EVAL_DIR"

  # ------------------------------
  # Basic checks
  # ------------------------------

  if [ ! -d "$MODEL_PATH" ]; then
    echo "[SKIP] No final_model directory: $MODEL_PATH"
    echo "SKIPPED: no final_model directory" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[SKIP] Missing config.json: $MODEL_PATH/config.json"
    echo "SKIPPED: missing config.json" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[SKIP] Missing tokenizer.json: $MODEL_PATH/tokenizer.json"
    echo "SKIPPED: missing tokenizer.json" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  # ------------------------------
  # Write model info
  # ------------------------------

  {
    echo "model_name: $MODEL_NAME"
    echo "model_path: $MODEL_PATH"
    echo "eval_dir: $MODEL_EVAL_DIR"
    echo "tasks: ${TASKS[*]}"
    echo "date: $(date)"
    echo "host: $(hostname)"
    echo "conda_env: $CONDA_DEFAULT_ENV"
    echo "hf_home: $HF_HOME"
    echo "hf_datasets_cache: $HF_DATASETS_CACHE"
    echo "transformers_cache: $TRANSFORMERS_CACHE"
    echo ""
    echo "git_commit:"
    git rev-parse HEAD 2>/dev/null || echo "unknown"
    echo ""
    echo "config.json:"
    cat "$MODEL_PATH/config.json"
    echo ""
    echo "tokenizer_config.json:"
    cat "$MODEL_PATH/tokenizer_config.json" 2>/dev/null || echo "missing"
  } > "$MODEL_EVAL_DIR/model_info.txt"

  # ------------------------------
  # Run each task independently
  # ------------------------------

  for TASK in "${TASKS[@]}"; do
    TASK_DIR="$MODEL_EVAL_DIR/$TASK"
    RESULT_DIR="$TASK_DIR/results"
    LOG_DIR="$TASK_DIR/logs"
    STATUS_FILE="$TASK_DIR/status.txt"

    mkdir -p "$RESULT_DIR" "$LOG_DIR"

    echo ""
    echo "----------------------------------------"
    echo "Model: $MODEL_NAME"
    echo "Task: $TASK"
    echo "Result dir: $RESULT_DIR"
    echo "----------------------------------------"

    # If already successful, skip
    if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
      echo "[SKIP] Already successful: $MODEL_NAME / $TASK"
      continue
    fi

    # Clean previous failed/incomplete results for this task only
    rm -f "$LOG_DIR/eval.out" "$LOG_DIR/eval.err"
    rm -f "$STATUS_FILE"

    START_TIME=$(date +%s)

    echo "RUNNING" > "$STATUS_FILE"
    echo "started_at: $(date)" >> "$STATUS_FILE"
    echo "model_name: $MODEL_NAME" >> "$STATUS_FILE"
    echo "model_path: $MODEL_PATH" >> "$STATUS_FILE"
    echo "task: $TASK" >> "$STATUS_FILE"

    lm_eval \
      --model hf \
      --model_args pretrained="$MODEL_PATH",tokenizer="$MODEL_PATH" \
      --include_path external/multilingual-evaluation/tasks \
      --tasks "$TASK" \
      --batch_size auto \
      --device cuda \
      --output_path "$RESULT_DIR" \
      --log_samples \
      > "$LOG_DIR/eval.out" \
      2> "$LOG_DIR/eval.err"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
      {
        echo "SUCCESS"
        echo "finished_at: $(date)"
        echo "duration_seconds: $DURATION"
        echo "exit_code: $EXIT_CODE"
        echo "model_name: $MODEL_NAME"
        echo "model_path: $MODEL_PATH"
        echo "task: $TASK"
        echo "result_dir: $RESULT_DIR"
      } > "$STATUS_FILE"

      echo "[SUCCESS] $MODEL_NAME / $TASK"

    else
      {
        echo "FAILED"
        echo "finished_at: $(date)"
        echo "duration_seconds: $DURATION"
        echo "exit_code: $EXIT_CODE"
        echo "model_name: $MODEL_NAME"
        echo "model_path: $MODEL_PATH"
        echo "task: $TASK"
        echo "result_dir: $RESULT_DIR"
        echo ""
        echo "Last 80 lines of stderr:"
        tail -80 "$LOG_DIR/eval.err"
      } > "$STATUS_FILE"

      echo "[FAILED] $MODEL_NAME / $TASK"
      echo "See: $LOG_DIR/eval.err"
    fi

  done

done

echo ""
echo "========================================"
echo "All final model evaluations finished at: $(date)"
echo "Results saved under: $EVAL_ROOT"
echo "========================================"
