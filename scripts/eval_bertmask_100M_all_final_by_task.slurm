#!/bin/bash
#SBATCH --job-name=eval_bertmask_100M
#SBATCH --output=logs/eval_bertmask_100M_%j.out
#SBATCH --error=logs/eval_bertmask_100M_%j.err
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

ROOT_DIR="outputs/baseline_pack_wrapped_lr_1e-4_100M"
EVAL_ROOT="outputs/evaluations/baseline_pack_wrapped_lr_1e-4_100M/final"

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

echo "========================================"
echo "BabyLM bertmask 100M final evaluation"
echo "Root dir: $ROOT_DIR"
echo "Eval root: $EVAL_ROOT"
echo "Tasks: ${TASKS[*]}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ==============================
# Function: fix tokenizer config
# ==============================

fix_tokenizer_config () {
  MODEL_PATH="$1"

  if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[TOKENIZER FIX SKIP] No tokenizer.json in $MODEL_PATH"
    return
  fi

  if [ -f "$MODEL_PATH/tokenizer_config.json" ]; then
    if [ ! -f "$MODEL_PATH/tokenizer_config.json.bak" ]; then
      cp "$MODEL_PATH/tokenizer_config.json" "$MODEL_PATH/tokenizer_config.json.bak"
    fi
  fi

  python - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
cfg_path = model_dir / "tokenizer_config.json"

if cfg_path.exists():
    try:
        old_cfg = json.loads(cfg_path.read_text())
    except Exception:
        old_cfg = {}
else:
    old_cfg = {}

new_cfg = dict(old_cfg)

# Transformers does not know TokenizersBackend.
new_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"

# Keep original special tokens if present.
new_cfg["unk_token"] = old_cfg.get("unk_token", "<unk>")
new_cfg["bos_token"] = old_cfg.get("bos_token", "<s>")
new_cfg["eos_token"] = old_cfg.get("eos_token", "</s>")
new_cfg["pad_token"] = old_cfg.get("pad_token", "<pad>")
new_cfg["mask_token"] = old_cfg.get("mask_token", "<mask>")

# Avoid huge placeholder max length in tokenizer_config.
new_cfg["model_max_length"] = 1024
new_cfg["clean_up_tokenization_spaces"] = False
new_cfg["backend"] = old_cfg.get("backend", "tokenizers")

cfg_path.write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False))

special_map = {
    "unk_token": new_cfg["unk_token"],
    "bos_token": new_cfg["bos_token"],
    "eos_token": new_cfg["eos_token"],
    "pad_token": new_cfg["pad_token"],
    "mask_token": new_cfg["mask_token"],
}

(model_dir / "special_tokens_map.json").write_text(
    json.dumps(special_map, indent=2, ensure_ascii=False)
)

print(f"[TOKENIZER FIXED] {model_dir}")
PY
}

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

  if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
    echo "[SKIP] Missing model weights: $MODEL_PATH/model.safetensors or pytorch_model.bin"
    echo "SKIPPED: missing model weights" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[SKIP] Missing tokenizer.json: $MODEL_PATH/tokenizer.json"
    echo "SKIPPED: missing tokenizer.json" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  # ------------------------------
  # Fix tokenizer wrapper config
  # ------------------------------

  fix_tokenizer_config "$MODEL_PATH"

  # ------------------------------
  # Precheck
  # ------------------------------

  python - "$MODEL_PATH" > "$MODEL_EVAL_DIR/precheck.out" 2> "$MODEL_EVAL_DIR/precheck.err" <<'PY'
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModel

model_dir = Path(sys.argv[1])

tok = AutoTokenizer.from_pretrained(model_dir)
cfg = AutoConfig.from_pretrained(model_dir)

print("tokenizer:", type(tok))
print("tokenizer_vocab_size:", len(tok))
print("config_vocab_size:", getattr(cfg, "vocab_size", None))
print("model_type:", getattr(cfg, "model_type", None))
print("architectures:", getattr(cfg, "architectures", None))
print("max_position_embeddings:", getattr(cfg, "max_position_embeddings", None))
print("n_positions:", getattr(cfg, "n_positions", None))
print("n_ctx:", getattr(cfg, "n_ctx", None))
print("unk:", tok.unk_token, tok.unk_token_id)
print("bos:", tok.bos_token, tok.bos_token_id)
print("eos:", tok.eos_token, tok.eos_token_id)
print("pad:", tok.pad_token, tok.pad_token_id)
print("mask:", tok.mask_token, tok.mask_token_id)

if getattr(cfg, "vocab_size", None) != len(tok):
    print("WARNING: config vocab_size != tokenizer vocab size")

print("precheck passed")
PY

  PRECHECK_EXIT=$?

  if [ $PRECHECK_EXIT -ne 0 ]; then
    echo "[SKIP] Precheck failed for $MODEL_NAME"
    {
      echo "SKIPPED: precheck failed"
      echo "model_name: $MODEL_NAME"
      echo "model_path: $MODEL_PATH"
      echo "see: $MODEL_EVAL_DIR/precheck.err"
    } > "$MODEL_EVAL_DIR/model_status.txt"
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
    echo "precheck.out:"
    cat "$MODEL_EVAL_DIR/precheck.out"
    echo ""
    echo "config.json:"
    cat "$MODEL_PATH/config.json"
    echo ""
    echo "tokenizer_config.json:"
    cat "$MODEL_PATH/tokenizer_config.json" 2>/dev/null || echo "missing"
  } > "$MODEL_EVAL_DIR/model_info.txt"

  echo "READY" > "$MODEL_EVAL_DIR/model_status.txt"

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

    if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
      echo "[SKIP] Already successful: $MODEL_NAME / $TASK"
      continue
    fi

    rm -f "$LOG_DIR/eval.out" "$LOG_DIR/eval.err"
    rm -f "$STATUS_FILE"

    START_TIME=$(date +%s)

    {
      echo "RUNNING"
      echo "started_at: $(date)"
      echo "model_name: $MODEL_NAME"
      echo "model_path: $MODEL_PATH"
      echo "task: $TASK"
      echo "result_dir: $RESULT_DIR"
    } > "$STATUS_FILE"

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
        echo "Last 100 lines of stderr:"
        tail -100 "$LOG_DIR/eval.err"
      } > "$STATUS_FILE"

      echo "[FAILED] $MODEL_NAME / $TASK"
      echo "See: $LOG_DIR/eval.err"
    fi

  done

done

echo ""
echo "========================================"
echo "All bertmask 100M final evaluations finished at: $(date)"
echo "Results saved under: $EVAL_ROOT"
echo "========================================"
