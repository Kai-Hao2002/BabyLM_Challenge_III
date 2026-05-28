#!/bin/bash
#SBATCH --job-name=eval_100M_all_ckpts
#SBATCH --output=logs/eval_100M_all_ckpts_%j.out
#SBATCH --error=logs/eval_100M_all_ckpts_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set +e

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III

mkdir -p logs

# ==============================
# Model roots
# ==============================

ROOTS=(
  "outputs/baseline_chunk_lr_1e-4_100M"
  "outputs/baseline_pack_wrapped_lr_1e-4_100M"
  "outputs/baseline_gpt2_100M_v2"
)

TASKS=(
  babybabellm_eng
  babybabellm_nld
  babybabellm_zho
)

# ==============================
# HuggingFace cache
# ==============================

export HF_HOME=~/hf_cache_babylm_eval
export HF_DATASETS_CACHE=~/hf_cache_babylm_eval/datasets
export TRANSFORMERS_CACHE=~/hf_cache_babylm_eval/transformers

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "========================================"
echo "BabyLM 100M all checkpoints evaluation"
echo "Roots: ${ROOTS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ==============================
# Function: fix tokenizer config
# ==============================

fix_tokenizer_config () {
  CKPT_PATH="$1"

  if [ ! -f "$CKPT_PATH/tokenizer.json" ]; then
    echo "[TOKENIZER FIX SKIP] No tokenizer.json in $CKPT_PATH"
    return
  fi

  if [ -f "$CKPT_PATH/tokenizer_config.json" ]; then
    if [ ! -f "$CKPT_PATH/tokenizer_config.json.bak" ]; then
      cp "$CKPT_PATH/tokenizer_config.json" "$CKPT_PATH/tokenizer_config.json.bak"
    fi
  fi

  python - "$CKPT_PATH" <<'PY'
import json
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
cfg_path = ckpt_dir / "tokenizer_config.json"

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

# Keep project special tokens.
new_cfg["unk_token"] = old_cfg.get("unk_token", "<unk>")
new_cfg["bos_token"] = old_cfg.get("bos_token", "<s>")
new_cfg["eos_token"] = old_cfg.get("eos_token", "</s>")
new_cfg["pad_token"] = old_cfg.get("pad_token", "<pad>")
new_cfg["mask_token"] = old_cfg.get("mask_token", "<mask>")

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

(ckpt_dir / "special_tokens_map.json").write_text(
    json.dumps(special_map, indent=2, ensure_ascii=False)
)

print(f"[TOKENIZER FIXED] {ckpt_dir}")
PY
}

# ==============================
# Function: precheck checkpoint
# ==============================

precheck_checkpoint () {
  CKPT_PATH="$1"
  OUT_DIR="$2"

  python - "$CKPT_PATH" > "$OUT_DIR/precheck.out" 2> "$OUT_DIR/precheck.err" <<'PY'
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

ckpt_dir = Path(sys.argv[1])

tok = AutoTokenizer.from_pretrained(ckpt_dir)
cfg = AutoConfig.from_pretrained(ckpt_dir)

print("checkpoint:", ckpt_dir)
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

  return $?
}

# ==============================
# Main loop
# ==============================

for ROOT_DIR in "${ROOTS[@]}"; do
  if [ ! -d "$ROOT_DIR" ]; then
    echo "[SKIP ROOT] Not found: $ROOT_DIR"
    continue
  fi

  ROOT_NAME=$(basename "$ROOT_DIR")
  EVAL_ROOT="outputs/evaluations/${ROOT_NAME}/checkpoints"

  mkdir -p "$EVAL_ROOT"

  echo ""
  echo "########################################"
  echo "Processing root: $ROOT_NAME"
  echo "Root dir: $ROOT_DIR"
  echo "Eval root: $EVAL_ROOT"
  echo "########################################"

  for MODEL_DIR in "$ROOT_DIR"/*; do
    if [ ! -d "$MODEL_DIR" ]; then
      continue
    fi

    MODEL_NAME=$(basename "$MODEL_DIR")
    CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"

    if [ ! -d "$CHECKPOINTS_DIR" ]; then
      echo "[SKIP] No checkpoints dir: $CHECKPOINTS_DIR"
      continue
    fi

    echo ""
    echo "========================================"
    echo "Model: $MODEL_NAME"
    echo "Checkpoints dir: $CHECKPOINTS_DIR"
    echo "========================================"

    for CKPT_PATH in "$CHECKPOINTS_DIR"/*; do
      if [ ! -d "$CKPT_PATH" ]; then
        continue
      fi

      CKPT_NAME=$(basename "$CKPT_PATH")
      CKPT_EVAL_DIR="$EVAL_ROOT/$MODEL_NAME/$CKPT_NAME"

      mkdir -p "$CKPT_EVAL_DIR"

      echo ""
      echo "----------------------------------------"
      echo "Root: $ROOT_NAME"
      echo "Model: $MODEL_NAME"
      echo "Checkpoint: $CKPT_NAME"
      echo "Path: $CKPT_PATH"
      echo "Eval dir: $CKPT_EVAL_DIR"
      echo "----------------------------------------"

      # ------------------------------
      # Basic checks
      # ------------------------------

      if [ ! -f "$CKPT_PATH/config.json" ]; then
        echo "[SKIP] Missing config.json: $CKPT_PATH"
        echo "SKIPPED: missing config.json" > "$CKPT_EVAL_DIR/checkpoint_status.txt"
        continue
      fi

      if [ ! -f "$CKPT_PATH/model.safetensors" ] && [ ! -f "$CKPT_PATH/pytorch_model.bin" ]; then
        echo "[SKIP] Missing model weights: $CKPT_PATH"
        echo "SKIPPED: missing model weights" > "$CKPT_EVAL_DIR/checkpoint_status.txt"
        continue
      fi

      if [ ! -f "$CKPT_PATH/tokenizer.json" ]; then
        echo "[SKIP] Missing tokenizer.json: $CKPT_PATH"
        echo "SKIPPED: missing tokenizer.json" > "$CKPT_EVAL_DIR/checkpoint_status.txt"
        continue
      fi

      # ------------------------------
      # Fix tokenizer wrapper config
      # ------------------------------

      fix_tokenizer_config "$CKPT_PATH"

      # ------------------------------
      # Precheck
      # ------------------------------

      precheck_checkpoint "$CKPT_PATH" "$CKPT_EVAL_DIR"
      PRECHECK_EXIT=$?

      if [ $PRECHECK_EXIT -ne 0 ]; then
        echo "[SKIP] Precheck failed: $ROOT_NAME / $MODEL_NAME / $CKPT_NAME"
        {
          echo "SKIPPED: precheck failed"
          echo "root_name: $ROOT_NAME"
          echo "model_name: $MODEL_NAME"
          echo "checkpoint_name: $CKPT_NAME"
          echo "checkpoint_path: $CKPT_PATH"
          echo "see: $CKPT_EVAL_DIR/precheck.err"
        } > "$CKPT_EVAL_DIR/checkpoint_status.txt"
        continue
      fi

      # ------------------------------
      # Write checkpoint info
      # ------------------------------

      {
        echo "root_name: $ROOT_NAME"
        echo "root_dir: $ROOT_DIR"
        echo "model_name: $MODEL_NAME"
        echo "model_dir: $MODEL_DIR"
        echo "checkpoint_name: $CKPT_NAME"
        echo "checkpoint_path: $CKPT_PATH"
        echo "eval_dir: $CKPT_EVAL_DIR"
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
        cat "$CKPT_EVAL_DIR/precheck.out"
        echo ""
        echo "config.json:"
        cat "$CKPT_PATH/config.json"
        echo ""
        echo "tokenizer_config.json:"
        cat "$CKPT_PATH/tokenizer_config.json" 2>/dev/null || echo "missing"
      } > "$CKPT_EVAL_DIR/checkpoint_info.txt"

      echo "READY" > "$CKPT_EVAL_DIR/checkpoint_status.txt"

      # ------------------------------
      # Run each task independently
      # ------------------------------

      for TASK in "${TASKS[@]}"; do
        TASK_DIR="$CKPT_EVAL_DIR/$TASK"
        RESULT_DIR="$TASK_DIR/results"
        LOG_DIR="$TASK_DIR/logs"
        STATUS_FILE="$TASK_DIR/status.txt"

        mkdir -p "$RESULT_DIR" "$LOG_DIR"

        echo ""
        echo "Running:"
        echo "  Root: $ROOT_NAME"
        echo "  Model: $MODEL_NAME"
        echo "  Checkpoint: $CKPT_NAME"
        echo "  Task: $TASK"

        if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
          echo "[SKIP] Already successful: $ROOT_NAME / $MODEL_NAME / $CKPT_NAME / $TASK"
          continue
        fi

        rm -f "$LOG_DIR/eval.out" "$LOG_DIR/eval.err"
        rm -f "$STATUS_FILE"

        START_TIME=$(date +%s)

        {
          echo "RUNNING"
          echo "started_at: $(date)"
          echo "root_name: $ROOT_NAME"
          echo "model_name: $MODEL_NAME"
          echo "checkpoint_name: $CKPT_NAME"
          echo "checkpoint_path: $CKPT_PATH"
          echo "task: $TASK"
          echo "result_dir: $RESULT_DIR"
        } > "$STATUS_FILE"

        lm_eval \
          --model hf \
          --model_args pretrained="$CKPT_PATH",tokenizer="$CKPT_PATH" \
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
            echo "root_name: $ROOT_NAME"
            echo "model_name: $MODEL_NAME"
            echo "checkpoint_name: $CKPT_NAME"
            echo "checkpoint_path: $CKPT_PATH"
            echo "task: $TASK"
            echo "result_dir: $RESULT_DIR"
          } > "$STATUS_FILE"

          echo "[SUCCESS] $ROOT_NAME / $MODEL_NAME / $CKPT_NAME / $TASK"

        else
          {
            echo "FAILED"
            echo "finished_at: $(date)"
            echo "duration_seconds: $DURATION"
            echo "exit_code: $EXIT_CODE"
            echo "root_name: $ROOT_NAME"
            echo "model_name: $MODEL_NAME"
            echo "checkpoint_name: $CKPT_NAME"
            echo "checkpoint_path: $CKPT_PATH"
            echo "task: $TASK"
            echo "result_dir: $RESULT_DIR"
            echo ""
            echo "Last 120 lines of stderr:"
            tail -120 "$LOG_DIR/eval.err"
          } > "$STATUS_FILE"

          echo "[FAILED] $ROOT_NAME / $MODEL_NAME / $CKPT_NAME / $TASK"
          echo "See: $LOG_DIR/eval.err"
        fi

      done

    done
  done
done

echo ""
echo "========================================"
echo "All 100M checkpoint evaluations finished at: $(date)"
echo "========================================"
