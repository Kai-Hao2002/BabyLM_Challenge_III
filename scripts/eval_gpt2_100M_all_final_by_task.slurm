#!/bin/bash
#SBATCH --job-name=eval_gpt2_100M_final
#SBATCH --output=logs/eval_gpt2_100M_final_%j.out
#SBATCH --error=logs/eval_gpt2_100M_final_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set +e

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III

mkdir -p logs

# ==============================
# Basic paths for 100M models
# ==============================

ROOT_DIR="outputs/baseline_pack_wrapped_lr_1e-4_gpt2_100M"
EVAL_ROOT="outputs/evaluations/baseline_pack_wrapped_lr_1e-4_gpt2_100M/final"

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
# Global run information
# ==============================

echo "========================================"
echo "BabyLM 100M GPT-2 final model evaluation"
echo "Root dir: $ROOT_DIR"
echo "Eval root: $EVAL_ROOT"
echo "Tasks: ${TASKS[*]}"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "========================================"

# ==============================
# Check root dir
# ==============================

if [ ! -d "$ROOT_DIR" ]; then
  echo "[FATAL] ROOT_DIR does not exist: $ROOT_DIR"
  exit 1
fi

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
  # Fix tokenizer config for HF AutoTokenizer
  # ------------------------------

  echo "[INFO] Fixing tokenizer config if needed: $MODEL_NAME"

  python - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
cfg_path = model_dir / "tokenizer_config.json"
special_path = model_dir / "special_tokens_map.json"

if cfg_path.exists():
    old_cfg = json.loads(cfg_path.read_text())
    backup = model_dir / "tokenizer_config.json.bak"
    if not backup.exists():
        backup.write_text(json.dumps(old_cfg, indent=2, ensure_ascii=False))
else:
    old_cfg = {
        "backend": "tokenizers",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }

new_cfg = dict(old_cfg)
new_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"

# Avoid the huge default placeholder max length.
# Actual model max length is determined by model config n_positions/n_ctx.
new_cfg["model_max_length"] = 1024
new_cfg["clean_up_tokenization_spaces"] = False

cfg_path.write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False))

special_map = {
    "unk_token": new_cfg.get("unk_token", "<unk>"),
    "bos_token": new_cfg.get("bos_token", "<s>"),
    "eos_token": new_cfg.get("eos_token", "</s>"),
    "pad_token": new_cfg.get("pad_token", "<pad>"),
    "mask_token": new_cfg.get("mask_token", "<mask>")
}

special_path.write_text(json.dumps(special_map, indent=2, ensure_ascii=False))

print("[TOKENIZER_CONFIG_FIXED]", model_dir)
PY

  TOKENIZER_FIX_EXIT=$?
  if [ $TOKENIZER_FIX_EXIT -ne 0 ]; then
    echo "[SKIP] Tokenizer config fix failed for $MODEL_NAME"
    echo "SKIPPED: tokenizer config fix failed" > "$MODEL_EVAL_DIR/model_status.txt"
    continue
  fi

  # ------------------------------
  # Validate tokenizer and vocab size
  # ------------------------------

  echo "[INFO] Validating tokenizer and vocab size: $MODEL_NAME"

  python - "$MODEL_PATH" "$MODEL_EVAL_DIR" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

model_dir = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])

config = json.loads((model_dir / "config.json").read_text())
config_vocab = config.get("vocab_size")

tok = AutoTokenizer.from_pretrained(model_dir)
tok_vocab = len(tok)

report = []
report.append(f"model_dir: {model_dir}")
report.append(f"config_vocab_size: {config_vocab}")
report.append(f"tokenizer_vocab_size: {tok_vocab}")
report.append(f"unk_token: {tok.unk_token} {tok.unk_token_id}")
report.append(f"bos_token: {tok.bos_token} {tok.bos_token_id}")
report.append(f"eos_token: {tok.eos_token} {tok.eos_token_id}")
report.append(f"pad_token: {tok.pad_token} {tok.pad_token_id}")
report.append(f"mask_token: {tok.mask_token} {tok.mask_token_id}")

expected = {
    "unk": 0,
    "bos": 1,
    "eos": 2,
    "pad": 3,
    "mask": 4,
}

ok_special = (
    tok.unk_token_id == expected["unk"] and
    tok.bos_token_id == expected["bos"] and
    tok.eos_token_id == expected["eos"] and
    tok.pad_token_id == expected["pad"] and
    tok.mask_token_id == expected["mask"]
)

ok_vocab = (config_vocab == tok_vocab)

report.append(f"special_token_ids_ok: {ok_special}")
report.append(f"vocab_size_ok: {ok_vocab}")

(eval_dir / "tokenizer_validation.txt").write_text("\n".join(report) + "\n")

print("\n".join(report))

if not ok_special:
    raise SystemExit("Special token ids are not as expected.")

if not ok_vocab:
    raise SystemExit("config.vocab_size does not match tokenizer vocab size.")
PY

  VALIDATE_EXIT=$?
  if [ $VALIDATE_EXIT -ne 0 ]; then
    echo "[SKIP] Tokenizer validation failed for $MODEL_NAME"
    echo "SKIPPED: tokenizer validation failed. See tokenizer_validation.txt" > "$MODEL_EVAL_DIR/model_status.txt"
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
    echo ""
    echo "special_tokens_map.json:"
    cat "$MODEL_PATH/special_tokens_map.json" 2>/dev/null || echo "missing"
    echo ""
    echo "tokenizer_validation.txt:"
    cat "$MODEL_EVAL_DIR/tokenizer_validation.txt" 2>/dev/null || echo "missing"
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

    # If already successful, skip
    if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
      echo "[SKIP] Already successful: $MODEL_NAME / $TASK"
      continue
    fi

    # Clean previous failed/incomplete logs for this task only
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
echo "All 100M final model evaluations finished at: $(date)"
echo "Results saved under: $EVAL_ROOT"
echo "========================================"
