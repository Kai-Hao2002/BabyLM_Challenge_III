#!/bin/bash
#SBATCH --job-name=eval_final_100M_pack
#SBATCH --output=logs/eval_final_100M_pack_%j.out
#SBATCH --error=logs/eval_final_100M_pack_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set +e
shopt -s nullglob

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III

mkdir -p logs

# ============================================================
# Model roots
# ============================================================

ROOTS=(
  "outputs/baseline_pack_wrapped_lr_1e-4_100M"
)

TASKS=(
  babybabellm_eng
  babybabellm_nld
  babybabellm_zho
)

# ============================================================
# Cache settings
# Use a dedicated cache dir for this evaluation batch.
# ============================================================

export HF_HOME=~/hf_cache_babylm_eval_100M
export HF_DATASETS_CACHE=~/hf_cache_babylm_eval_100M/datasets
export TRANSFORMERS_CACHE=~/hf_cache_babylm_eval_100M/transformers
export HF_HUB_CACHE=~/hf_cache_babylm_eval_100M/hub

export HF_HUB_DISABLE_TELEMETRY=1

# ============================================================
# HuggingFace authentication
# Avoid anonymous HF rate limit on shared LRZ IP.
# Put your read token in ~/.hf_token with chmod 600 ~/.hf_token.
# The token is loaded into environment variables, never printed.
# ============================================================

if [ -f ~/.hf_token ]; then
  export HF_TOKEN=$(tr -d '\n\r' < ~/.hf_token)
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "[HF AUTH] HF_TOKEN loaded from ~/.hf_token"
else
  echo "[HF AUTH WARNING] ~/.hf_token not found. HuggingFace may rate limit anonymous requests."
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

# ============================================================
# Warmup settings
# ============================================================

WARMUP_ROOT="outputs/evaluations/_warmup_cache/babylm_100M_three_roots"
WARMUP_LIMIT=5
WARMUP_DEVICE="cuda"

# Use GPT-2 512 final_model for warmup if available.
# This avoids bertmask incompatibility and 32-context issues.
WARMUP_MODEL="outputs/baseline_gpt2_100M_v2/baseline_pack_wrapped_512_lr_1e-4_gpt2_100M/checkpoints/final_model"

# If the preferred warmup model does not exist, auto-find another GPT-2 final_model.
if [ ! -d "$WARMUP_MODEL" ]; then
  echo "[WARMUP] Preferred warmup model not found:"
  echo "         $WARMUP_MODEL"
  echo "[WARMUP] Trying to auto-find a GPT-2 final_model..."

  for CANDIDATE in outputs/baseline_gpt2_100M_v2/*/checkpoints/final_model; do
    if [ -f "$CANDIDATE/config.json" ] && [ -f "$CANDIDATE/tokenizer.json" ]; then
      WARMUP_MODEL="$CANDIDATE"
      echo "[WARMUP] Found warmup model: $WARMUP_MODEL"
      break
    fi
  done
fi

mkdir -p "$WARMUP_ROOT"

# ============================================================
# General run info
# ============================================================

echo "============================================================"
echo "BabyLM 100M final_model evaluation with lm_eval warmup"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Roots:"
for R in "${ROOTS[@]}"; do
  echo "  - $R"
done
echo ""
echo "Tasks: ${TASKS[*]}"
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "Warmup model: $WARMUP_MODEL"
echo "Warmup root: $WARMUP_ROOT"
echo "============================================================"

# ============================================================
# Function: fix tokenizer config
# ============================================================

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

(ckpt_dir / "special_tokens_map.json").write_text(
    json.dumps(special_map, indent=2, ensure_ascii=False)
)

print(f"[TOKENIZER FIXED] {ckpt_dir}")
PY
}

# ============================================================
# Function: precheck checkpoint
# ============================================================

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

# ============================================================
# Phase 1: lm_eval warmup
# This downloads/caches datasets and task resources.
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 1: lm_eval warmup"
echo "============================================================"

# Warmup must be online so datasets can be downloaded into cache.
# Full evaluation will switch to offline/cache mode after warmup succeeds.
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "[WARMUP ONLINE] HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "[WARMUP ONLINE] HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "[WARMUP ONLINE] TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

if [ ! -d "$WARMUP_MODEL" ]; then
  echo "[FATAL] No valid warmup model found."
  echo "Expected warmup model path:"
  echo "$WARMUP_MODEL"
  exit 1
fi

fix_tokenizer_config "$WARMUP_MODEL"

WARMUP_ALL_SUCCESS=true

for TASK in "${TASKS[@]}"; do
  TASK_WARMUP_DIR="$WARMUP_ROOT/$TASK"
  mkdir -p "$TASK_WARMUP_DIR/logs" "$TASK_WARMUP_DIR/results"

  STATUS_FILE="$TASK_WARMUP_DIR/status.txt"

  if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
    echo "[WARMUP SKIP] Already successful: $TASK"
    continue
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo "[WARMUP] Task: $TASK"
  echo "[WARMUP] Model: $WARMUP_MODEL"
  echo "[WARMUP] Output: $TASK_WARMUP_DIR"
  echo "------------------------------------------------------------"

  {
    echo "RUNNING"
    echo "started_at: $(date)"
    echo "task: $TASK"
    echo "warmup_model: $WARMUP_MODEL"
    echo "warmup_limit: $WARMUP_LIMIT"
  } > "$STATUS_FILE"

  lm_eval \
    --model hf \
    --model_args pretrained="$WARMUP_MODEL",tokenizer="$WARMUP_MODEL" \
    --include_path external/multilingual-evaluation/tasks \
    --tasks "$TASK" \
    --batch_size 1 \
    --limit "$WARMUP_LIMIT" \
    --device "$WARMUP_DEVICE" \
    --output_path "$TASK_WARMUP_DIR/results" \
    --log_samples \
    > "$TASK_WARMUP_DIR/logs/warmup.out" \
    2> "$TASK_WARMUP_DIR/logs/warmup.err"

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    {
      echo "SUCCESS"
      echo "finished_at: $(date)"
      echo "exit_code: $EXIT_CODE"
      echo "task: $TASK"
      echo "warmup_model: $WARMUP_MODEL"
      echo "result_dir: $TASK_WARMUP_DIR/results"
    } > "$STATUS_FILE"

    echo "[WARMUP SUCCESS] $TASK"

  else
    WARMUP_ALL_SUCCESS=false

    {
      echo "FAILED"
      echo "finished_at: $(date)"
      echo "exit_code: $EXIT_CODE"
      echo "task: $TASK"
      echo "warmup_model: $WARMUP_MODEL"
      echo ""
      echo "Last 120 lines of stderr:"
      tail -120 "$TASK_WARMUP_DIR/logs/warmup.err"
    } > "$STATUS_FILE"

    echo "[WARMUP FAILED] $TASK"
    echo "See: $TASK_WARMUP_DIR/logs/warmup.err"
  fi
done

if [ "$WARMUP_ALL_SUCCESS" != true ]; then
  echo ""
  echo "[FATAL] Warmup did not fully succeed."
  echo "Do not start full evaluation, because datasets may not be fully cached."
  echo "Check:"
  echo "  $WARMUP_ROOT/*/status.txt"
  exit 1
fi

echo ""
echo "================================================------------"
echo "Warmup finished successfully."
echo "Now enabling offline/cache mode for full evaluation."
echo "================================================------------"

# ============================================================
# Phase 2: full evaluation using cache
# ============================================================

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

for ROOT_DIR in "${ROOTS[@]}"; do
  if [ ! -d "$ROOT_DIR" ]; then
    echo "[SKIP ROOT] Not found: $ROOT_DIR"
    continue
  fi

  ROOT_NAME=$(basename "$ROOT_DIR")
  EVAL_ROOT="outputs/evaluations/${ROOT_NAME}/final_model_only"

  mkdir -p "$EVAL_ROOT"

  echo ""
  echo "############################################################"
  echo "Processing root: $ROOT_NAME"
  echo "Root dir: $ROOT_DIR"
  echo "Eval root: $EVAL_ROOT"
  echo "############################################################"

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
    echo "============================================================"
    echo "Model: $MODEL_NAME"
    echo "Checkpoints dir: $CHECKPOINTS_DIR"
    echo "============================================================"

    # Only evaluate final_model.
    for CKPT_PATH in "$CHECKPOINTS_DIR"/final_model; do
      if [ ! -d "$CKPT_PATH" ]; then
        continue
      fi

      CKPT_NAME=$(basename "$CKPT_PATH")
      CKPT_EVAL_DIR="$EVAL_ROOT/$MODEL_NAME/$CKPT_NAME"

      mkdir -p "$CKPT_EVAL_DIR"

      echo ""
      echo "------------------------------------------------------------"
      echo "Root: $ROOT_NAME"
      echo "Model: $MODEL_NAME"
      echo "Checkpoint: $CKPT_NAME"
      echo "Path: $CKPT_PATH"
      echo "Eval dir: $CKPT_EVAL_DIR"
      echo "------------------------------------------------------------"

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
        echo "hf_hub_offline: $HF_HUB_OFFLINE"
        echo "hf_datasets_offline: $HF_DATASETS_OFFLINE"
        echo "transformers_offline: $TRANSFORMERS_OFFLINE"
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
        echo "  Result: $RESULT_DIR"

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
echo "============================================================"
echo "All 100M final_model evaluations finished at: $(date)"
echo "============================================================"
