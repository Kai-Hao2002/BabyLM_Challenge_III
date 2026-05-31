#!/bin/bash
#SBATCH --job-name=ft_full_100M_all
#SBATCH --output=logs/ft_full_100M_all_%j.out
#SBATCH --error=logs/ft_full_100M_all_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# BabyLM 100M finetune FULL run for all final_model checkpoints.
# - Traverses all three model roots.
# - Runs only existing language/task data files.
# - Supports resume/skip via status.txt.
# - Writes results under outputs/evaluations_finetune_100M/full.

set +e
shopt -s nullglob

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III || exit 1
mkdir -p logs

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT="$PWD"
FINETUNE_DIR="external/multilingual-evaluation/finetuning_babybabel_lm"
FINETUNE_PY="$FINETUNE_DIR/finetune_classification.py"
DATA_ROOT="$FINETUNE_DIR/finetuning_data"
OUT_ROOT="outputs/evaluations_finetune_100M/full"
LOCAL_MODEL_ROOT="$OUT_ROOT/_local_model_links"

ROOTS=(
  "outputs/baseline_chunk_lr_1e-4_100M"
  "outputs/baseline_pack_wrapped_lr_1e-4_100M"
  "outputs/baseline_gpt2_100M_v2"
)

# Full finetune: all multilingual finetuning languages.
LANGUAGES=(eng nld zho)
TASKS=(arc belebele bmlama include mmlu mnli sib200 truthfulqa xnli)

LR="5e-5"
PATIENCE="3"
BSZ="16"
MAX_EPOCHS="10"
SEED="12"
MAX_TRAIN_SAMPLES=""
MAX_EVAL_SAMPLES=""
MAX_PREDICT_SAMPLES=""
MAX_SEQ_LENGTH="128"

# ============================================================
# Cache settings, consistent with previous eval scripts.
# ============================================================

export HF_HOME=~/hf_cache_babylm_eval_100M
export HF_DATASETS_CACHE=~/hf_cache_babylm_eval_100M/datasets
export TRANSFORMERS_CACHE=~/hf_cache_babylm_eval_100M/transformers
export HF_HUB_CACHE=~/hf_cache_babylm_eval_100M/hub
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [ -f ~/.hf_token ]; then
  export HF_TOKEN=$(tr -d '\n\r' < ~/.hf_token)
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "[HF AUTH] HF_TOKEN loaded from ~/.hf_token"
else
  echo "[HF AUTH WARNING] ~/.hf_token not found. HuggingFace may rate limit anonymous requests."
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
mkdir -p "$OUT_ROOT" "$LOCAL_MODEL_ROOT"

# Keep online mode. The finetuning datasets are local, but HF may still need cached model/config helpers.
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

# ============================================================
# Utilities
# ============================================================

sanitize_name () {
  echo "$1" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9._-]#_#g'
}

fix_tokenizer_config () {
  CKPT_PATH="$1"

  if [ ! -f "$CKPT_PATH/tokenizer.json" ]; then
    echo "[TOKENIZER FIX SKIP] No tokenizer.json in $CKPT_PATH"
    return
  fi

  if [ -f "$CKPT_PATH/tokenizer_config.json" ] && [ ! -f "$CKPT_PATH/tokenizer_config.json.bak" ]; then
    cp "$CKPT_PATH/tokenizer_config.json" "$CKPT_PATH/tokenizer_config.json.bak"
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
new_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
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
(ckpt_dir / "special_tokens_map.json").write_text(json.dumps(special_map, indent=2, ensure_ascii=False))
print(f"[TOKENIZER FIXED] {ckpt_dir}")
PY
}

precheck_checkpoint () {
  CKPT_PATH="$1"
  OUT_DIR="$2"

  python - "$CKPT_PATH" > "$OUT_DIR/precheck.out" 2> "$OUT_DIR/precheck.err" <<'PY'
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

ckpt_dir = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)

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

make_model_arg () {
  CKPT_PATH="$1"
  ROOT_NAME="$2"
  MODEL_NAME="$3"
  UNIQUE_NAME="$4"

  # The custom finetuning script switches to GPTBertForSequenceClassification
  # only when the model path contains gpt_bert/gptbert. Non-GPT2 roots are
  # bertmask-style models, so give their symlink a gpt_bert marker.
  if [[ "$ROOT_NAME" == *gpt2* || "$MODEL_NAME" == *gpt2* || "$CKPT_PATH" == *gpt2* ]]; then
    LINK_NAME="$UNIQUE_NAME"
  else
    LINK_NAME="gpt_bert_${UNIQUE_NAME}"
  fi

  LINK_PATH="$LOCAL_MODEL_ROOT/$LINK_NAME"
  rm -f "$LINK_PATH"
  ln -s "$PROJECT_ROOT/$CKPT_PATH" "$LINK_PATH"
  echo "$LINK_PATH"
}

# ============================================================
# Run info
# ============================================================

echo "============================================================"
echo "BabyLM 100M finetune FULL run"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Project root: $PROJECT_ROOT"
echo "Finetune dir: $FINETUNE_DIR"
echo "Output root: $OUT_ROOT"
echo "Languages: ${LANGUAGES[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Params: lr=$LR bsz=$BSZ max_epochs=$MAX_EPOCHS patience=$PATIENCE seed=$SEED"
echo "Sample limits: train=$MAX_TRAIN_SAMPLES eval=$MAX_EVAL_SAMPLES predict=$MAX_PREDICT_SAMPLES"
echo "HF_HOME: $HF_HOME"
echo "============================================================"

if [ ! -f "$FINETUNE_PY" ]; then
  echo "[FATAL] Missing finetune script: $FINETUNE_PY"
  exit 1
fi

# ============================================================
# Main loop
# ============================================================

for ROOT_DIR in "${ROOTS[@]}"; do
  if [ ! -d "$ROOT_DIR" ]; then
    echo "[SKIP ROOT] Missing root: $ROOT_DIR"
    continue
  fi

  ROOT_NAME=$(basename "$ROOT_DIR")
  echo ""
  echo "[ROOT] $ROOT_NAME -> $ROOT_DIR"

  while IFS= read -r CKPT_PATH; do
    MODEL_DIR=$(dirname "$(dirname "$CKPT_PATH")")
    MODEL_NAME=$(basename "$MODEL_DIR")
    CKPT_NAME="final_model"
    UNIQUE_NAME=$(sanitize_name "${ROOT_NAME}__${MODEL_NAME}__${CKPT_NAME}")
    CKPT_EVAL_DIR="$OUT_ROOT/$ROOT_NAME/$MODEL_NAME/$CKPT_NAME"

    mkdir -p "$CKPT_EVAL_DIR"

    echo ""
    echo "------------------------------------------------------------"
    echo "Checkpoint: $CKPT_PATH"
    echo "Output:     $CKPT_EVAL_DIR"
    echo "Unique:     $UNIQUE_NAME"
    echo "------------------------------------------------------------"

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

    fix_tokenizer_config "$CKPT_PATH"
    precheck_checkpoint "$CKPT_PATH" "$CKPT_EVAL_DIR"
    PRECHECK_EXIT=$?
    if [ $PRECHECK_EXIT -ne 0 ]; then
      echo "[SKIP] Precheck failed: $ROOT_NAME / $MODEL_NAME"
      {
        echo "SKIPPED: precheck failed"
        echo "checkpoint_path: $CKPT_PATH"
        echo "see: $CKPT_EVAL_DIR/precheck.err"
        tail -120 "$CKPT_EVAL_DIR/precheck.err"
      } > "$CKPT_EVAL_DIR/checkpoint_status.txt"
      continue
    fi

    MODEL_ARG=$(make_model_arg "$CKPT_PATH" "$ROOT_NAME" "$MODEL_NAME" "$UNIQUE_NAME")

    {
      echo "root_name: $ROOT_NAME"
      echo "root_dir: $ROOT_DIR"
      echo "model_name: $MODEL_NAME"
      echo "model_dir: $MODEL_DIR"
      echo "checkpoint_name: $CKPT_NAME"
      echo "checkpoint_path: $CKPT_PATH"
      echo "model_arg: $MODEL_ARG"
      echo "eval_dir: $CKPT_EVAL_DIR"
      echo "date: $(date)"
      echo "host: $(hostname)"
      echo "conda_env: ${CONDA_DEFAULT_ENV:-}"
      echo "hf_home: $HF_HOME"
      echo "git_commit:"
      git rev-parse HEAD 2>/dev/null || echo "unknown"
      echo ""
      echo "precheck.out:"
      cat "$CKPT_EVAL_DIR/precheck.out"
    } > "$CKPT_EVAL_DIR/checkpoint_info.txt"
    echo "READY" > "$CKPT_EVAL_DIR/checkpoint_status.txt"

    for LANGUAGE in "${LANGUAGES[@]}"; do
      for TASK in "${TASKS[@]}"; do
        TRAIN_FILE="$DATA_ROOT/$TASK/$LANGUAGE/${TASK}_${LANGUAGE}.train.jsonl"
        VALID_FILE="$DATA_ROOT/$TASK/$LANGUAGE/${TASK}_${LANGUAGE}.valid.jsonl"

        TASK_DIR="$CKPT_EVAL_DIR/$LANGUAGE/$TASK"
        LOG_DIR="$TASK_DIR/logs"
        STATUS_FILE="$TASK_DIR/status.txt"
        RESULT_DIR="$TASK_DIR/results"
        mkdir -p "$RESULT_DIR" "$LOG_DIR"

        if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VALID_FILE" ]; then
          echo "[SKIP TASK] Missing data: $LANGUAGE / $TASK"
          {
            echo "SKIPPED: missing train/valid data"
            echo "train_file: $TRAIN_FILE"
            echo "valid_file: $VALID_FILE"
          } > "$STATUS_FILE"
          continue
        fi

        if [ -f "$STATUS_FILE" ] && grep -q "^SUCCESS" "$STATUS_FILE"; then
          echo "[SKIP] Already successful: $ROOT_NAME / $MODEL_NAME / $LANGUAGE / $TASK"
          continue
        fi

        rm -f "$LOG_DIR/finetune.out" "$LOG_DIR/finetune.err" "$STATUS_FILE"
        START_TIME=$(date +%s)

        echo "[RUN] $ROOT_NAME / $MODEL_NAME / $LANGUAGE / $TASK"
        {
          echo "RUNNING"
          echo "started_at: $(date)"
          echo "root_name: $ROOT_NAME"
          echo "model_name: $MODEL_NAME"
          echo "checkpoint_path: $CKPT_PATH"
          echo "model_arg: $MODEL_ARG"
          echo "language: $LANGUAGE"
          echo "task: $TASK"
          echo "train_file: $TRAIN_FILE"
          echo "valid_file: $VALID_FILE"
          echo "result_dir: $RESULT_DIR"
        } > "$STATUS_FILE"

        EXTRA_SAMPLE_ARGS=()
        if [ -n "$MAX_TRAIN_SAMPLES" ]; then
          EXTRA_SAMPLE_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
        fi
        if [ -n "$MAX_EVAL_SAMPLES" ]; then
          EXTRA_SAMPLE_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
        fi
        if [ -n "$MAX_PREDICT_SAMPLES" ]; then
          EXTRA_SAMPLE_ARGS+=(--max_predict_samples "$MAX_PREDICT_SAMPLES")
        fi

        python3 "$FINETUNE_PY" \
          --model_name_or_path "$MODEL_ARG" \
          --language "$LANGUAGE" \
          --output_dir "$RESULT_DIR" \
          --train_file "$TRAIN_FILE" \
          --validation_file "$VALID_FILE" \
          --do_train True \
          --do_eval \
          --do_predict \
          --max_seq_length "$MAX_SEQ_LENGTH" \
          --per_device_train_batch_size "$BSZ" \
          --learning_rate "$LR" \
          --num_train_epochs "$MAX_EPOCHS" \
          --patience "$PATIENCE" \
          --eval_strategy epoch \
          --save_strategy epoch \
          --overwrite_output_dir \
          --seed "$SEED" \
          --cache_dir "$HF_HOME" \
          "${EXTRA_SAMPLE_ARGS[@]}" \
          > "$LOG_DIR/finetune.out" \
          2> "$LOG_DIR/finetune.err"

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
            echo "checkpoint_path: $CKPT_PATH"
            echo "model_arg: $MODEL_ARG"
            echo "language: $LANGUAGE"
            echo "task: $TASK"
            echo "result_dir: $RESULT_DIR"
          } > "$STATUS_FILE"
          echo "[SUCCESS] $ROOT_NAME / $MODEL_NAME / $LANGUAGE / $TASK"
        else
          {
            echo "FAILED"
            echo "finished_at: $(date)"
            echo "duration_seconds: $DURATION"
            echo "exit_code: $EXIT_CODE"
            echo "root_name: $ROOT_NAME"
            echo "model_name: $MODEL_NAME"
            echo "checkpoint_path: $CKPT_PATH"
            echo "model_arg: $MODEL_ARG"
            echo "language: $LANGUAGE"
            echo "task: $TASK"
            echo "result_dir: $RESULT_DIR"
            echo ""
            echo "Last 120 lines of stderr:"
            tail -120 "$LOG_DIR/finetune.err"
          } > "$STATUS_FILE"
          echo "[FAILED] $ROOT_NAME / $MODEL_NAME / $LANGUAGE / $TASK"
          echo "See: $LOG_DIR/finetune.err"
        fi
      done
    done

  done < <(find "$ROOT_DIR" -type d -path "*/checkpoints/final_model" | sort)
done

echo ""
echo "============================================================"
echo "All 100M finetune FULL runs finished at: $(date)"
echo "============================================================"
