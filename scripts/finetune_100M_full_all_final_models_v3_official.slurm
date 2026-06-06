#!/bin/bash
#SBATCH --job-name=ft_full_100M_all_v3
#SBATCH --output=logs/ft_full_100M_all_v3_%j.out
#SBATCH --error=logs/ft_full_100M_all_v3_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# BabyLM 100M FULL finetune evaluation for all final_model checkpoints.
# Uses the updated official BabyLM eval repo:
#   external/babylm-eval/multilingual
#
# Design goals:
# - Traverse all three 100M roots and all */checkpoints/final_model directories.
# - Use unique symlink names so all checkpoints called final_model do not overwrite each other.
# - Store actual results under BabyLM_Challenge_III/outputs/evaluations_finetune_100M/full.
# - Keep official-compatible finetune/results/<model_name>/... path via symlink.
# - Skip tasks already marked SUCCESS.
# - Skip missing language-task data instead of killing the whole job.
# - Continue after individual task failures.

set +e
shopt -s nullglob

source ~/.bashrc
conda activate babylm_eval

cd ~/BabyLM_Challenge_III || exit 1
mkdir -p logs

PROJECT_ROOT="$PWD"
FINETUNE_DIR="$PROJECT_ROOT/external/babylm-eval/multilingual"
FINETUNE_PY="$FINETUNE_DIR/finetune/finetune_classification.py"
DATA_ROOT="$FINETUNE_DIR/finetune/data/multilingual"
OUT_ROOT="$PROJECT_ROOT/outputs/evaluations_finetune_100M/full"
LOCAL_MODEL_ROOT="$OUT_ROOT/_local_model_links"
STATUS_ROOT="$OUT_ROOT/status"
LOG_ROOT="$OUT_ROOT/logs"
RESULTS_ROOT="$OUT_ROOT/results"

ROOTS=(
  "outputs/baseline_chunk_lr_1e-4_100M"
  "outputs/baseline_pack_wrapped_lr_1e-4_100M"
  "outputs/baseline_gpt2_100M_v2"
)

# Official finetune_model.sh accepts eng/nld/zho and internally maps them to en/nl/zh.
# Here we run the Python script directly, so we use the mapped ISO-1 codes.
LANGUAGES=(en nl zh)
TASKS=(arc belebele bmlama include mnli sib200 truthfulqa xnli)

LR="5e-5"
PATIENCE="3"
BSZ="16"
MAX_EPOCHS="10"
SEED="12"
MAX_SEQ_LENGTH="128"
# Full run: no sample limits.

# Reuse your existing HF cache, as requested.
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
mkdir -p "$OUT_ROOT" "$LOCAL_MODEL_ROOT" "$STATUS_ROOT" "$LOG_ROOT" "$RESULTS_ROOT"

# Keep online mode. Local models/data are used, but transformers may still need helpers from cache.
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

sanitize_name () {
  echo "$1" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9._-]#_#g'
}

is_gpt2_model () {
  local path="$1"
  local name="$2"
  if [[ "$path" == *gpt2* || "$name" == *gpt2* ]]; then
    return 0
  fi
  return 1
}

make_model_link () {
  local ckpt_path="$1"
  local root_name="$2"
  local model_name="$3"
  local unique_name="$4"

  local link_name
  # The finetune code uses GPTBertForSequenceClassification when the model path contains gpt_bert/gptbert.
  # Your chunk/pack bertmask models need this marker; GPT-2 models should not have it.
  if is_gpt2_model "$ckpt_path" "$model_name"; then
    link_name="$unique_name"
  else
    link_name="gpt_bert_${unique_name}"
  fi

  local link_path="$LOCAL_MODEL_ROOT/$link_name"
  rm -f "$link_path"
  ln -s "$PROJECT_ROOT/$ckpt_path" "$link_path"
  echo "$link_path"
}

ensure_official_result_symlink () {
  local model_basename="$1"
  local target_dir="$RESULTS_ROOT/$model_basename"
  local official_dir="$FINETUNE_DIR/finetune/results/$model_basename"

  mkdir -p "$target_dir"
  mkdir -p "$FINETUNE_DIR/finetune/results"

  if [ -L "$official_dir" ]; then
    rm -f "$official_dir"
  fi

  if [ -e "$official_dir" ] && [ ! -L "$official_dir" ]; then
    echo "[WARN] Official result dir already exists and is not a symlink: $official_dir"
    echo "[WARN] Leaving it untouched. The run will write there, then copy back may be needed."
  else
    ln -s "$target_dir" "$official_dir"
  fi
}

run_one_task () {
  local model_arg="$1"
  local model_basename="$2"
  local language="$3"
  local task="$4"
  local model_out="$OUT_ROOT/results/$model_basename"

  local train_file="$DATA_ROOT/$language/$task/${task}_${language}.train.jsonl"
  local valid_file="$DATA_ROOT/$language/$task/${task}_${language}.valid.jsonl"
  local test_file="$DATA_ROOT/$language/$task/${task}_${language}.test.jsonl"

  local task_status_dir="$STATUS_ROOT/$model_basename/$language/$task"
  local task_log_dir="$LOG_ROOT/$model_basename/$language/$task"
  local status_file="$task_status_dir/status.txt"
  mkdir -p "$task_status_dir" "$task_log_dir" "$model_out/$language/$task"

  if [ -f "$status_file" ] && grep -q '^SUCCESS$' "$status_file"; then
    echo "[SKIP DONE] $model_basename $language/$task"
    return 0
  fi

  if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ]; then
    {
      echo "SKIPPED: missing train/valid data"
      echo "train_file: $train_file"
      echo "valid_file: $valid_file"
      echo "time: $(date)"
    } > "$status_file"
    echo "[SKIP MISSING DATA] $model_basename $language/$task"
    return 0
  fi

  local predict_args=()
  if [ -f "$test_file" ]; then
    predict_args+=(--test_file "$test_file" --do_predict)
    :
  else
    echo "[WARN] Missing test file for $language/$task; running train/eval only. test_file=$test_file"
  fi

  local sample_args=()

  echo "RUNNING" > "$status_file"
  {
    echo "model_arg: $model_arg"
    echo "model_basename: $model_basename"
    echo "language: $language"
    echo "task: $task"
    echo "train_file: $train_file"
    echo "valid_file: $valid_file"
    echo "test_file: $test_file"
    echo "output_dir: finetune/results/$model_basename/$language/$task"
    echo "started: $(date)"
  } >> "$status_file"

  cd "$FINETUNE_DIR" || return 1

  python3 finetune/finetune_classification.py \
    --model_name_or_path "$model_arg" \
    --language "$language" \
    --output_dir "finetune/results/$model_basename/$language/$task" \
    --train_file "$train_file" \
    --validation_file "$valid_file" \
    --do_train True \
    --do_eval \
    "${predict_args[@]}" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$BSZ" \
    --learning_rate "$LR" \
    --num_train_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --seed "$SEED" \
    "${sample_args[@]}" \
    > "$task_log_dir/finetune.out" 2> "$task_log_dir/finetune.err"

  local rc=$?
  cd "$PROJECT_ROOT" || exit 1

  if [ "$rc" -eq 0 ]; then
    {
      echo "SUCCESS"
      echo "finished: $(date)"
    } > "$status_file"
    echo "[SUCCESS] $model_basename $language/$task"
  else
    {
      echo "FAILED"
      echo "exit_code: $rc"
      echo "finished: $(date)"
      echo "--- last 80 stderr lines ---"
      tail -n 80 "$task_log_dir/finetune.err"
    } > "$status_file"
    echo "[FAILED] $model_basename $language/$task exit_code=$rc"
  fi
}

collate_one_model () {
  local model_basename="$1"
  local collate_dir="$OUT_ROOT/collated/$model_basename"
  mkdir -p "$collate_dir"

  cd "$FINETUNE_DIR" || return 1
  python3 scripts/collate_results.py --model_name "$model_basename" \
    > "$collate_dir/collate.out" 2> "$collate_dir/collate.err"
  local rc=$?
  cd "$PROJECT_ROOT" || exit 1

  if [ "$rc" -eq 0 ]; then
    echo "[COLLATE SUCCESS] $model_basename"
  else
    echo "[COLLATE FAILED] $model_basename; see $collate_dir/collate.err"
  fi
}

# ============================================================
# Run info and preflight
# ============================================================

echo "============================================================"
echo "BabyLM 100M FULL finetune evaluation"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Project root: $PROJECT_ROOT"
echo "Official finetune dir: $FINETUNE_DIR"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUT_ROOT"
echo "Languages: ${LANGUAGES[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Params: lr=$LR bsz=$BSZ max_epochs=$MAX_EPOCHS patience=$PATIENCE seed=$SEED max_seq_length=$MAX_SEQ_LENGTH"
echo "Sample limits: none"
echo "HF_HOME: $HF_HOME"
echo "============================================================"

if [ ! -f "$FINETUNE_PY" ]; then
  echo "[FATAL] Missing official finetune script: $FINETUNE_PY"
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "[FATAL] Missing official finetune data root: $DATA_ROOT"
  echo "Check whether external/babylm-eval was cloned completely."
  exit 1
fi

echo "[DATA PREFLIGHT] Example train/valid/test files:"
find "$DATA_ROOT" -maxdepth 4 -type f \( -name "*.train.jsonl" -o -name "*.valid.jsonl" -o -name "*.test.jsonl" \) | head -40

echo "[MODEL PREFLIGHT] Searching final_model checkpoints..."
mapfile -t FINAL_MODELS < <(
  for ROOT_DIR in "${ROOTS[@]}"; do
    if [ -d "$ROOT_DIR" ]; then
      find "$ROOT_DIR" -mindepth 3 -maxdepth 3 -type d -path "*/checkpoints/final_model"
    else
      echo "[WARN] Missing root: $ROOT_DIR" >&2
    fi
  done | sort
)

if [ "${#FINAL_MODELS[@]}" -eq 0 ]; then
  echo "[FATAL] No final_model checkpoints found under configured roots."
  exit 1
fi

printf '[MODEL FOUND] %s\n' "${FINAL_MODELS[@]}"

# ============================================================
# Main loop
# ============================================================

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

for CKPT_PATH in "${FINAL_MODELS[@]}"; do
  ROOT_NAME="unknown_root"
  for R in "${ROOTS[@]}"; do
    if [[ "$CKPT_PATH" == "$R"/* ]]; then
      ROOT_NAME=$(basename "$R")
      break
    fi
  done

  MODEL_NAME=$(basename "$(dirname "$(dirname "$CKPT_PATH")")")
  UNIQUE_BASE=$(sanitize_name "${ROOT_NAME}__${MODEL_NAME}__final_model")
  MODEL_ARG=$(make_model_link "$CKPT_PATH" "$ROOT_NAME" "$MODEL_NAME" "$UNIQUE_BASE")
  MODEL_BASENAME=$(basename "$MODEL_ARG")

  ensure_official_result_symlink "$MODEL_BASENAME"

  echo "------------------------------------------------------------"
  echo "Checkpoint: $CKPT_PATH"
  echo "Model name: $MODEL_NAME"
  echo "Model arg: $MODEL_ARG"
  echo "Result basename: $MODEL_BASENAME"
  echo "Results: $OUT_ROOT/results/$MODEL_BASENAME"
  echo "------------------------------------------------------------"

  for LANGUAGE in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
      TOTAL=$((TOTAL + 1))
      BEFORE_STATUS="$STATUS_ROOT/$MODEL_BASENAME/$LANGUAGE/$TASK/status.txt"
      run_one_task "$MODEL_ARG" "$MODEL_BASENAME" "$LANGUAGE" "$TASK"
      if [ -f "$BEFORE_STATUS" ] && grep -q '^SUCCESS$' "$BEFORE_STATUS"; then
        SUCCESS=$((SUCCESS + 1))
      elif [ -f "$BEFORE_STATUS" ] && grep -q '^FAILED$' "$BEFORE_STATUS"; then
        FAILED=$((FAILED + 1))
      elif [ -f "$BEFORE_STATUS" ] && grep -q '^SKIPPED:' "$BEFORE_STATUS"; then
        SKIPPED=$((SKIPPED + 1))
      fi
    done
  done

  collate_one_model "$MODEL_BASENAME"
done

echo "============================================================"
echo "Finished at: $(date)"
echo "Total attempted task slots: $TOTAL"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"
echo "Skipped missing data: $SKIPPED"
echo "Output root: $OUT_ROOT"
echo "============================================================"
