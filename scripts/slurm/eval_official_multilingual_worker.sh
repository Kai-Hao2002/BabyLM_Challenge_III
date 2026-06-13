#!/bin/bash
#SBATCH --job-name=babylm_eval
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --ntasks=1

set +e
shopt -s nullglob

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/BabyLM_Challenge_III}"
CONDA_ENV="${CONDA_ENV:-babylm}"
RUN_ID="${RUN_ID:-manual}"
EVAL_MODEL_ID="${EVAL_MODEL_ID:?EVAL_MODEL_ID is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_NAME="${MODEL_NAME:-unknown_model}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-unknown_root}"
LANGS="${LANGS:-eng nld zho}"
BATCH_SIZE="${BATCH_SIZE:-auto:10}"

FT_LR="${FT_LR:-5e-5}"
FT_BSZ="${FT_BSZ:-16}"
FT_MAX_EPOCHS="${FT_MAX_EPOCHS:-10}"
FT_PATIENCE="${FT_PATIENCE:-3}"
FT_SEED="${FT_SEED:-12}"
FT_MAX_SEQ_LENGTH="${FT_MAX_SEQ_LENGTH:-128}"

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs/slurm

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PROJECT_ROOT="$PWD"
OFFICIAL_DIR="$PROJECT_ROOT/external/babylm-eval/multilingual"
DATA_ROOT="$OFFICIAL_DIR/finetune/data/multilingual"
OUT_ROOT="$PROJECT_ROOT/outputs/evaluations/official_multilingual"
MODEL_OUT="$OUT_ROOT/models/$EVAL_MODEL_ID"
STATUS_ROOT="$MODEL_OUT/status"
LOG_ROOT="$MODEL_OUT/logs"
ZS_ROOT="$MODEL_OUT/zeroshot"
FT_ROOT="$MODEL_OUT/finetune"
COLLATED_ROOT="$MODEL_OUT/collated"
LINK_ROOT="$OUT_ROOT/_official_links"
MODEL_LINK_ROOT="$LINK_ROOT/model_links"
LM_EVAL_COMPAT="$PROJECT_ROOT/scripts/evaluation/lm_eval_compat.py"
FINETUNE_COMPAT="$PROJECT_ROOT/scripts/evaluation/finetune_compat.py"
OFFICIAL_ZS_ROOT="$OFFICIAL_DIR/results/main"
OFFICIAL_ZS_MODEL_DIR="$OFFICIAL_ZS_ROOT/$EVAL_MODEL_ID"
OFFICIAL_FT_MODEL_DIR="$OFFICIAL_DIR/finetune/results/$EVAL_MODEL_ID"

mkdir -p "$STATUS_ROOT" "$LOG_ROOT" "$ZS_ROOT" "$FT_ROOT" "$COLLATED_ROOT"
mkdir -p "$MODEL_LINK_ROOT" "$OFFICIAL_ZS_MODEL_DIR" "$OFFICIAL_DIR/finetune/results"

export HF_HOME="${HF_HOME:-$HOME/hf_cache_babylm_eval}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN
  HF_TOKEN=$(tr -d '\n\r' < "$HOME/.hf_token")
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

sanitize_name () {
  echo "$1" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9._-]#_#g'
}

is_gpt_style () {
  local config_file="$1/config.json"
  if [ ! -f "$config_file" ]; then
    return 1
  fi
  python3 - "$config_file" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)

text = " ".join(cfg.get("architectures", [])) + " " + cfg.get("model_type", "")
if any(key in text.lower() for key in ("gpt", "causal", "llama")):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

prepare_model_arg () {
  local abs_model_path="$1"
  local link_name="$EVAL_MODEL_ID"
  if ! is_gpt_style "$abs_model_path"; then
    link_name="gpt_bert_${EVAL_MODEL_ID}"
  fi
  local link_path="$MODEL_LINK_ROOT/$link_name"
  rm -f "$link_path"
  ln -s "$abs_model_path" "$link_path"
  echo "$link_path"
}

write_model_info () {
  python3 - "$MODEL_OUT/model_info.json" <<PY
import json
from pathlib import Path

info = {
    "run_id": "$RUN_ID",
    "eval_model_id": "$EVAL_MODEL_ID",
    "experiment_root": "$EXPERIMENT_ROOT",
    "model_name": "$MODEL_NAME",
    "model_path": "$MODEL_PATH",
    "model_arg": "$MODEL_ARG",
    "official_dir": "$OFFICIAL_DIR",
}
config_path = Path("$ABS_MODEL_PATH") / "config.json"
if config_path.exists():
    info["config"] = json.loads(config_path.read_text())
Path("$MODEL_OUT/model_info.json").write_text(json.dumps(info, indent=2) + "\\n")
PY
}

write_environment_info () {
  python3 - "$MODEL_OUT/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

packages = ["torch", "transformers", "lm-eval", "datasets", "accelerate", "evaluate"]
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

info = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": versions,
}
Path(sys.argv[1]).write_text(json.dumps(info, indent=2) + "\n")
PY
}

preflight_environment () {
  local status_file="$STATUS_ROOT/environment.status"
  write_environment_info

  python3 - "$LM_EVAL_COMPAT" "$FINETUNE_COMPAT" <<'PY'
import importlib.util
import importlib.metadata
import sys
from pathlib import Path

for module in ("torch", "transformers", "lm_eval", "datasets", "accelerate", "evaluate"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Missing required Python module: {module}")

for path in sys.argv[1:]:
    if not Path(path).is_file():
        raise SystemExit(f"Missing compatibility runner: {path}")

transformers_version = importlib.metadata.version("transformers")
try:
    transformers_major = int(transformers_version.split(".", 1)[0])
except ValueError:
    raise SystemExit(f"Cannot parse Transformers version: {transformers_version}")
if transformers_major >= 5:
    raise SystemExit(
        "BabyLM lm-eval 0.4.9 requires Transformers 4.x; found "
        f"{transformers_version}. Install requirements-evaluation.txt."
    )
PY
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    mark_status "$status_file" "SUCCESS" "environment_file: $MODEL_OUT/environment.json"
  else
    mark_status "$status_file" "FAILED" "exit_code: $rc" "environment_file: $MODEL_OUT/environment.json"
  fi
  return "$rc"
}

mark_status () {
  local file="$1"
  local state="$2"
  shift 2
  {
    echo "$state"
    echo "time: $(date)"
    echo "host: $(hostname)"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    for item in "$@"; do
      echo "$item"
    done
  } > "$file"
}

copy_latest_lmeval_result () {
  local search_dir="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  local latest
  latest=$(find "$search_dir" -type f -name 'results_*.json' -print 2>/dev/null | sort | tail -n 1)
  if [ -n "$latest" ]; then
    cp "$latest" "$target_dir/"
  fi
}

run_zeroshot_lang () {
  local lang="$1"
  local task="zeroshot_$lang"
  local task_dir="$ZS_ROOT/$lang"
  local result_dir="$task_dir/results"
  local log_dir="$LOG_ROOT/zeroshot/$lang"
  local status_file="$STATUS_ROOT/zeroshot_$lang.status"
  mkdir -p "$result_dir" "$log_dir"

  if [ -f "$status_file" ] && grep -q '^SUCCESS$' "$status_file"; then
    echo "[SKIP DONE] zeroshot $lang"
    return 0
  fi

  echo "[RUN] zeroshot $lang"
  mark_status "$status_file" "RUNNING" "task: $task"
  cd "$OFFICIAL_DIR" || return 1
  python3 "$LM_EVAL_COMPAT" \
    --model hf \
    --model_args "pretrained=$MODEL_ARG,tokenizer=$MODEL_ARG,trust_remote_code=True" \
    --tasks "$task" \
    --device cuda \
    --output_path "$result_dir" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot 0 \
    --log_samples \
    --include_path tasks/ \
    > "$log_dir/eval.out" 2> "$log_dir/eval.err"
  local rc=$?
  cd "$PROJECT_ROOT" || exit 1

  if [ "$rc" -eq 0 ]; then
    copy_latest_lmeval_result "$result_dir" "$OFFICIAL_ZS_MODEL_DIR"
    mark_status "$status_file" "SUCCESS" "task: $task" "result_dir: $result_dir"
    echo "[SUCCESS] zeroshot $lang"
  else
    mark_status "$status_file" "FAILED" "task: $task" "exit_code: $rc" "stderr_tail: $(tail -n 40 "$log_dir/eval.err" | tr '\n' '|')"
    echo "[FAILED] zeroshot $lang"
  fi
}

tasks_for_language () {
  case "$1" in
    en) echo "arc belebele bmlama mnli sib200 truthfulqa xnli" ;;
    nl) echo "arc belebele bmlama include mnli sib200 truthfulqa" ;;
    zh) echo "arc belebele bmlama include mnli sib200 truthfulqa xnli" ;;
    *) echo "" ;;
  esac
}

short_lang () {
  case "$1" in
    eng) echo "en" ;;
    nld) echo "nl" ;;
    zho) echo "zh" ;;
    *) echo "$1" ;;
  esac
}

ensure_finetune_link () {
  if [ -L "$OFFICIAL_FT_MODEL_DIR" ]; then
    rm -f "$OFFICIAL_FT_MODEL_DIR"
  fi
  if [ -e "$OFFICIAL_FT_MODEL_DIR" ] && [ ! -L "$OFFICIAL_FT_MODEL_DIR" ]; then
    echo "[WARN] Official finetune dir exists and is not a symlink: $OFFICIAL_FT_MODEL_DIR"
  else
    ln -s "$FT_ROOT" "$OFFICIAL_FT_MODEL_DIR"
  fi
}

run_finetune_task () {
  local lang="$1"
  local task="$2"
  local train_file="$DATA_ROOT/$lang/$task/${task}_${lang}.train.jsonl"
  local valid_file="$DATA_ROOT/$lang/$task/${task}_${lang}.valid.jsonl"
  local task_dir="$FT_ROOT/$lang/$task"
  local log_dir="$LOG_ROOT/finetune/$lang/$task"
  local status_file="$STATUS_ROOT/finetune_${lang}_${task}.status"
  mkdir -p "$task_dir" "$log_dir"

  if [ -f "$status_file" ] && grep -q '^SUCCESS$' "$status_file"; then
    echo "[SKIP DONE] finetune $lang/$task"
    return 0
  fi

  if [ ! -f "$train_file" ] || [ ! -f "$valid_file" ]; then
    mark_status "$status_file" "SKIPPED" "reason: missing train/valid data" "train_file: $train_file" "valid_file: $valid_file"
    echo "[SKIP MISSING DATA] finetune $lang/$task"
    return 0
  fi

  echo "[RUN] finetune $lang/$task"
  mark_status "$status_file" "RUNNING" "task: $task" "language: $lang"
  cd "$OFFICIAL_DIR" || return 1
  python3 "$FINETUNE_COMPAT" "$OFFICIAL_DIR/finetune/finetune_classification.py" \
    --model_name_or_path "$MODEL_ARG" \
    --language "$lang" \
    --output_dir "finetune/results/$EVAL_MODEL_ID/$lang/$task" \
    --train_file "$train_file" \
    --validation_file "$valid_file" \
    --do_train True \
    --do_eval \
    --do_predict \
    --max_seq_length "$FT_MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$FT_BSZ" \
    --learning_rate "$FT_LR" \
    --num_train_epochs "$FT_MAX_EPOCHS" \
    --patience "$FT_PATIENCE" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --seed "$FT_SEED" \
    > "$log_dir/finetune.out" 2> "$log_dir/finetune.err"
  local rc=$?
  cd "$PROJECT_ROOT" || exit 1

  if [ "$rc" -eq 0 ]; then
    mark_status "$status_file" "SUCCESS" "task: $task" "language: $lang" "result_dir: $task_dir"
    echo "[SUCCESS] finetune $lang/$task"
  else
    mark_status "$status_file" "FAILED" "task: $task" "language: $lang" "exit_code: $rc" "stderr_tail: $(tail -n 40 "$log_dir/finetune.err" | tr '\n' '|')"
    echo "[FAILED] finetune $lang/$task"
  fi
}

collate_results () {
  local log_dir="$LOG_ROOT/collate"
  local status_file="$STATUS_ROOT/collate.status"
  local submission_file="$COLLATED_ROOT/${EVAL_MODEL_ID}_submission.json"
  local predictions_file="$COLLATED_ROOT/${EVAL_MODEL_ID}_predictions.json"
  mkdir -p "$log_dir"
  rm -f "$submission_file" "$predictions_file"
  cd "$OFFICIAL_DIR" || return 1
  python3 scripts/collate_results.py \
    --model_name "$EVAL_MODEL_ID" \
    --output "$submission_file" \
    --output_predictions "$predictions_file" \
    > "$log_dir/collate.out" 2> "$log_dir/collate.err"
  local rc=$?
  cd "$PROJECT_ROOT" || exit 1
  if [ "$rc" -eq 0 ] && [ -s "$submission_file" ] && [ -s "$predictions_file" ]; then
    mark_status "$status_file" "SUCCESS" "submission: $submission_file" "predictions: $predictions_file"
    echo "[SUCCESS] collate"
  else
    mark_status "$status_file" "FAILED" "exit_code: $rc" "submission_exists: $([ -s "$submission_file" ] && echo yes || echo no)" "predictions_exists: $([ -s "$predictions_file" ] && echo yes || echo no)" "stderr_tail: $(tail -n 40 "$log_dir/collate.err" | tr '\n' '|')"
    echo "[FAILED] collate"
  fi
}

if [[ "$MODEL_PATH" = /* ]]; then
  ABS_MODEL_PATH="$MODEL_PATH"
else
  ABS_MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"
fi

if [ ! -d "$ABS_MODEL_PATH" ]; then
  echo "[FATAL] Missing model path: $ABS_MODEL_PATH"
  exit 1
fi

MODEL_ARG=$(prepare_model_arg "$ABS_MODEL_PATH")
ensure_finetune_link
write_model_info
if ! preflight_environment; then
  echo "[FATAL] Evaluation environment preflight failed. See $MODEL_OUT/environment.json"
  exit 1
fi

echo "============================================================"
echo "BabyLM official multilingual evaluation"
echo "Run id: $RUN_ID"
echo "Eval model id: $EVAL_MODEL_ID"
echo "Model path: $ABS_MODEL_PATH"
echo "Model arg: $MODEL_ARG"
echo "Output: $MODEL_OUT"
echo "Languages: $LANGS"
echo "Conda env: $CONDA_ENV"
echo "Started: $(date)"
echo "============================================================"

for lang in $LANGS; do
  run_zeroshot_lang "$lang"
done

for long_lang in $LANGS; do
  lang=$(short_lang "$long_lang")
  for task in $(tasks_for_language "$lang"); do
    run_finetune_task "$lang" "$task"
  done
done

collate_results

echo "============================================================"
echo "Finished: $(date)"
echo "Output: $MODEL_OUT"
echo "============================================================"
