# Multilingual Evaluation Quickstart

This note describes the smallest reliable path for running the
`babylm-org/multilingual-evaluation` pipeline from this project.

## 1. Prepare the Evaluation Pipeline

The official multilingual evaluation repository should live under:

```bash
external/multilingual-evaluation
```

If it is not present yet, add it as a submodule:

```bash
git submodule add https://github.com/babylm-org/multilingual-evaluation external/multilingual-evaluation
git submodule update --init --recursive
```

## 2. Create a Local Evaluation Environment

For local smoke tests, use a project-local virtual environment:

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install datasets accelerate evaluate lm-eval==0.4.9
```

`lm-eval==0.4.9` matches the version pinned by the official multilingual
evaluation repository. Do not install the full official requirements locally
unless you specifically need them; `vllm` and `flash_attn` are CUDA-oriented and
are better installed on LRZ.

## 3. Make Sure a Checkpoint Exists

The evaluation command expects a HuggingFace-compatible checkpoint, for example:

```bash
outputs/checkpoints/final_model
```

The directory should contain files like:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

If you need to run a local training smoke test first:

```bash
.venv/bin/python main.py --config configs/experiment_10M.yaml
```

For real training data, you must first authenticate with Hugging Face and accept
access to the gated BabyLM datasets:

```bash
huggingface-cli login
```

Then run:

```bash
.venv/bin/python src/data_pipeline/download.py
.venv/bin/python src/data_pipeline/clean_and_mix.py
.venv/bin/python main.py --config configs/experiment_10M.yaml
```

## 4. Run a Minimal Local Evaluation

Use a very small limit first. This verifies that the model checkpoint, tokenizer,
task path, and evaluation data all connect correctly.

```bash
.venv/bin/lm_eval \
  --model hf \
  --model_args pretrained=outputs/checkpoints/final_model \
  --include_path external/multilingual-evaluation/tasks \
  --tasks babybabellm_eng \
  --batch_size 1 \
  --limit 1 \
  --device cpu \
  --output_path outputs/eval/smoke_eng
```

The output will be written under:

```text
outputs/eval/smoke_eng/
```

The first run may take a while because `lm_eval` downloads datasets from
Hugging Face and expands all subtasks inside `babybabellm_eng`.

## 5. Task Names

The current multilingual evaluation checkout uses three-letter language codes.
For our project languages, use:

```text
babybabellm_eng
babybabellm_nld
babybabellm_zho
```

For example:

```bash
.venv/bin/lm_eval \
  --model hf \
  --model_args pretrained=outputs/checkpoints/final_model \
  --include_path external/multilingual-evaluation/tasks \
  --tasks babybabellm_nld \
  --batch_size 1 \
  --limit 1 \
  --device cpu \
  --output_path outputs/eval/smoke_nld
```

## 6. Important Notes for Our BERT Model

Our current model is `BertForMaskedLM`, but the 2026 multilingual evaluation
pipeline currently works through the standard `hf` backend:

```bash
--model hf
```

Do not use these commands for the current pipeline:

```bash
--model hf-mlm
--model_args pretrained=...,backend=mlm
```

In the tested `lm-eval` versions, `hf-mlm` is not registered and `backend=mlm`
is rejected. The smoke test can still run with `--model hf`, but before treating
the numbers as final official results, confirm whether the organizers expect
causal-LM scoring or masked-LM pseudo-log-likelihood scoring for BERT-style
submissions.

## 7. Full Evaluation on LRZ

For full runs, use LRZ rather than a laptop:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/final_model \
  --include_path /path/to/multilingual-evaluation/tasks \
  --tasks babybabellm_eng \
  --batch_size auto \
  --device cuda \
  --output_path /path/to/outputs/eval/final_eng \
  --log_samples
```

Repeat for:

```text
babybabellm_eng
babybabellm_nld
babybabellm_zho
```

Only remove `--limit` for real evaluation runs.
