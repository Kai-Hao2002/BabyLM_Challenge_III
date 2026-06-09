# LRZ Official Multilingual Evaluation

This is the project wrapper around the official BabyLM 2026 multilingual
evaluation pipeline in `external/babylm-eval/multilingual`.

It evaluates local HuggingFace checkpoints under `outputs/` and writes all
project-facing outputs under:

```text
outputs/evaluations/official_multilingual/
```

## Dry Run

Inspect discovered models without submitting Slurm jobs:

```bash
python scripts/lrz_eval_launcher.py
```

Limit to one model while testing:

```bash
python scripts/lrz_eval_launcher.py --limit 1
```

Filter by experiment/model name:

```bash
python scripts/lrz_eval_launcher.py --include '100M.*gpt2'
python scripts/lrz_eval_launcher.py --roots outputs/18k_baseline_chunk_10M
```

Explicitly name several models:

```bash
python scripts/lrz_eval_launcher.py \
  --models \
  outputs/18k_baseline_chunk_10M/baseline_chunk_32_lr_1e-4_bertmask \
  outputs/18k_baseline_chunk_10M/baseline_chunk_64_lr_1e-4_bertmask
```

`--models` also accepts full checkpoint paths:

```bash
python scripts/lrz_eval_launcher.py \
  --models \
  outputs/18k_baseline_chunk_10M/baseline_chunk_32_lr_1e-4_bertmask/checkpoints/final_model \
  outputs/34k_baseline_gpt2_100M/baseline_pack_wrapped_128_lr_1e-4_gpt2/checkpoints/final_model
```

The launcher scans for:

```text
outputs/*/*/checkpoints/final_model
```

## Submit Jobs

Submit one Slurm job per discovered model:

```bash
python scripts/lrz_eval_launcher.py --submit
```

Common LRZ overrides:

```bash
python scripts/lrz_eval_launcher.py \
  --roots outputs/34k_baseline_gpt2_100M \
  --partition lrz-v100x2 \
  --time 24:00:00 \
  --mem 64G \
  --cpus-per-task 8 \
  --submit
```

The worker activates:

```bash
conda activate babylm
```

Override if needed:

```bash
python scripts/lrz_eval_launcher.py --conda-env babylm_eval --submit
```

## What Each Job Runs

Zero-shot languages:

```text
eng nld zho
```

Finetune tasks:

```text
en: arc belebele bmlama mnli sib200 truthfulqa xnli
nl: arc belebele bmlama include mnli sib200 truthfulqa
zh: arc belebele bmlama include mnli sib200 truthfulqa xnli
```

The worker then runs official collation and writes:

```text
outputs/evaluations/official_multilingual/models/<eval_model_id>/collated/
```

## Output Layout

For each model:

```text
outputs/evaluations/official_multilingual/models/<eval_model_id>/
├── model_info.json
├── zeroshot/
├── finetune/
├── collated/
│   ├── <eval_model_id>_submission.json
│   └── <eval_model_id>_predictions.json
├── logs/
└── status/
```

Per-run manifests are stored in:

```text
outputs/evaluations/official_multilingual/runs/<run_id>/
├── launcher_config.json
├── manifest.jsonl
└── submitted_jobs.tsv
```

## Re-running

The worker skips tasks whose status file starts with `SUCCESS`. To rerun a
single failed model, submit with an include regex:

```bash
python scripts/lrz_eval_launcher.py --include '34k_baseline_gpt2_100M/.+128' --submit
```

To force rerun a successful task, remove the relevant status file under:

```text
outputs/evaluations/official_multilingual/models/<eval_model_id>/status/
```

## Notes

- The wrapper keeps official-compatible symlinks and copied result files under
  `external/babylm-eval/multilingual/results/main` and
  `external/babylm-eval/multilingual/finetune/results` so that the official
  `collate_results.py` can be used unchanged.
- Evaluation IDs use `--` as the root/model separator because the official
  collation code treats `__` specially in zero-shot result folder names.
- BERT-style non-GPT checkpoints are evaluated through a local `gpt_bert_...`
  symlink, matching the convention used by the existing project Slurm scripts.
