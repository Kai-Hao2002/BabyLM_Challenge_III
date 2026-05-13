# BabyLM 2026 - Multilingual Track (100M & 10M)

This repository contains the implementation for the 2026 BabyLM Challenge, MultiLingual Track. Our goal is to train a highly sample-efficient trilingual (Chinese, English, Dutch) language model using the BabyBabelLM dataset, strictly under the 100M token limit. We employ a "validate on 10M, scale to 100M" strategy and utilize the LRZ cluster for distributed training.

## 📂 Project Structure

```text
.
├── data/                       # Dataset storage
│   ├── raw/                    # Raw BabyBabelLM trilingual data
│   ├── processed_10M/          # Processed 10M mixed data for testing
│   └── processed_100M/         # Processed 100M official mixed data
│
├── tokenizers/                 # Tokenizer models
│   ├── tokenizer_10M.json      # BPE tokenizer trained on 10M
│   └── tokenizer_100M.json     # BPE tokenizer trained on 100M
│
├── src/                        # Core source code
│   ├── data_pipeline/          # Data processing modules(A)
│   │   ├── download.py         # Data downloading script
│   │   ├── clean_and_mix.py    # Data cleaning and ratio mixing
│   │   └── train_tokenizer.py  # Tokenizer training script
│   │
│   ├── model/                  # Model architecture modules(B)
│   │   ├── architecture.py     # (Small LLaMA-style) / Model definition
│   │   └── config.py           # Model configuration variables
│   │
│   └── training/               # Core training modules(B)
│       ├── trainer.py          # Training loop & Checkpointing
│       └── dataset.py          # PyTorch Dataset/DataLoader
│
├── configs/                    # Experiment configurations
│   ├── experiment_10M.yaml     # 10M testing configs
│   └── experiment_100M.yaml    # 100M official training configs
│
├── scripts/                    # LRZ Cluster Slurm scripts(B)
│   ├── run_10M.slurm           # Single-node test script for 10M
│   └── run_100M.slurm          # Multi-node training script for 100M
│
├── external/                   # External official evaluation pipeline
│   └── multilingual-evaluation/
│
├── docs/                       # Project documentation
│   └── evaluation_quickstart.md
│
├── notebooks/                  # Visualization & Analysis notebooks(A,C)
│   ├── 01_data_eda.ipynb       # Corpus distribution EDA
│   └── 02_plot_results.ipynb   # Plotting charts (Radar, Loss curves)
│
├── .gitignore                  # Git ignore list
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── main.py                     # Main entry point for training
```

## 🚀 How to Run

### 1. Setup Environment 

Create a virtual environment locally or on the LRZ cluster, then install the required dependencies:

```bash
conda create -n babylm python=3.10 -y
conda activate babylm
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=babylm --display-name "Python (babylm_conda)"
```

download raw dataset
```bash
python src/data_pipeline/download.py    
```

Generate baseline dataset, adjusting tokenizer and total budget you want
```bash
python src/data_pipeline/create_baseline.py
```

Generate stage curriculum dataset, adjusting TOTAL_BUDGET and vocab_configs (fill the tokenizer you want)
```bash
python src/data_pipeline/clean_and_mix.py
```

If you haven't initialized the evaluation module, load the official multilingual
evaluation submodule:

```bash
git submodule add https://github.com/babylm-org/multilingual-evaluation external/multilingual-evaluation
git submodule update --init --recursive
```

### 2. Test with 10M Dataset 

Before official training, use the 10M configuration to test and ensure the pipeline works correctly:

```bash
python main.py --config configs/experiment_10M.yaml
```

### 3. Official 100M Training 

Once the 10M test succeeds, switch to the 100M config for official training. If on the LRZ cluster, submit the job via Slurm:

```bash
python main.py --config configs/experiment_100M.yaml
```

On LRZ Cluster, use:

```bash
sbatch scripts/run_100M.slurm
```

## 📊 Evaluation & Visualization 

Once training checkpoints are generated, run evaluations using the official
multilingual evaluation pipeline. Start with a tiny local smoke test before
running full evaluation on LRZ.

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

For our target languages, the current task group names are:

```text
babybabellm_eng
babybabellm_nld
babybabellm_zho
```

For full LRZ runs, remove `--limit`, use `--device cuda`, and set
`--batch_size auto`. More details and caveats are in
[`docs/evaluation_quickstart.md`](docs/evaluation_quickstart.md).

## 📝 Notes 

- This project is part of the **BabyLM Challenge 2026 (Multilingual Track)**.
- Ensure all datasets are properly downloaded and processed before training.
- The current model is `BertForMaskedLM`. The official 2026 multilingual
  pipeline currently runs through `--model hf`; `hf-mlm` and `backend=mlm` are
  not available in the tested `lm-eval` versions.
