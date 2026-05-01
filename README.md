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
├── notebooks/                  # Visualization & Analysis notebooks(A,C)
│   ├── 01_data_eda.ipynb       # Corpus distribution EDA
│   └── 02_plot_results.ipynb   # Plotting charts (Radar, Loss curves)
│
├── babylm-eval/                # Official evaluation pipeline(C)
├── .gitignore                  # Git ignore list
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── main.py                     # Main entry point for training
```

## 🚀 How to Run

### 1. Setup Environment 

Create a virtual environment locally or on the LRZ cluster, then install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you haven't initialized the evaluation module, load the babylm-eval submodule:

```bash
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

Once training checkpoints are generated, run evaluations using the official pipeline:

```bash
# Example evaluation command inside the babylm-eval folder
cd babylm-eval
python run_eval.py --model_path ../checkpoints/100M_final --tasks default
```

## 📝 Notes 

- This project is part of the **BabyLM Challenge 2026 (Multilingual Track)**.
- Ensure all datasets are properly downloaded and processed before training.

