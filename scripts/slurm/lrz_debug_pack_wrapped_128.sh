#!/bin/bash
#SBATCH --job-name=babylm-debug-pack128
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --output=/dss/dsshome1/00/go46lic2/BabyLM_Challenge_III/logs/slurm/%x-%j.out
#SBATCH --error=/dss/dsshome1/00/go46lic2/BabyLM_Challenge_III/logs/slurm/%x-%j.err
#SBATCH --chdir=/dss/dsshome1/00/go46lic2/BabyLM_Challenge_III
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

source ~/.bashrc
conda activate babylm

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python --version

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
python -c "import transformers, datasets, trl; print('hf stack ok')"

python main.py --config configs/lrz_debug_pack_wrapped_128.yaml
