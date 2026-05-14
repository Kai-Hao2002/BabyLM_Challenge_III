#!/bin/bash
#SBATCH --job-name=babylm-train
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

CONFIG_PATH=$1

cd ~/BabyLM_Challenge_III

source ~/miniconda3/etc/profile.d/conda.sh
conda activate babylm

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_PATH}"
echo "Python: $(which python)"
python --version

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

python main.py --config "${CONFIG_PATH}"
