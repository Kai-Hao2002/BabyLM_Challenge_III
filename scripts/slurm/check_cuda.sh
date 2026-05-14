#!/bin/bash
#SBATCH --job-name=check-cuda
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=00:05:00

cd ~/BabyLM_Challenge_III
source ~/miniconda3/etc/profile.d/conda.sh
conda activate babylm

hostname
nvidia-smi
which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
