#!/bin/bash
#SBATCH --job-name=olmo-5langs
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --output=logs/pretrain_olmo-05-05-2025.out
#SBATCH --partition=short-unkillable

echo Running on $(hostname)

# activate environment
module load miniconda/3
module load cuda/12.2.2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
source ~/.bashrc 
conda activate OLMO

# wandb login  # authenticate via the WANDB_API_KEY environment variable; never commit keys



torchrun --nproc_per_node=4 train.py configs/config_olmo_test_multilingual.yml
