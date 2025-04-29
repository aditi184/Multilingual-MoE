#!/bin/bash
#SBATCH --job-name=olmoe-no-lbl
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --output=logs/pretrain_multilingual-nolbl-23-04-2025.out
#SBATCH --partition=short-unkillable

echo Running on $(hostname)

# activate environment
module load miniconda/3
module load cuda/12.2.2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
source ~/.bashrc 
conda activate OLMO

# wandb login 37116fe56d7716a68ca696610da1b9da40f4690e



torchrun --nproc_per_node=4 train.py configs/config_multilingual_no_lbl.yml
