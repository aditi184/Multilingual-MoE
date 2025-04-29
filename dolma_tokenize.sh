#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/tokenize_multilingual.out


echo Running on $(hostname)
module load miniconda/3
module load cuda/12.2.2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
source ~/.bashrc 
conda activate OLMO


dolma tokens \
--documents /home/mila/k/khandela/scratch/CulturaX_text/test/zh \
--destination /home/mila/k/khandela/scratch/olmoe-data/test/zh \
--tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
--max_size '2_147_483_648' \
--seed 0 \
--tokenizer.eos_token_id 50279 \
--tokenizer.pad_token_id 1 \
--processes 4