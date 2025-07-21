#!/bin/bash

# List of languages
languages=("ja")  # Add more language codes as needed

# Base paths
base_input="/home/mila/k/khandela/scratch/CulturaX_text/mutilang_lrl"
base_output="/home/mila/k/khandela/scratch/olmoe-data/test"

# Loop over each language
for lang in "${languages[@]}"; do
  echo "Processing language: $lang"
  
  dolma tokens \
    --documents "${base_input}/${lang}_test.jsonl" \
    --destination "${base_output}/${lang}" \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 4
done
