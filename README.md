This repo is adapted from [OLMo branch](https://github.com/allenai/OLMo/tree/Muennighoff/MoE).

**Pretraining**

Clone this [OLMo branch](https://github.com/allenai/OLMo/tree/Muennighoff/MoE) & create an environment with its dependencies via cd OLMo; pip install -e .. If you want to use new features in OLMo clone from the main branch instead.
Run pip install git+https://github.com/Muennighoff/megablocks.git@olmoe
Setup a config file. configs/OLMoE-1B-7B-0924.yml was used for the pretraining of OLMoE-1B-7B-0924. You can find configs from various ablations in configs/ablations.
Tokenize it via the command below and adapt the paths in your training config to point to it.

```bash
dolma tokens \
--documents ${PATH_TO_DOWNLOADED_DATA} \
--destination ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA} \
--tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
--max_size '2_147_483_648' \
--seed 0 \
--tokenizer.eos_token_id 50279 \
--tokenizer.pad_token_id 1 \
--processes ${NUMBER_OF_CPU_CORES_TO_USE}
```

Run via
```
torchrun --nproc_per_node=4 train.py configs/config_mulitlingual.yml
```

Run Analysis using analysis.ipynb
