# Run script using python convert_hf_to_olmo.py --hf_model_path allenai/OLMoE-1B-7B-0924-Instruct --output_path /home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base --tokenizer_path allenai/OLMoE-1B-7B-0924

import os
import torch
import yaml
import argparse
from pathlib import Path
from transformers import OlmoeForCausalLM, AutoTokenizer

def convert_hf_to_olmo(hf_model_path, output_path, tokenizer_path=None):
    os.makedirs(output_path, exist_ok=True)
    print(f"Loading Hugging Face model from {hf_model_path}")
    model = OlmoeForCausalLM.from_pretrained(hf_model_path)
    state_dict = model.state_dict()
    
    olmo_state_dict = {}
    
    print("Reformatting weights to OLMo pretraining format...")
    for key, value in state_dict.items():
        new_key = key.replace("model.layers", "transformer.blocks").replace(".self_attn.", ".att_proj.")
        new_key = new_key.replace("q_proj", "q_proj").replace("k_proj", "k_proj").replace("v_proj", "v_proj")
        new_key = new_key.replace("o_proj", "attn_out").replace("gate_proj", "ffn.router.layer")
        new_key = new_key.replace("up_proj", "ffn.experts.mlp.v1").replace("down_proj", "ffn.experts.mlp.w2")
        new_key = new_key.replace("input_layernorm", "attn_norm").replace("post_attention_layernorm", "ff_norm")
        
        olmo_state_dict[new_key] = value
    
    torch.save(olmo_state_dict, os.path.join(output_path, "model.pt"))
    print(f"Converted model saved to {output_path}/model.pt")
    
    config = {
        "model": {
            "n_layers": model.config.num_hidden_layers,
            "n_heads": model.config.num_attention_heads,
            "d_model": model.config.hidden_size,
            "max_sequence_length": model.config.max_position_embeddings,
            "embedding_size": model.config.vocab_size,
            "pad_token_id": model.config.pad_token_id,
            "eos_token_id": model.config.eos_token_id,
            "weight_tying": model.config.tie_word_embeddings
        }
    }
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    print(f"Config file saved to {output_path}/config.yaml")
    
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        tokenizer.save_pretrained(output_path)
        print(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", required=True, help="Path to the Hugging Face model.")
    parser.add_argument("--output_path", required=True, help="Directory to save the converted OLMo model.")
    parser.add_argument("--tokenizer_path", default=None, help="Optional: Path to Hugging Face tokenizer.")
    args = parser.parse_args()
    
    convert_hf_to_olmo(args.hf_model_path, args.output_path, args.tokenizer_path)
