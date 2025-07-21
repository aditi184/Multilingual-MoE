import argparse
import json
import os
import torch
import yaml
from pathlib import Path
from transformers import OlmoForCausalLM

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=4)

def save_olmo_checkpoint(model_path, hf_model_path):
    os.makedirs(model_path, exist_ok=True)

    # Load HF model
    hf_checkpoint = OlmoForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    hf_config = hf_checkpoint.config
    state_dict = hf_checkpoint.state_dict()

    olmo_state_dict = {}

    # Convert transformer layers
    for layer_i in range(hf_config.num_hidden_layers):
        q_proj = state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        k_proj = state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"]
        v_proj = state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]

        # Fuse Q, K, V into a single tensor
        att_proj = torch.cat([q_proj, k_proj, v_proj], dim=0)
        olmo_state_dict[f"transformer.blocks.{layer_i}.att_proj.weight"] = att_proj
        olmo_state_dict[f"transformer.blocks.{layer_i}.attn_out.weight"] = state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]
        olmo_state_dict[f"transformer.blocks.{layer_i}.mlp.fc1.weight"] = state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"]
        olmo_state_dict[f"transformer.blocks.{layer_i}.mlp.fc2.weight"] = state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"]
        olmo_state_dict[f"transformer.blocks.{layer_i}.mlp.fc3.weight"] = state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"]

    # Embedding & output
    olmo_state_dict["transformer.wte.weight"] = state_dict["model.embed_tokens.weight"]
    olmo_state_dict["transformer.ff_out.weight"] = state_dict["lm_head.weight"]

    # Save model.pt
    torch.save(olmo_state_dict, os.path.join(model_path, "model.pt"))

    # Save config.yaml
    olmo_config = {
        "model": {
            "n_layers": hf_config.num_hidden_layers,
            "n_heads": hf_config.num_attention_heads,
            "d_model": hf_config.hidden_size,
            "embedding_size": hf_config.vocab_size,
            "max_sequence_length": hf_config.max_position_embeddings,
            "pad_token_id": hf_config.pad_token_id,
            "eos_token_id": hf_config.eos_token_id,
            "weight_tying": hf_config.tie_word_embeddings,
            "block_type": "dense",
            "n_kv_heads": None,
            "clip_qkv": None,
            "include_bias": False,
            "bias_for_layer_norm": False
        }
    }

    with open(os.path.join(model_path, "config.yaml"), "w") as f:
        yaml.dump(olmo_config, f)

    print(f"Converted OLMo model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", required=True, help="Path to the HF model directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save OLMo checkpoint.")
    args = parser.parse_args()
    save_olmo_checkpoint(args.output_dir, args.hf_model_path)

if __name__ == "__main__":
    main()
