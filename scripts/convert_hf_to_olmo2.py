# Run script using python convert_hf_to_olmo2.py --hf_model_path allenai/OLMoE-1B-7B-0924-Instruct --output_dir /home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base-0924

import argparse
import json
import os
import torch
import yaml
from pathlib import Path
from transformers import OlmoeForCausalLM, OlmoeConfig
import ipdb
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=4)

def save_olmo_checkpoint(model_path, hf_model_path):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)
    
    # Load HF model
    hf_checkpoint = OlmoeForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    hf_config = hf_checkpoint.config
    print(hf_config)
    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    dim = hf_config.hidden_size
    dims_per_head = dim // n_heads
    num_key_value_heads = hf_config.num_key_value_heads
    num_experts = hf_config.intermediate_size // (dim // num_key_value_heads)
    print(num_experts)  # Estimate number of experts
    base = hf_config.rope_theta

    state_dict = hf_checkpoint.state_dict()
    
    olmoe_state_dict = {}
    
    # Convert transformer layers
    for layer_i in range(hf_config.num_hidden_layers):
        q_proj_weight = state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        k_proj_weight = state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"]
        v_proj_weight = state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"]
        
        # Fuse Q, K, V into a single tensor (OLMoE format)
        att_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)

        # Map back to OLMoE format
        olmoe_state_dict[f"transformer.blocks.{layer_i}.att_proj.weight"] = att_proj_weight
        olmoe_state_dict[f"transformer.blocks.{layer_i}.attn_out.weight"] = state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"]
        olmoe_state_dict[f"transformer.blocks.{layer_i}.q_norm.weight"] = state_dict[f"model.layers.{layer_i}.self_attn.q_norm.weight"]
        olmoe_state_dict[f"transformer.blocks.{layer_i}.k_norm.weight"] = state_dict[f"model.layers.{layer_i}.self_attn.k_norm.weight"]
        olmoe_state_dict[f"transformer.blocks.{layer_i}.ffn.router.layer.weight"] = state_dict[f"model.layers.{layer_i}.mlp.gate.weight"]
        olmoe_state_dict[f"transformer.blocks.{layer_i}.attn_norm.weight"] = state_dict[f"model.layers.{layer_i}.input_layernorm.weight"]
        olmoe_state_dict[f"transformer.blocks.{layer_i}.ff_norm.weight"] = state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"]

        
        num_experts = state_dict[f"model.layers.{layer_i}.mlp.gate.weight"].shape[0]
        # dim_per_expert = state_dict[f"model.layers.{layer_i}.mlp.experts.0.gate_proj.weight"].shape[1]
        print(num_experts)
        # Reconstruct MoE expert weights
        w1_list = [state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.gate_proj.weight"] for expert_i in range(num_experts)]
        v1_list = [state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.up_proj.weight"] for expert_i in range(num_experts)]
        w2_list = [state_dict[f"model.layers.{layer_i}.mlp.experts.{expert_i}.down_proj.weight"].T for expert_i in range(num_experts)]

        olmoe_state_dict[f"transformer.blocks.{layer_i}.ffn.experts.mlp.w1"] = torch.cat(w1_list, dim=0)
        olmoe_state_dict[f"transformer.blocks.{layer_i}.ffn.experts.mlp.v1"] = torch.cat(v1_list, dim=0)
        olmoe_state_dict[f"transformer.blocks.{layer_i}.ffn.experts.mlp.w2"] = torch.cat(w2_list, dim=0)

        


    # ipdb.set_trace()
    dims_per_head = dim // n_heads
    print("here")
    print(n_heads)
    print(dim)
    print(dims_per_head)
    base = 10000.0
    # olmoe_state_dict["transformer.inv_freq"] = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    olmoe_state_dict["transformer.wte.weight"] = state_dict["model.embed_tokens.weight"]
    olmoe_state_dict["transformer.ff_out.weight"] = state_dict["lm_head.weight"]
    olmoe_state_dict["transformer.ln_f.weight"] = state_dict["model.norm.weight"]
    
    # Save model.pt
    torch.save(olmoe_state_dict, os.path.join(model_path, "model.pt"))
    
    # Save config.yaml
    olmoe_config = {
        "model": {
            "n_layers": hf_config.num_hidden_layers,
            "n_heads": hf_config.num_attention_heads,
            "d_model": hf_config.hidden_size,
            "embedding_size": hf_config.vocab_size,
            "max_sequence_length": hf_config.max_position_embeddings,
            "pad_token_id": hf_config.pad_token_id,
            "eos_token_id": hf_config.eos_token_id,
            "weight_tying": hf_config.tie_word_embeddings,
            "block_type": "moe",
            "n_kv_heads" : None,
            "clip_qkv" : None,
            "include_bias" : False,
            "bias_for_layer_norm" : False       
             }
    }
    print(olmoe_config)
    with open(os.path.join(model_path, "config.yaml"), "w") as f:
        yaml.dump(olmoe_config, f)
    
    print(f"Converted model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", required=True, help="Path to the HF model directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save OLMo checkpoint.")
    args = parser.parse_args()
    save_olmo_checkpoint(args.output_dir, args.hf_model_path)

if __name__ == "__main__":
    main()
