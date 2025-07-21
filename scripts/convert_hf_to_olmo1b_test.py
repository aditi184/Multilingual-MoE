#run via 
# python scripts/convert_hf_to_olmo1b_test.py --hf_model_path allenai/OLMo-1B-hf --output_dir /home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMo-1B/base-test
"""
Convert a Hugging Face OlmoForCausalLM checkpoint into the original OLMo dense model format.
Inspired by the OLMoE conversion script’s structure and formatting.
"""
import os
import json
import yaml
import torch
import shutil
from pathlib import Path
from transformers import OlmoForCausalLM
import argparse

def write_json(obj, path: str):
    """Write a JSON-serializable object to disk with indentation."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def save_olmo_checkpoint(output_dir: str, hf_model_path: str):
    """
    Load HF OlmoForCausalLM and reserialize into original OLMo checkpoint format.
    Creates `model.pt` under output_dir and writes `config.yaml`.
    """
    # Prepare directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(output_dir) / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Load HF model
    hf_model = OlmoForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    cfg = hf_model.config
    print("Loaded HF config:", cfg)

    # Compute rotary inv_freq
    dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = dim // n_heads
    base = cfg.rope_theta or 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Prepare new state dict
    state = {}
    hf_state = hf_model.state_dict()

    # Token embedding & lm head
    state["transformer.wte.weight"] = hf_state["model.embed_tokens.weight"].cpu().clone()
    state["transformer.ff_out.weight"] = hf_state["lm_head.weight"].cpu().clone()

    # Loop over each dense transformer block
    for i in range(cfg.num_hidden_layers):
        # QKV fusion
        q = hf_state[f"model.layers.{i}.self_attn.q_proj.weight"].cpu().clone()
        k = hf_state[f"model.layers.{i}.self_attn.k_proj.weight"].cpu().clone()
        v = hf_state[f"model.layers.{i}.self_attn.v_proj.weight"].cpu().clone()
        state[f"transformer.blocks.{i}.att_proj.weight"] = torch.cat([q, k, v], dim=0)

        # Attention out
        state[f"transformer.blocks.{i}.attn_out.weight"] = hf_state[f"model.layers.{i}.self_attn.o_proj.weight"].cpu().clone()

        # FF projections (up + gate)
        up = hf_state[f"model.layers.{i}.mlp.up_proj.weight"].cpu().clone()
        gate = hf_state[f"model.layers.{i}.mlp.gate_proj.weight"].cpu().clone()
        state[f"transformer.blocks.{i}.ff_proj.weight"] = torch.cat([up, gate], dim=0)

        # FF output
        state[f"transformer.blocks.{i}.ff_out.weight"] = hf_state[f"model.layers.{i}.mlp.down_proj.weight"].cpu().clone()

        # Rotary embedding freq
        # state[f"transformer.blocks.{i}.rotary_emb.inv_freq"] = inv_freq.clone()

        # LayerNorms
        # state[f"transformer.blocks.{i}.attn_norm.weight"] = hf_state[f"model.layers.{i}.input_layernorm.weight"].cpu().clone()
        # state[f"transformer.blocks.{i}.attn_norm.bias"] = hf_state[f"model.layers.{i}.input_layernorm.bias"].cpu().clone()
        # state[f"transformer.blocks.{i}.ff_norm.weight"] = hf_state[f"model.layers.{i}.post_attention_layernorm.weight"].cpu().clone()
        # state[f"transformer.blocks.{i}.ff_norm.bias"] = hf_state[f"model.layers.{i}.post_attention_layernorm.bias"].cpu().clone()

    # Final LayerNorm
    # state["transformer.ln_f.weight"] = hf_state["model.norm.weight"].cpu().clone()
    # state["transformer.ln_f.bias"] = hf_state["model.norm.bias"].cpu().clone()

    # Save consolidated checkpoint
    torch.save(state, tmp_dir / "model.pt")
    print(f"Saved OLMo checkpoint to {tmp_dir / 'model.pt'}")

    # Write config.yaml
    olmo_cfg = {
        "model": {
            "n_layers": cfg.num_hidden_layers,
            "n_heads": cfg.num_attention_heads,
            "d_model": cfg.hidden_size,
            "mlp_ratio": cfg.intermediate_size * 2 / cfg.hidden_size,
            "max_sequence_length": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "embedding_size": cfg.vocab_size,
            "pad_token_id": cfg.pad_token_id,
            "eos_token_id": cfg.eos_token_id,
            "multi_query_attention": False,
            "n_kv_heads": cfg.num_key_value_heads,
            "weight_tying": cfg.tie_word_embeddings,
            "clip_qkv": cfg.clip_qkv,
        }
    }
    with open(Path(output_dir) / "config.yaml", "w") as f:
        yaml.dump(olmo_cfg, f)
    print(f"Wrote config.yaml to {output_dir}/config.yaml")

    # Move tmp files into final structure
    final_pt = Path(output_dir) / "model.pt"
    shutil.move(str(tmp_dir / "model.pt"), final_pt)
    shutil.rmtree(tmp_dir)
    print(f"Conversion complete: {final_pt}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face OlmoForCausalLM → original OLMo dense checkpoint format"
    )
    parser.add_argument(
        "--hf_model_path", required=True,
        help="Hugging Face OlmoForCausalLM checkpoint directory"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write original OLMo checkpoint"
    )
    args = parser.parse_args()
    save_olmo_checkpoint(args.output_dir, args.hf_model_path)


if __name__ == "__main__":
    main()
