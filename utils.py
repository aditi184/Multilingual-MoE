from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as compute_entropy
import json
import argparse
import scipy.stats
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import OlmoeForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import time
import pickle as pkl
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
import argparse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import math


def analyze_expert_tokens(layer, expert_id, eid2token_layerwise, tokenizer, top_k=20):
    token_counter = eid2token_layerwise[layer][expert_id]
    most_common = token_counter.most_common(top_k)
    
    print(f"\nTop {top_k} tokens routed to Expert {expert_id} at Layer {layer}:\n")
    for token_id, count in most_common:
        token_str = tokenizer.decode([token_id])
        print(f"{token_str!r:20s} | Token ID: {token_id:<5d} | Count: {count}")

def analyze_all_experts(layer, eid2token_layerwise, tokenizer, top_k=10):
    print(f"\n--- Expert-wise Token Routing at Layer {layer} ---\n")
    for expert_id in sorted(eid2token_layerwise[layer].keys()):
        print(f"\nExpert {expert_id}:")
        analyze_expert_tokens(layer, expert_id, eid2token_layerwise, tokenizer, top_k)

def compute_distribution_metrics(expert_counts, layer):
    expert_ids = sorted(expert_counts.keys())
    expert_values = np.array([expert_counts[eid] for eid in expert_ids])
    expert_probs = expert_values / expert_values.sum()

    uniform = np.ones_like(expert_probs) / len(expert_probs)
    entropy = scipy.stats.entropy(expert_probs)
    kl_div = scipy.stats.entropy(expert_probs, uniform)

    return expert_probs, expert_ids, entropy, kl_div

from itertools import combinations

def analyze_expert_coactivation(text, layers_to_analyze, model, tokenizer):
    _, _, _, exp_ids = run_analysis(text, layers_to_analyze, model, tokenizer)

    coactivation_counts = {layer: Counter() for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        for idx in range(exp_ids.shape[0]):  # over tokens
            experts = exp_ids[idx, :, layer]
            # Consider all unordered pairs of activated experts
            for pair in combinations(sorted(experts), 2):
                coactivation_counts[layer][pair] += 1

    return coactivation_counts

def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=1)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-2)) / len(gate_logits)
    return overall_loss * num_experts

def load_analysis_data(tokenizer, text):
    np.random.seed(2024)
    tokens = tokenizer(text,max_length=4096)["input_ids"]
    return tokens

def load_model(model_name="allenai/OLMoE-1B-7B-0924-Instruct",device=0):
    model = OlmoeForCausalLM.from_pretrained(model_name).to(f'cuda:{device}')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
    return model, tokenizer

def print_expert_percentage(exp_counts):
    total = sum(exp_counts.values())
    for eid, ecount in exp_counts.most_common():
        print(f"Expert {eid}: {ecount/total*100:.2f}")

def run_analysis(text, layers_to_analyze, model, tokenizer):
    """Analyzes expert token routing in a mixture-of-experts (MoE) model."""
    layer_counters = defaultdict(Counter)
    crosslayer_counters = defaultdict(Counter)
    eid2token_layerwise = {layer: defaultdict(Counter) for layer in layers_to_analyze}

    total_token_count = 0
    aux_losses = []

    # Load entire tokenized text at once
    input_ids = load_analysis_data(tokenizer, text)

    # Convert to tensor and process
    input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
    out = model(input_ids=input_ids, output_router_logits=True)

    # print(out["router_logits"])
    # Compute auxiliary loss for load balancing
    aux_loss = load_balancing_loss_func(
        out["router_logits"],
        model.num_experts,
        model.num_experts_per_tok,
    )
    aux_losses.append(aux_loss.cpu().item())

    input_ids = input_ids[0].detach().cpu().numpy().tolist()
    total_token_count += len(input_ids)

    # Extract router logits and compute expert assignments
    router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]] #Raw scores that decide expert routing
    exp_ids = np.stack([np.argsort(-logits, -1)[:, :8].tolist() for logits in router_logits], -1) #Top-8 experts (with highest logits) chosen per token per layer. Shape: [tokens, top_k, layers].
    # print(f"Input IDs: {input_ids}")
    # print(f"Router logits: {len(router_logits)}")
    # print(f"Router logits shape: {router_logits[0].shape}")
    # print(f"Expert IDs: {exp_ids.shape}")
    # print(f"Router logits: {router_logits}")

    # Track expert-token associations per layer
    for layer in layers_to_analyze:
        exp_ids_layer = exp_ids[:, :, layer]
        for idx, token in enumerate(input_ids):
            experts = exp_ids_layer[idx, :]
            for e in experts:
                eid2token_layerwise[layer][e][token] += 1

    # Count expert selections per layer
    for layer in range(exp_ids.shape[2]):
        exp_counts = Counter(exp_ids[:, :, layer].flatten())
        layer_counters[layer].update(exp_counts)

    # Track cross-layer expert correlations
    for layer_i in range(exp_ids.shape[2] - 1):
        for layer_j in range(exp_ids.shape[2]):
            exps_counts = Counter(zip(exp_ids[:, :, layer_i].flatten(), exp_ids[:, :, layer_j].flatten()))
            crosslayer_counters[(layer_i, layer_j)].update(exps_counts)

    print(f"Average auxiliary loss: {np.mean(aux_losses)}")

    return layer_counters, crosslayer_counters, eid2token_layerwise, exp_ids






def analyze_expert_usage_by_position(text, layers_to_analyze, model, tokenizer):
    layer_counters, _, eid2token_layerwise, exp_ids = run_analysis(text, layers_to_analyze, model, tokenizer)

    num_tokens = exp_ids.shape[0]
    split1 = num_tokens // 3
    split2 = 2 * num_tokens // 3

    position_based_counts = {layer: {"early": Counter(), "middle": Counter(), "late": Counter()} for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        for idx in range(num_tokens):
            if idx < split1:
                pos = "early"
            elif idx < split2:
                pos = "middle"
            else:
                pos = "late"

            experts = exp_ids[idx, :, layer]
            position_based_counts[layer][pos].update(experts)

    return position_based_counts


def plot_all_langs_expert_token_counts(all_lang_expert_counts,layers_to_analyze):
    for lang, layer_expert_counts in all_lang_expert_counts.items():
        num_layers = len(layers_to_analyze)
        cols = 1
        rows = math.ceil(num_layers / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 4 * rows), squeeze=False)
        fig.suptitle(f"Expert Token Counts for {lang.upper()}", fontsize=16)

        for idx, (layer, expert_counts) in enumerate(sorted(layer_expert_counts.items())):
            ax = axes[idx // cols][idx % cols]
            expert_ids = sorted(expert_counts.keys())
            token_counts = np.array([expert_counts[eid] for eid in expert_ids])
            total = token_counts.sum()
            token_percents = (token_counts / total) * 100 if total > 0 else np.zeros_like(token_counts)

            sns.barplot(x=expert_ids, y=token_percents, palette="viridis", ax=ax)
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Token %")
            ax.set_xticks(range(len(expert_ids)))
            ax.set_ylim(0, 15)
        # Hide any unused subplots
        for i in range(num_layers, rows * cols):
            fig.delaxes(axes[i // cols][i % cols])

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to leave space for title
        # plt.show()
        plt.savefig(f"analysis/expert_tokens_percentages/expert_token_counts_{lang}.png")
        wandb.init(project="Analysis OLMoE", name=f"Expert_Tokens_Distribution_Global_MMLU_{lang}")
        wandb.log({f"Expert_Token_Counts_{lang}": wandb.Image(f"analysis/Expert_Tokens_Distribution_{lang}.png")})
        # wandb.log({f"Expert_Token_Counts_{lang}": wandb.Image(fig)})



def collect_language_expert_stats(text, layers_to_analyze, model, tokenizer):
    _, _, eid2token_layerwise, _ = run_analysis(text, layers_to_analyze, model, tokenizer)

    layer_expert_counts = {layer: Counter() for layer in layers_to_analyze}
    token_assignments = {layer: defaultdict(Counter) for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        for expert_id, token_counts in eid2token_layerwise[layer].items():
            total_tokens = sum(token_counts.values())
            layer_expert_counts[layer][expert_id] += total_tokens
            for token_id, count in token_counts.items():
                token_assignments[layer][expert_id][token_id] += count

    return layer_expert_counts, token_assignments

def symmetrized_kl(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Small constant to avoid division by 0
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    
    kl_pq = scipy.stats.entropy(p, q)
    kl_qp = scipy.stats.entropy(q, p)
    
    return 0.5 * (kl_pq + kl_qp)

def compute_expert_probs(counts_dict, layer, global_expert_ids):
    probs = []
    for eid in global_expert_ids:
        count = counts_dict[layer].get(eid, 0)
        probs.append(count)
    probs = np.array(probs, dtype=np.float64)
    probs /= probs.sum() if probs.sum() > 0 else 1.0
    return probs

def compute_pairwise_kl(all_lang_expert_counts, layer, languages):
    # Step 1: Get union of expert IDs across all languages for this layer
    expert_id_set = set()
    for lang in languages:
        expert_id_set.update(all_lang_expert_counts[lang][layer].keys())
    global_expert_ids = sorted(expert_id_set)

    # Step 2: Get aligned expert probabilities for each language
    lang_probs = {}
    for lang in languages:
        probs = compute_expert_probs(all_lang_expert_counts[lang], layer, global_expert_ids)
        lang_probs[lang] = probs

    # Step 3: Compute symmetric KL matrix
    kl_matrix = np.zeros((len(languages), len(languages)))
    for i, lang_i in enumerate(languages):
        for j, lang_j in enumerate(languages):
            p = lang_probs[lang_i]
            q = lang_probs[lang_j]
            kl_matrix[i, j] = symmetrized_kl(p, q)

    return kl_matrix

def plot_kl_heatmaps(all_lang_expert_counts, layers_to_analyze, languages):
    max_kl = 0
    kl_matrices = {}

    # Compute all KL matrices and find max value for shared scale
    for layer in layers_to_analyze:
        kl_matrix = compute_pairwise_kl(all_lang_expert_counts, layer, languages)
        kl_matrices[layer] = kl_matrix
        max_kl = max(max_kl, kl_matrix.max())

    # Plot heatmaps


# Create a grid of subplots
    n_layers = len(layers_to_analyze)
    n_cols = 3  # Adjust based on how many columns you want
    n_rows = math.ceil(n_layers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Flatten axes for easy iteration, even if 1D
    axes = axes.flatten()

    for idx, layer in enumerate(layers_to_analyze):
        ax = axes[idx]
        sns.heatmap(
            kl_matrices[layer],
            xticklabels=languages,
            yticklabels=languages,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=max_kl,
            ax=ax
        )
        ax.set_title(f"Symmetric KL Divergence (Layer {layer})")

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("analysis/pairwise_kl_divergence_layers.png")
    wandb.init(project="Analysis OLMoE", name=f"Pairwise_KL_Divergence_Layers")
    wandb.log({"Pairwise_KL_Divergence_Layers": wandb.Image("analysis/pairwise_kl_divergence_layers.png")})
    plt.show()

def plot_coactivation_heatmap(coactivation_counts, lang="en", layer=0):
    """Plot *normalized* heatmap of expert co-activations for a given language and layer."""
    counts = coactivation_counts[lang][layer]
    
    num_experts = max(max(pair) for pair in counts) + 1  # infer number of experts
    matrix = np.zeros((num_experts, num_experts))

    total_pairs = sum(counts.values())

    for (e1, e2), count in counts.items():
        normalized_count = count / total_pairs if total_pairs > 0 else 0
        matrix[e1, e2] = normalized_count
        matrix[e2, e1] = normalized_count  # symmetric

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="viridis", square=True)
    plt.title(f"Normalized Expert Co-Activation Heatmap (Layer {layer}, Language: {lang})")
    plt.xlabel("Expert ID")
    plt.ylabel("Expert ID")
    plt.tight_layout()
    plt.savefig(f"analysis/cooactivation_heatmap_{lang}_layer{layer}.png")

def plot_position_based_expert_usage(position_based_counts, lang="en", layer=0):
    """Plot *normalized* expert usage for early/middle/late token positions for a given language and layer."""
    pos_data = position_based_counts[lang][layer]
    df = []

    for pos in ["early", "middle", "late"]:
        total = sum(pos_data[pos].values())
        for expert_id, count in pos_data[pos].items():
            normalized_count = count / total if total > 0 else 0
            df.append({"Position": pos, "Expert ID": expert_id, "Normalized Count": normalized_count})

    import pandas as pd
    df = pd.DataFrame(df)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Expert ID", y="Normalized Count", hue="Position")
    plt.title(f"Normalized Expert Usage by Token Position (Layer {layer}, Language: {lang})")
    plt.xlabel("Expert ID")
    plt.ylabel("Fraction of Tokens")
    plt.legend(title="Sequence Part")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"analysis/position_based_expert_usage_lang{lang}_layer{layer}.png")




def plot_expert_distribution_across_langs(all_lang_expert_counts, layer):
    languages = all_lang_expert_counts.keys()
    for lang in languages:
        expert_probs, expert_ids, entropy, kl = compute_distribution_metrics(all_lang_expert_counts[lang][layer], layer)
        
        plt.figure(figsize=(16, 4))
        sns.barplot(x=expert_ids, y=expert_probs, palette="viridis")
        plt.title(f"{lang.upper()} - Layer {layer} | Entropy={entropy:.2f} | KL={kl:.2f}")
        plt.xlabel("Expert ID")
        plt.ylabel("Probability")
        plt.ylim(0, max(expert_probs) * 1.2)
        plt.tight_layout()
        plt.show()

def analyze_top_tokens_per_expert(token_assignments, layer, tokenizer, lang, top_k=10):
    print(f"\n[Language: {lang}] Expert-wise Token Summary at Layer {layer}")
    for expert_id, token_counter in token_assignments[lang][layer].items():
        top_tokens = token_counter.most_common(top_k)
        decoded = [tokenizer.decode([tok]) for tok, _ in top_tokens]
        print(f"\nExpert {expert_id} → Tokens: {decoded}")

def get_shared_and_unique_experts_decoded(all_lang_expert_counts, all_lang_token_assignments, layers_to_analyze, tokenizer, cutoff=0.01, n_examples=5):
    print("\n--- Shared and Unique Experts by Layer (with cutoff {:.2f}) ---\n".format(cutoff))
    languages = all_lang_expert_counts.keys()
    
    for layer in layers_to_analyze:
        lang_to_active_experts = {}

        for lang, layer_counts in all_lang_expert_counts.items():
            expert_counts = layer_counts[layer]
            total = sum(expert_counts.values())
            if total == 0:
                lang_to_active_experts[lang] = set()
                continue
            expert_probs = {eid: count / total for eid, count in expert_counts.items()}
            active_experts = {eid for eid, prob in expert_probs.items() if prob >= cutoff}
            lang_to_active_experts[lang] = active_experts

        all_sets = list(lang_to_active_experts.values())
        all_experts = set.union(*all_sets) if all_sets else set()
        
        shared = set.intersection(*all_sets) if all_sets else set()
        unique = {lang: experts - set.union(*(lang_to_active_experts[l] for l in languages if l != lang)) 
                  for lang, experts in lang_to_active_experts.items()}

        print(f"Layer {layer}:")
        print(f"  Total Experts Used: {len(all_experts)}")
        print(f"  Shared Experts (across all): {sorted(shared)}")

        # ========== Show shared experts' tokens ==========
        if shared:
            print(f"    Example Tokens from Shared Experts:")
            for eid in sorted(shared):
                tokens = []
                for lang in languages:
                    token_counter = all_lang_token_assignments[lang][layer][eid]
                    top_tokens = [token for token, _ in token_counter.most_common(n_examples)]
                    decoded = tokenizer.decode(top_tokens, skip_special_tokens=True)
                    tokens.append(f"{lang.upper()}: {decoded}")
                print(f"      Expert {eid}:")
                for tok in tokens:
                    print(f"        {tok}")
        
        # ========== Show unique experts' tokens ==========
        for lang in lang_to_active_experts:
            if unique[lang]:
                print(f"    {lang.upper()} Unique Experts and Example Tokens:")
                for eid in sorted(unique[lang]):
                    token_counter = all_lang_token_assignments[lang][layer][eid]
                    top_tokens = [token for token, _ in token_counter.most_common(n_examples)]
                    decoded = tokenizer.decode(top_tokens, skip_special_tokens=True)
                    print(f"      Expert {eid}: {decoded}")
        
        print()



def get_shared_and_unique_experts(all_lang_expert_counts, layers_to_analyze, cutoff=0.01):
    print("\n--- Shared and Unique Experts by Layer (with cutoff {:.2f}) ---\n".format(cutoff))
    languages = all_lang_expert_counts.keys()
    for layer in layers_to_analyze:
        lang_to_active_experts = {}

        for lang, layer_counts in all_lang_expert_counts.items():
            expert_counts = layer_counts[layer]
            total = sum(expert_counts.values())
            if total == 0:
                lang_to_active_experts[lang] = set()
                continue
            expert_probs = {eid: count / total for eid, count in expert_counts.items()}
            active_experts = {eid for eid, prob in expert_probs.items() if prob >= cutoff}
            lang_to_active_experts[lang] = active_experts

        all_sets = list(lang_to_active_experts.values())
        all_experts = set.union(*all_sets) if all_sets else set()
        
        shared = set.intersection(*all_sets) if all_sets else set()
        unique = {lang: experts - set.union(*(lang_to_active_experts[l] for l in languages if l != lang)) 
                  for lang, experts in lang_to_active_experts.items()}

        print(f"Layer {layer}:")
        print(f"  Total Experts Used: {len(all_experts)}")
        print(f"  Shared Experts (across all): {sorted(shared)}")
        for lang in lang_to_active_experts:
            print(f"    {lang.upper()} Unique Experts: {sorted(unique[lang])}")
        print()

def plot_entropy_kl_summary(all_lang_expert_counts, layers_to_analyze):
    languages = all_lang_expert_counts.keys()
    entropies = {lang: [] for lang in languages}
    kls = {lang: [] for lang in languages}

    for layer in layers_to_analyze:
        for lang in languages:
            expert_counts = all_lang_expert_counts[lang][layer]
            expert_probs, _, entropy, kl = compute_distribution_metrics(expert_counts, layer)
            entropies[lang].append(entropy)
            kls[lang].append(kl)

    return entropies, kls

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import entropy as scipy_entropy

# --- Helper Functions ---

def compute_active_experts(expert_counts, threshold_percent=5.0):
    """Find active experts per layer based on token assignment percentage."""
    active_experts = {}
    for layer, counts in expert_counts.items():
        total_tokens = sum(counts.values())
        active = set()
        for expert_id, count in counts.items():
            if total_tokens > 0 and (count / total_tokens) * 100 >= threshold_percent:
                active.add(expert_id)
        active_experts[layer] = active
    return active_experts

def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

# --- 1. Cross-Language Expert Overlap Analysis ---

def cross_language_overlap(all_lang_expert_counts, layers_to_analyze, threshold_percent=5.0):
    """Compute average Jaccard similarity of active experts across language pairs."""
    lang_active_experts = {
        lang: compute_active_experts(layer_counts, threshold_percent)
        for lang, layer_counts in all_lang_expert_counts.items()
    }

    layerwise_overlap = {layer: [] for layer in layers_to_analyze}
    langs = list(all_lang_expert_counts.keys())

    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            lang1, lang2 = langs[i], langs[j]
            for layer in layers_to_analyze:
                jac = jaccard_similarity(
                    lang_active_experts[lang1][layer],
                    lang_active_experts[lang2][layer]
                )
                layerwise_overlap[layer].append(jac)

    # average overlap across language pairs for each layer
    avg_overlap_per_layer = {layer: np.mean(overlaps) for layer, overlaps in layerwise_overlap.items()}
    return avg_overlap_per_layer

# --- 2. Language-Expert Affinity Heatmap ---

def compute_lang_expert_matrix(all_lang_expert_counts, layers_to_analyze, num_experts):
    """Builds a Language x Expert matrix where each cell = % of tokens assigned to expert."""
    lang_expert_matrices = {}
    
    for layer in layers_to_analyze:
        lang_expert_matrix = np.zeros((len(all_lang_expert_counts), num_experts))
        langs = list(all_lang_expert_counts.keys())
        for i, lang in enumerate(langs):
            counts = all_lang_expert_counts[lang][layer]
            total_tokens = sum(counts.values())
            for expert_id in range(num_experts):
                if expert_id in counts:
                    lang_expert_matrix[i, expert_id] = (counts[expert_id] / total_tokens) * 100 if total_tokens > 0 else 0
        lang_expert_matrices[layer] = (langs, lang_expert_matrix)
    return lang_expert_matrices

def plot_lang_expert_heatmap(lang_expert_matrices, title_prefix=""):
    """Plots the Language x Expert matrix as a heatmap."""
    for layer, (langs, matrix) in lang_expert_matrices.items():
        plt.figure(figsize=(14, max(6, len(langs)//2)))
        sns.heatmap(matrix, cmap="viridis", xticklabels=True, yticklabels=langs, annot=False)
        plt.title(f"{title_prefix} Language-Expert Affinity (Layer {layer})")
        plt.xlabel("Expert ID")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.show()

# --- 3. Per-Language Expert Entropy ---

def compute_language_entropy(all_lang_expert_counts, layers_to_analyze, num_experts):
    """Computes entropy of token distribution over experts for each language."""
    lang_entropy_per_layer = {lang: {} for lang in all_lang_expert_counts.keys()}

    for lang, layer_counts in all_lang_expert_counts.items():
        for layer in layers_to_analyze:
            counts = layer_counts[layer]
            total_tokens = sum(counts.values())
            probs = np.zeros(num_experts)
            for expert_id in range(num_experts):
                if expert_id in counts:
                    probs[expert_id] = counts[expert_id] / total_tokens if total_tokens > 0 else 0
            ent = scipy_entropy(probs, base=2) if total_tokens > 0 else 0
            lang_entropy_per_layer[lang][layer] = ent

    return lang_entropy_per_layer

