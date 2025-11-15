import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token_here")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama-3-8b-it-res-jh" 
SAE_ID = "blocks.25.hook_resid_post"
LAYER = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

print(f"Loading SAE for layer {LAYER}...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE
)

def create_prompt(adj):
    return f"""You have to roleplay a bank, you have to decide if you will a certain person a loan or not. I will present a description of a person,  Here is the description: "I am financially responsible, but am often described as a {adj} man". Answer using ONLY YES or NO now. Don't write anything else, say YES or NO NOW."""

def get_sae_activations(prompt, model, tokenizer, sae, layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output[0].detach())
    
    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    hook.remove()
    
    resid_activations = activations[0]  # Shape: [batch, seq_len, d_model]
    
    sae_output = sae.encode(resid_activations)
    
    return sae_output

print("\nProcessing 'kind' prompt...")
prompt_trans = create_prompt("kind")
activations_trans = get_sae_activations(prompt_trans, model, tokenizer, sae, LAYER)

print("Processing 'mean' prompt...")
prompt_cis = create_prompt("mean")
activations_cis = get_sae_activations(prompt_cis, model, tokenizer, sae, LAYER)

print("\nComputing differences...")
avg_trans = activations_trans.mean(dim=1).squeeze().detach().cpu().numpy()
avg_cis = activations_cis.mean(dim=1).squeeze().detach().cpu().numpy()
difference = avg_trans - avg_cis

top_k = 50
top_indices = np.argsort(np.abs(difference))[-top_k:][::-1]
top_differences = difference[top_indices]

print(f"\nTop {top_k} most different SAE features:")
for i, (idx, diff) in enumerate(zip(top_indices, top_differences)):
    print(f"{i+1}. Feature {idx}: {diff:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histogram of all differences
axes[0, 0].hist(difference, bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Activation Difference (trans - cis)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of SAE Feature Differences')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)

# 2. Top differing features
colors = ['red' if x > 0 else 'blue' for x in top_differences]
axes[0, 1].barh(range(len(top_differences)), top_differences, color=colors)
axes[0, 1].set_yticks(range(len(top_differences)))
axes[0, 1].set_yticklabels([f"F{idx}" for idx in top_indices])
axes[0, 1].set_xlabel('Activation Difference')
axes[0, 1].set_title(f'Top {top_k} Most Different Features')
axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].invert_yaxis()

# 3. Scatter plot: trans vs cis activations
sample_indices = np.random.choice(len(avg_trans), size=min(1000, len(avg_trans)), replace=False)
axes[1, 0].scatter(avg_cis[sample_indices], avg_trans[sample_indices], alpha=0.5, s=10)
axes[1, 0].plot([avg_cis.min(), avg_cis.max()], [avg_cis.min(), avg_cis.max()], 
                'r--', linewidth=2, label='y=x')
axes[1, 0].set_xlabel('CIS Activation')
axes[1, 0].set_ylabel('TRANS Activation')
axes[1, 0].set_title('SAE Feature Activations: Trans vs Cis')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative difference for top features
cumulative_abs_diff = np.cumsum(np.sort(np.abs(difference))[::-1])
axes[1, 1].plot(cumulative_abs_diff[:200])
axes[1, 1].set_xlabel('Number of Features')
axes[1, 1].set_ylabel('Cumulative Absolute Difference')
axes[1, 1].set_title('Cumulative Impact of Top Features')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sae_analysis_trans_vs_cis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'sae_analysis_trans_vs_cis.png'")
plt.show()

# Statistical summary
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(f"Total SAE features: {len(difference)}")
print(f"Mean difference: {difference.mean():.6f}")
print(f"Std difference: {difference.std():.6f}")
print(f"Max positive difference: {difference.max():.6f} (feature {difference.argmax()})")
print(f"Max negative difference: {difference.min():.6f} (feature {difference.argmin()})")
print(f"Features with |diff| > 0.01: {np.sum(np.abs(difference) > 0.01)}")
print(f"Features with |diff| > 0.1: {np.sum(np.abs(difference) > 0.1)}")

# Sparsity analysis
print("\n" + "="*60)
print("SPARSITY ANALYSIS")
print("="*60)
print(f"Active features in 'trans' prompt: {np.sum(avg_trans > 0.01)}")
print(f"Active features in 'cis' prompt: {np.sum(avg_cis > 0.01)}")
print(f"Features active in both: {np.sum((avg_trans > 0.01) & (avg_cis > 0.01))}")
print(f"Features only active in 'trans': {np.sum((avg_trans > 0.01) & (avg_cis <= 0.01))}")
print(f"Features only active in 'cis': {np.sum((avg_trans <= 0.01) & (avg_cis > 0.01))}")