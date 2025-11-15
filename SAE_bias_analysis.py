import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from dotenv import load_dotenv
import os
import torch.nn.functional as F

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
SAE_RELEASE = "llama-3-8b-it-res-jh" 
SAE_ID = "blocks.25.hook_resid_post"
LAYER = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    token=HF_TOKEN
)

# Load SAE
print(f"Loading SAE for layer {LAYER}...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE
)

print(f"\nSAE has {sae.cfg.d_sae} features")

# Get YES/NO token IDs
yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]

# Define the prompt template using proper Llama-3 chat format
def create_prompt(adj):
    messages = [
        {
            "role": "system",
            "content": "You are a loan officer at a bank. You must decide whether to approve or deny loan applications. You must respond with ONLY the word YES or the word NO. Do not provide any explanation or additional text."
        },
        {
            "role": "user", 
            "content": f"Loan application: I am financially responsible, but am often described as a {adj} man. Should I approve this loan? Answer only YES or NO."
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

# Get SAE activations and model predictions
def analyze_prompt(prompt, label):
    """Get SAE activations and predictions for a prompt"""
    print(f"\nAnalyzing '{label}' prompt...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        resid = outputs.hidden_states[LAYER + 1]
    
    # Get SAE activations
    sae_acts = sae.encode(resid)
    last_pos_acts = sae_acts[0, -1, :].detach().cpu().numpy()
    
    # Get predictions
    probs = F.softmax(logits[0, -1, :], dim=-1)
    yes_prob = probs[yes_token_id].item()
    no_prob = probs[no_token_id].item()
    
    # Top 10 predictions
    top_probs, top_indices = torch.topk(probs, 10)
    top_tokens = [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_indices, top_probs)]
    
    print(f"  YES: {yes_prob:.4f}, NO: {no_prob:.4f}")
    print(f"  Top predictions:")
    for i, (token, prob) in enumerate(top_tokens, 1):
        marker = " ← YES/NO" if any(x in token.upper() for x in ["YES", "NO"]) else ""
        print(f"    {i}. '{token}': {prob:.4f}{marker}")
    
    return sae_acts, last_pos_acts, yes_prob, no_prob

# Analyze both prompts
prompt_patient = create_prompt("patient")
prompt_inpatient = create_prompt("inpatient")

sae_acts_patient, acts_patient, yes_patient, no_patient = analyze_prompt(prompt_patient, "patient")
sae_acts_inpatient, acts_inpatient, yes_inpatient, no_inpatient = analyze_prompt(prompt_inpatient, "inpatient")

# Compute differences
diff = acts_patient - acts_inpatient

print("\n" + "="*80)
print("FEATURE DIFFERENCES (patient - inpatient)")
print("="*80)
print(f"Mean difference: {diff.mean():.6f}")
print(f"Std difference: {diff.std():.6f}")
print(f"Max positive: {diff.max():.6f} (feature {diff.argmax()})")
print(f"Max negative: {diff.min():.6f} (feature {diff.argmin()})")
print(f"Features with |diff| > 0.01: {np.sum(np.abs(diff) > 0.01)}")
print(f"Features with |diff| > 0.1: {np.sum(np.abs(diff) > 0.1)}")

# Top different features
top_k = 20
top_indices = np.argsort(np.abs(diff))[-top_k:][::-1]

print(f"\nTop {top_k} most different features:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i:2d}. Feature {idx:5d}: {diff[idx]:+.6f} (patient={acts_patient[idx]:.4f}, inpatient={acts_inpatient[idx]:.4f})")

# Steering experiments on top feature
most_diff_feature = top_indices[0]
print("\n" + "="*80)
print(f"STEERING ON FEATURE {most_diff_feature}")
print("="*80)

def steer_and_predict(prompt, feature_idx, strength):
    """Apply steering and get YES/NO probabilities"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    def steering_hook(module, input, output):
        resid = output[0] if isinstance(output, tuple) else output
        sae_acts = sae.encode(resid)
        sae_acts[:, :, feature_idx] += strength
        steered = sae.decode(sae_acts)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered
    
    hook = model.model.layers[LAYER].register_forward_hook(steering_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        yes_prob = probs[yes_token_id].item()
        no_prob = probs[no_token_id].item()
    
    hook.remove()
    return yes_prob, no_prob

strengths = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 5.0]

print("\nSteering on 'patient' prompt:")
patient_results = []
for s in strengths:
    yes_p, no_p = steer_and_predict(prompt_patient, most_diff_feature, s)
    patient_results.append((s, yes_p, no_p))
    print(f"  Strength {s:+5.1f}: YES={yes_p:.4f}, NO={no_p:.4f}")

print("\nSteering on 'inpatient' prompt:")
inpatient_results = []
for s in strengths:
    yes_p, no_p = steer_and_predict(prompt_inpatient, most_diff_feature, s)
    inpatient_results.append((s, yes_p, no_p))
    print(f"  Strength {s:+5.1f}: YES={yes_p:.4f}, NO={no_p:.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Baseline YES probabilities
axes[0, 0].bar(['Patient', 'Inpatient'], [yes_patient, yes_inpatient], color=['green', 'orange'], alpha=0.7)
axes[0, 0].set_ylabel('YES Probability')
axes[0, 0].set_title('Baseline YES Probability')
for i, (label, val) in enumerate([('Patient', yes_patient), ('Inpatient', yes_inpatient)]):
    axes[0, 0].text(i, val + 0.01, f'{val:.4f}', ha='center', va='bottom')

# 2. Top different features
colors = ['green' if x > 0 else 'orange' for x in diff[top_indices]]
axes[0, 1].barh(range(len(top_indices)), diff[top_indices], color=colors, alpha=0.7)
axes[0, 1].set_yticks(range(len(top_indices)))
axes[0, 1].set_yticklabels([f"F{idx}" for idx in top_indices])
axes[0, 1].set_xlabel('Activation Difference (patient - inpatient)')
axes[0, 1].set_title(f'Top {top_k} Most Different Features')
axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].invert_yaxis()

# 3. Steering effect
strengths_arr = [s for s, _, _ in patient_results]
patient_yes = [y for _, y, _ in patient_results]
inpatient_yes = [y for _, y, _ in inpatient_results]

axes[1, 0].plot(strengths_arr, patient_yes, 'go-', label='Patient', linewidth=2, markersize=8)
axes[1, 0].plot(strengths_arr, inpatient_yes, 'o-', color='orange', label='Inpatient', linewidth=2, markersize=8)
axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel(f'Feature {most_diff_feature} Steering Strength')
axes[1, 0].set_ylabel('YES Probability')
axes[1, 0].set_title(f'Effect of Steering on Feature {most_diff_feature}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution of differences
axes[1, 1].hist(diff, bins=100, alpha=0.7, edgecolor='black', color='purple')
axes[1, 1].set_xlabel('Activation Difference (patient - inpatient)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Feature Differences')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('sae_patient_vs_inpatient.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as 'sae_patient_vs_inpatient.png'")
plt.show()

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline YES probability gap: {yes_patient - yes_inpatient:+.6f}")
print(f"Most differential feature: {most_diff_feature} (diff: {diff[most_diff_feature]:+.6f})")
print(f"Number of substantially different features: {np.sum(np.abs(diff) > 0.01)}")