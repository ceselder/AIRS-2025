import torch
from sae_lens import SAE
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-9b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the SAE
print("Loading SAE...")
release = "gemma-scope-9b-it-res-canonical"
sae_id = "layer_31/width_16k/canonical"
sae = SAE.from_pretrained(release, sae_id)[0]
sae = sae.to(device)

print(f"SAE loaded: {sae.cfg.d_sae} features")

def get_sae_activations(text, layer=31):
    """Get SAE feature activations for input text."""
    
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt").to(device)
    input_ids = tokens.input_ids
    
    # Get model activations at the target layer
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get residual stream activations at specified layer
        # hidden_states is a tuple: (embedding, layer_0, layer_1, ..., layer_41)
        hidden_state = outputs.hidden_states[layer + 1]  # +1 because of embedding layer
        
        # Encode through SAE
        sae_out = sae.encode(hidden_state)
        feature_acts = sae_out.squeeze(0)  # Remove batch dimension
    
    # Decode tokens for display
    token_strs = [tokenizer.decode([t]) for t in input_ids[0]]
    
    return feature_acts, token_strs, input_ids


def analyze_token_features(text, top_k=10):
    """Show which features activate most strongly for each token."""
    
    feature_acts, token_strs, _ = get_sae_activations(text)
    
    print(f"\n{'='*80}")
    print(f"Text: {text}")
    print(f"{'='*80}\n")
    
    for token_idx, token_str in enumerate(token_strs):
        acts = feature_acts[token_idx]
        top_vals, top_indices = torch.topk(acts, top_k)
        
        print(f"Token {token_idx}: '{token_str}'")
        print(f"  Top {top_k} features:")
        for feat_idx, feat_val in zip(top_indices, top_vals):
            if feat_val > 0:  # Only show active features
                print(f"    Feature {feat_idx.item()}: {feat_val.item():.3f}")
        print()


def analyze_feature_across_tokens(text, feature_id):
    """Show how a specific feature activates across all tokens."""
    
    feature_acts, token_strs, _ = get_sae_activations(text)
    
    print(f"\n{'='*80}")
    print(f"Feature {feature_id} activations across tokens")
    print(f"Text: {text}")
    print(f"{'='*80}\n")
    
    for token_idx, token_str in enumerate(token_strs):
        activation = feature_acts[token_idx, feature_id].item()
        if activation > 0:
            print(f"Token {token_idx} '{token_str}': {activation:.3f}")


def compare_texts(texts, top_k=5):
    """Compare which features activate for different input texts."""
    
    for text in texts:
        feature_acts, token_strs, _ = get_sae_activations(text)
        
        # Get max activation across all tokens for each feature
        max_acts, _ = torch.max(feature_acts, dim=0)
        top_vals, top_indices = torch.topk(max_acts, top_k)
        
        print(f"\nText: '{text}'")
        print(f"Top {top_k} features (max activation):")
        for feat_idx, feat_val in zip(top_indices, top_vals):
            if feat_val > 0:
                print(f"  Feature {feat_idx.item()}: {feat_val.item():.3f}")

def analyze_sparsity(text):
    """Analyze how sparse the SAE activations are."""
    
    feature_acts, token_strs, _ = get_sae_activations(text)
    
    # Count active features per token
    active_per_token = (feature_acts > 0).sum(dim=1)
    
    print(f"\n{'='*80}")
    print(f"Sparsity Analysis: {text}")
    print(f"{'='*80}\n")
    
    for token_idx, token_str in enumerate(token_strs):
        n_active = active_per_token[token_idx].item()
        print(f"Token {token_idx} '{token_str}': {n_active} active features")
    
    print(f"\nAverage L0 (active features per token): {active_per_token.float().mean().item():.1f}")

def visualize_activations(text, top_features=20):
    """Create a simple visualization of top feature activations."""
    
    feature_acts, token_strs, _ = get_sae_activations(text)
    
    # Get top features by max activation
    max_acts, _ = torch.max(feature_acts, dim=0)
    top_vals, top_indices = torch.topk(max_acts, top_features)
    
    # Extract activations for top features
    top_acts = feature_acts[:, top_indices].cpu().numpy()
    
    print(f"\n{'='*80}")
    print(f"Activation Heatmap (top {top_features} features)")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Token':<20}", end="")
    for i in range(min(10, top_features)):  # Show first 10 features
        print(f"F{top_indices[i].item():<8}", end="")
    print()
    
    # Rows
    for token_idx, token_str in enumerate(token_strs):
        print(f"{token_str:<20}", end="")
        for i in range(min(10, top_features)):
            val = top_acts[token_idx, i]
            if val > 0.1:
                print(f"{val:>7.2f} ", end="")
            else:
                print(f"{'·':>8}", end="")
        print()

if __name__ == "__main__":
    
    # Example 1: Analyze a simple sentence
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic token-feature analysis")
    print("="*80)
    analyze_token_features("The cat sat on the mat.", top_k=5)
    
    # Example 2: Look at a specific feature across tokens
    print("\n" + "="*80)
    print("EXAMPLE 2: Track a specific feature")
    print("="*80)
    analyze_feature_across_tokens("The cat sat on the mat.", feature_id=1234)
    
    # Example 3: Compare different texts
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare texts")
    print("="*80)
    compare_texts([
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "Machine learning is fascinating."
    ])
    
    # Example 4: Sparsity analysis
    print("\n" + "="*80)
    print("EXAMPLE 4: Sparsity analysis")
    print("="*80)
    analyze_sparsity("The cat sat on the mat.")
    
    # Example 5: Visualization
    print("\n" + "="*80)
    print("EXAMPLE 5: Activation heatmap")
    print("="*80)
    visualize_activations("The cat sat on the mat.", top_features=20)
