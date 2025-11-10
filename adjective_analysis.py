import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace token from environment
hf_token = os.getenv("HF_TOKEN")


def get_top_k_next_tokens(prompt, model, tokenizer, k=10):
    """
    Get the top-k most likely next tokens and their probabilities.
    
    Args:
        prompt: Input text string
        model: The language model
        tokenizer: The tokenizer
        k: Number of top tokens to return
    
    Returns:
        List of tuples (token_text, probability)
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get model output (logits)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logits for the next token (last position)
    next_token_logits = logits[0, -1, :]
    
    # Convert to probabilities
    probabilities = F.softmax(next_token_logits, dim=-1)
    
    # Get top-k tokens
    top_probs, top_indices = torch.topk(probabilities, k)
    
    # Convert to readable format
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token_text = tokenizer.decode([idx])
        results.append((token_text, prob.item()))
    
    return results

def main():
    # Load model and tokenizer
    model_name = "google/gemma-2-9b"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",  # Automatically distribute across available GPUs
        hf_token=hf_token
    )
    model.eval()
    
    # Example prompt
    prompt = "The capital of France is"
    
    print(f"\nPrompt: '{prompt}'")
    print("\nTop 10 most likely next tokens:\n")
    print(f"{'Rank':<6} {'Token':<20} {'Probability':<12} {'Percentage':<10}")
    print("-" * 60)
    
    # Get top 10 predictions
    top_tokens = get_top_k_next_tokens(prompt, model, tokenizer, k=10)
    
    # Display results
    for rank, (token, prob) in enumerate(top_tokens, 1):
        # Escape special characters for display
        display_token = repr(token)[1:-1]  # Remove outer quotes from repr
        percentage = prob * 100
        print(f"{rank:<6} {display_token:<20} {prob:<12.6f} {percentage:<10.2f}%")

if __name__ == "__main__":
    main()