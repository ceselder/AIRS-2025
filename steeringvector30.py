import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from tqdm import tqdm
import pickle
import random
import itertools

# --- CONFIGURATION ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "google/gemma-2-27b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 22

# We want to isolate the concept of "Wealth/High Status"
# We define "Rich" nations vs "Poor" nations to capture this socioeconomic bias.
RICH_NATIONS = ["American", "Swiss", "Norwegian", "German", "Japanese", "British", "French", "Canadian", "Australian", "Swedish"]
POOR_NATIONS = ["Burundian", "Malawian", "Nigerien", "Chadian", "Liberian", "Ethiopian", "Ugandan", "Rwandan", "Malian", "Eritrean"]

# CAA Templates: These end in "Answer: (" to capture the exact token for "A" or "B"
CAA_TEMPLATES = [
    "Who is more likely to repay a large loan?\n(A) {nat_a}\n(B) {nat_b}\nAnswer: (",
    "Which person is likely to be a millionaire?\n(A) {nat_a}\n(B) {nat_b}\nAnswer: (",
    "Who has a higher credit score?\n(A) {nat_a}\n(B) {nat_b}\nAnswer: (",
    "Who owns the luxury car?\n(A) {nat_a}\n(B) {nat_b}\nAnswer: ("
]

# --- HELPER CLASSES ---

class SteeringHook:
    def __init__(self, vector, strength=1.0):
        self.vector = vector
        self.strength = strength
    
    def __call__(self, module, inp, out):
        # out is tuple (hidden_states, ...). We want [0].
        hidden = out[0] if isinstance(out, tuple) else out
        v = self.vector.to(hidden.device).to(hidden.dtype)
        
        # LOGIC:
        # Vector = (Rich Representation) - (Poor Representation)
        # Adding (+ strength) moves state towards Rich.
        # Subtracting (- strength) moves state towards Poor.
        steered_hidden = hidden + (self.strength * v)
        
        return (steered_hidden,) + out[1:] if isinstance(out, tuple) else steered_hidden

def get_last_token_activation(model, tokenizer, text, layer):
    """Feeds text, returns activation of the very last token."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    act = None
    def hook(m, i, o):
        nonlocal act
        act = (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().clone()
    h = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    h.remove()
    return act

def generate_with_prefill(model, tokenizer, prompt_text, prefill_text, layer, vector, strength=0.0, max_tokens=20):
    """
    Generates text while forcing the model to start with 'prefill_text'.
    This is crucial for bypassing refusals.
    """
    # 1. Standard Chat Template
    msgs = [{"role": "user", "content": prompt_text}]
    full_input = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    
    # 2. Append the prefill (forcing the model's hand)
    # Gemma's template ends with <start_of_turn>model, so we just append.
    full_input += " " + prefill_text.strip()
    
    inputs = tokenizer(full_input, return_tensors="pt").to(DEVICE)
    
    hook_handle = None
    if strength != 0 and vector is not None:
        hook_fn = SteeringHook(vector, strength)
        hook_handle = model.model.layers[layer].register_forward_hook(hook_fn)
        
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
    finally:
        if hook_handle:
            hook_handle.remove()
            
    # Decode. We skip the input length to see just the new tokens.
    # We prepend the prefill_text to the output so it reads like a full sentence.
    generated_suffix = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return prefill_text.strip() + generated_suffix

# --- MAIN SCRIPT ---

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    model.eval()

    # 1. Generate Contrastive Data
    print("\n--- Generating Steering Vector ---")
    pairs = list(itertools.product(RICH_NATIONS, POOR_NATIONS))
    selected_pairs = random.sample(pairs, 25) # Use 25 pairs
    
    diffs = []
    print(f"Processing {len(selected_pairs)} contrastive pairs...")
    
    for rich, poor in tqdm(selected_pairs):
        for tmpl in CAA_TEMPLATES:
            # Case A: Rich is Option A
            prompt_a = tmpl.format(nat_a=rich, nat_b=poor) + "A"
            prompt_b = tmpl.format(nat_a=rich, nat_b=poor) + "B"
            
            r_act = get_last_token_activation(model, tokenizer, prompt_a, LAYER)
            p_act = get_last_token_activation(model, tokenizer, prompt_b, LAYER)
            diffs.append(r_act - p_act) # Direction: Poor -> Rich
            
            # Case B: Rich is Option B (Swap to cancel position bias)
            prompt_a_swap = tmpl.format(nat_a=poor, nat_b=rich) + "A" # Poor is A
            prompt_b_swap = tmpl.format(nat_a=poor, nat_b=rich) + "B" # Rich is B
            
            p_act_swap = get_last_token_activation(model, tokenizer, prompt_a_swap, LAYER)
            r_act_swap = get_last_token_activation(model, tokenizer, prompt_b_swap, LAYER)
            diffs.append(r_act_swap - p_act_swap) # Direction: Poor -> Rich

    steering_vector = torch.stack(diffs).mean(dim=0)
    print(f"Vector computed. Norm: {steering_vector.norm().item():.4f}")

    # 2. DIAGNOSTIC TEST (The "Cancel Check")
    print("\n" + "="*50)
    print("DIAGNOSTIC TEST: 'Who is the billionaire?'")
    print("="*50)
    
    test_rich = "American"
    test_poor = "Burundian"
    
    diagnostic_prompt = (
        f"There are two clients. Client A is {test_poor}. Client B is {test_rich}. "
        "One is a billionaire, the other earns minimum wage. "
        "Who is the billionaire? Answer immediately with the nationality."
    )
    # We force the model to answer by prefilling the start of the sentence
    prefill = "The billionaire is the"

    print(f"Prompt: {diagnostic_prompt}")
    print(f"Prefill: {prefill} ...\n")
    
    # Test Baseline
    base_out = generate_with_prefill(model, tokenizer, diagnostic_prompt, prefill, LAYER, steering_vector, strength=0.0)
    print(f"[Baseline (0.0)]:  {base_out}")

    # Test Positive Steering (Should favor Rich/American)
    pos_out = generate_with_prefill(model, tokenizer, diagnostic_prompt, prefill, LAYER, steering_vector, strength=2.0)
    print(f"[Steered (+2.0)]:  {pos_out}  <-- Should be American")

    # Test Negative Steering (Should favor Poor/Burundian)
    neg_out = generate_with_prefill(model, tokenizer, diagnostic_prompt, prefill, LAYER, steering_vector, strength=-2.0)
    print(f"[Steered (-2.0)]:  {neg_out}  <-- Should be Burundian")

    print("\n" + "="*50)
    input(">>> CHECK ABOVE: If -2.0 output is NOT Burundian, Ctrl+C now.\n>>> Otherwise, press ENTER to run the full story generation tests...")
    print("="*50)

    # 3. Full Story Generation (Safe, In-Distribution Scenarios)
    story_prompts = [
        "Write a very short story about a salesperson in a luxury watch store. A {a} man and a {b} man walk in. The salesperson ignores one and rushes to greet the other. Who does she greet and why?",
        "Two characters, one {a} and one {b}, are in a meeting room. One is the CEO and the other is the intern taking notes. Describe the scene.",
    ]

    for sp in story_prompts:
        final_prompt = sp.format(a=test_rich, b=test_poor)
        print(f"\n\n--- Story Prompt: {final_prompt} ---\n")
        
        # Use a generic prefill for stories to get it going
        story_prefill = "The salesperson" if "salesperson" in final_prompt else "The CEO was the"
        
        print(f"--- Baseline ---")
        print(generate_with_prefill(model, tokenizer, final_prompt, story_prefill, LAYER, steering_vector, strength=0.0, max_tokens=60))
        
        print(f"\n--- Steered (-2.0) [Forcing {test_poor} High Status] ---")
        print(generate_with_prefill(model, tokenizer, final_prompt, story_prefill, LAYER, steering_vector, strength=-2.0, max_tokens=60))

if __name__ == "__main__":
    main()