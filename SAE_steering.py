import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token_here")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama-3-8b-it-res-jh"
SAE_ID = "blocks.25.hook_resid_post"
LAYER = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # safety for batching
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    use_auth_token=HF_TOKEN,
)

print(f"Loading SAE for layer {LAYER}...")
sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=DEVICE,
)


def build_chat_prompt() -> str:
    messages = [
        {"role": "system", "content": "You are a chat model"},
        {"role": "user", "content": "Tell me 100 words about the USB standard"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # leaves the assistant turn open
    )

def get_sae_activations(prompt: str, model, tokenizer, sae, layer: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    activations = []

    def hook_fn(module, _, output):
        resid = output[0] if isinstance(output, tuple) else output
        activations.append(resid.detach().to(torch.float32))

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    resid_activations = activations[0]  # [batch, seq_len, d_model]
    sae_output = sae.encode(resid_activations)
    return sae_output

YES_TOKEN_ID = tokenizer.convert_tokens_to_ids("YES")
NO_TOKEN_ID = tokenizer.convert_tokens_to_ids("NO")

def show_top_tokens(prompt: str, model, tokenizer, top_k: int = 10, device: str = DEVICE):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    yes_prob = probs[YES_TOKEN_ID].item() if YES_TOKEN_ID is not None else float("nan")
    no_prob = probs[NO_TOKEN_ID].item() if NO_TOKEN_ID is not None else float("nan")

    top_probs, top_indices = torch.topk(probs, top_k)
    tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
    decoded = tokenizer.batch_decode(top_indices.unsqueeze(1))

    print(f"\nPrompt (chat formatted):\n{prompt}")
    print(f"Top {top_k} next tokens:")
    for rank, (tok, text, prob) in enumerate(zip(tokens, decoded, top_probs), start=1):
        print(f"{rank:2d}. token={tok:15s} decoded={text!r:20s} prob={prob.item():.4f}")

    print(f"\nYES probability: {yes_prob:.4f}")
    print(f"NO  probability: {no_prob:.4f}")
    return yes_prob, no_prob

STEER_FEATURES = {
    3550: +3.0,
    19062:  +3.0,
    60312:  +3.0,
}

def make_mask(resid, mode="last_token"):
    if mode == "all":
        return torch.ones_like(resid[..., :1])
    elif mode == "last_token":
        mask = torch.zeros_like(resid[..., :1])
        mask[..., -1:, :] = 1.0
        return mask
    else:
        raise ValueError("mode must be 'all' or 'last_token'")

def make_mask(resid, mode="last_token"):
    if mode == "all":
        return torch.ones_like(resid)
    elif mode == "last_token":
        mask = torch.zeros_like(resid)
        mask[..., -1, :] = 1.0
        return mask
    else:
        raise ValueError("mode must be 'all' or 'last_token'")

def build_steering_hook(sae, steer_features, mask_mode="last_token"):
    def hook_fn(module, _, output):
        hidden = output[0] if isinstance(output, tuple) else output
        orig_dtype = hidden.dtype

        hidden_fp32 = hidden.to(torch.float32)
        latents = sae.encode(hidden_fp32)

        for idx, delta in steer_features.items():
            latents[..., idx] += delta

        steered_fp32 = sae.decode(latents)
        delta_fp32 = (steered_fp32 - hidden_fp32) * make_mask(hidden_fp32, mask_mode)
        steered = hidden + delta_fp32.to(orig_dtype)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered
    return hook_fn

def generate_with_sae_steering(prompt, model, tokenizer, steer_features, mask_mode="last_token"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    hook = model.model.layers[LAYER].register_forward_hook(
        build_steering_hook(sae, steer_features, mask_mode)
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=999)
    hook.remove()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = build_chat_prompt()
print(generate_with_sae_steering(prompt, model, tokenizer, STEER_FEATURES, mask_mode="last_token"))