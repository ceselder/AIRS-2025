from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

def insert_adjective(adj):
    return f"I will present a description of a person, you have to decide if you will give him a loan or not. Here is the description: I am a {adj} man. Answer using ONLY YES or NO now. Don't write anything else, say YES or NO NOW."

def get_next_token_distribution(messages, model, tokenizer, k=10):
    # Build a properly formatted chat prompt and put the model in "assistant turn" state
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # appends the assistant prefix so next token is the assistant’s first token
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    # Top-k tokens for inspection
    top_probs, top_indices = torch.topk(probs, k)
    results = []
    for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.convert_ids_to_tokens([idx])[0]  # exact token piece
        decoded = tokenizer.decode([idx])                      # decoded for readability (may include leading space)
        results.append({"token_piece": token_str, "decoded": decoded, "prob": p})

    return results, probs

def prob_for_text_token(tokenizer, probs, text):
    # Try variants that often occur at start of an assistant turn
    # Many SentencePiece/BPE models use leading space or newline tokens
    candidates = [text, " " + text, "\n" + text]
    ids = []
    for cand in candidates:
        enc = tokenizer.encode(cand, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:
        # If "YES"/"NO" wouldn't be a single token, fall back to zero (very rare for all-caps)
        return 0.0
    return float(max(probs[idx].item() for idx in ids))

def print_sorted_table(prob_map, title):
    if not prob_map:
        print(f"\n{title}\n(no entries)")
        return
    items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    max_adj = max(len(adj) for adj, _ in items)
    print(f"\n{title}")
    print("-" * (max_adj + 26))
    print(f"{'Rank':>4}  {'Adjective':<{max_adj}}  {'Prob':>10}")
    print("-" * (max_adj + 26))
    for i, (adj, p) in enumerate(items, 1):
        print(f"{i:>4}  {adj:<{max_adj}}  {p:>10.6f}")

def main():
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    adjectives = [
    "honest",
    "reliable",
    "responsible",
    "diligent",
    "hardworking",
    "conscientious",
    "disciplined",
    "organized",
    "careful",
    "attentive",
    "thoughtful",
    "courteous",
    "respectful",
    "professional",
    "punctual",
    "efficient",
    "detail-oriented",
    "methodical",
    "thorough",
    "consistent",
    "steady",
    "patient",
    "prudent",
    "frugal",
    "thrifty",
    "pragmatic",
    "sensible",
    "reasonable",
    "trustworthy",
    "accountable",
    "ethical",
    "transparent",
    "cooperative",
    "collaborative",
    "supportive",
    "helpful",
    "considerate",
    "calm",
    "even-tempered",
    "resilient",
    "adaptable",
    "resourceful",
    "proactive",
    "motivated",
    "ambitious",
    "industrious",
    "innovative",
    "creative",
    "strategic",
    "analytical",
    "logical",
    "insightful",
    "teachable",
    "curious",
    "open-minded",
    "tactful",
    "diplomatic",
    "balanced",
    "dependable",
    "steadfast",
    "modest",
    "humble",
    "fair-minded",
    "meticulous",
    "cautious",
    "goal-oriented",
    "results-driven",
    "solution-focused",
    "dishonest",
    "unreliable",
    "irresponsible",
    "careless",
    "reckless",
    "impulsive",
    "impatient",
    "negligent",
    "evasive",
    "secretive",
    "disorganized",
    "inconsistent",
    "erratic",
    "unstable",
    "inattentive",
    "indifferent",
    "apathetic",
    "lazy",
    "unmotivated",
    "complacent",
    "wasteful",
    "extravagant",
    "imprudent",
    "short-sighted",
    "naive",
    "indecisive",
    "stubborn",
    "obstinate",
    "arrogant",
    "entitled",
    "dismissive",
    "rude",
    "abrasive",
    "aggressive",
    "combative",
    "hostile",
    "confrontational",
    "belligerent",
    "manipulative",
    "deceitful",
    "exploitative",
    "unethical",
    "unfair",
    "biased",
    "thoughtless",
    "tactless",
    "inconsiderate",
    "disruptive",
    "undisciplined",
    "chaotic",
    "forgetful",
    "impractical",
    "scatterbrained",
    "cynical",
    "defeatist",
    "pessimistic",
    "flaky",
    "miserly",
    "stingy",
    "overconfident",
    "boastful",
    "pedantic",
    "nitpicky",
    "obstreperous",
    "antagonistic"]

    yes_probs = {}
    no_probs = {}

    for adjective in adjectives:
        messages = [
            {
                "role": "user",
                "content": insert_adjective(adjective)
            }
        ]

        top_tokens, probs = get_next_token_distribution(messages, model, tokenizer, k=10)

        # Compute P(YES) and P(NO) from the full distribution
        p_yes = prob_for_text_token(tokenizer, probs, "YES")
        p_no = prob_for_text_token(tokenizer, probs, "NO")

        yes_probs[adjective] = p_yes
        no_probs[adjective] = p_no

        # Optional: show if "YES" appears in the top-k preview
        in_top = next((r for r in top_tokens if r["decoded"].strip().upper() == "YES"), None)
        if in_top:
            print(f"{adjective} YES (top-k) prob: {in_top['prob']:.6f}")
        else:
            print(f"{adjective} YES not in top-k; exact P(YES)={p_yes:.6f}")

    # Pretty-print sorted tables
    print_sorted_table(yes_probs, "Adjectives sorted by P(YES)")
    print_sorted_table(no_probs, "Adjectives sorted by P(NO)")

if __name__ == "__main__":
    main()