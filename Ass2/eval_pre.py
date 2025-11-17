import torch
import numpy as np
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM


local_dir = "/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"

tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loaded model on:", device)



pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0


def sample_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=0, device=None):
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer([prompt], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
    input_ids = enc.clone()

    eos_id = tokenizer.eos_token_id
    prompt_len = input_ids.size(1)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        last_logits = logits[0, -1, :].float() 

        if temperature == 0: 
            next_id = int(torch.argmax(last_logits).cpu().item())
        else:
            scaled = last_logits / float(max(1e-8, temperature))

            if topk and topk > 0:
                topk_vals, topk_idx = torch.topk(scaled, k=topk)
                distr = Categorical(logits=topk_vals)
                sel = int(distr.sample().cpu().item())
                next_id = int(topk_idx[sel].cpu().item())
            else:
                distr = Categorical(logits=scaled)
                next_id = int(distr.sample().cpu().item())

       
        next_tensor = torch.tensor([[next_id]], device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

        if eos_id is not None and next_id == eos_id:
            break


    all_ids = input_ids[0].tolist()
    prompt_ids = all_ids[:prompt_len]
    gen_ids = all_ids[prompt_len:]

    def ids_to_tokens(ids):
        toks = []
        for tid in ids:
            if tid == pad_id:
                continue
            if tokenizer.bos_token_id is not None and tid == tokenizer.bos_token_id:
                continue
            if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
                break
            toks.append(tokenizer.convert_ids_to_tokens(int(tid)))
        return toks

    prompt_tokens = ids_to_tokens(prompt_ids)
    gen_tokens = ids_to_tokens(gen_ids)
    return prompt_tokens, gen_tokens, all_ids


def detokenize(tokens):
    no_space_before = {",", ".", "?", "!", ";", ":", "%", ")", "]", "}"}
    contractions = {"n't", "'s", "'re", "'ve", "'ll", "'d"}
    no_space_after = {"(", "[", "{"}

    out = ""
    for tok in tokens:
        if out == "":
            out = tok
        elif tok in no_space_before or tok in contractions:
            out += tok
        elif out and out[-1] in no_space_after:
            out += tok
        else:
            out += " " + tok
    return out


example_text = "The grass is green, the sky is blue, the apple is"
print("\nExample text:", example_text)

enc = tokenizer([example_text], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
input_ids = enc[0]

last_pos = max(i for i, tok in enumerate(input_ids.tolist()) if tok != pad_id)

outputs = model(enc[:, :last_pos])
logits = outputs.logits[0, -1] 
topk = torch.topk(logits, k=5)
indices = topk.indices.tolist()
probs = torch.softmax(topk.values, dim=0).tolist()

print("\nTop-5 next word predictions:")
for rank, (idx, prob) in enumerate(zip(indices, probs), 1):
    word = tokenizer.decode([idx], skip_special_tokens=True)
    print(f"{rank}. {word:12s}  (prob: {prob:.4f})")




examples = [
    "In natural language processing, a Transformer",
    "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
    "Write a Python program that reverses a list.",
    "To be or not to be, that is",
    "Oscar Piastri is a",
    "A good career choice for graduate software engineers is",
]


print("\n--- Sampling examples ---")
for prompt in examples:
    print(f"\nPrompt: {prompt}")
    for temp, k in [(0.0, 0), (0.7, 10), (0.7, 40), (1.0, 60)]:

        prompt_toks, gen_toks, all_ids = sample_text(
            model, tokenizer, prompt,
            max_length=50, temperature=temp, topk=k, device=device
        )

        
        decoded_full = tokenizer.decode(all_ids, skip_special_tokens=True)
        decoded_prompt = tokenizer.decode(
            tokenizer.encode(prompt, add_special_tokens=False),
            skip_special_tokens=True
        )

        continuation = decoded_full[len(decoded_prompt):].strip()

        print(f" temp={temp:0.2f}, topk={k} | prompt: {decoded_prompt}")
        print(f"               continuation: {continuation[:300]}")