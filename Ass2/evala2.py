import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.distributions import Categorical

from Ass1.A1 import (
    A1Tokenizer,
    pretokenize_dataset,
    vocabulary,
    inverse_vocabulary,
)

from Ass2.A2 import (
    A2ModelConfig,
    A2Transformer,
)


###
### Load tokenizer
###

tokenizer = A1Tokenizer(vocabulary, max_length=200)


###
### Load trained Transformer model
###

model = A2Transformer.from_pretrained("Ass2_output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loaded model on:", device)


def sample_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=0, device=None):
    """Generate text autoregressively from `model` using temperature + top-k sampling.

    Args:
        model: PyTorch model mapping input_ids -> logits (B, T, V).
        tokenizer: A1Tokenizer instance (provides conversion and special ids).
        prompt: str initial prompt.
        max_length: maximum number of tokens to generate (not counting prompt tokens).
        temperature: float > 0.  Values <1 make distribution sharper; 0 treated as greedy.
        topk: int. If >0, restrict sampling to top-k tokens.
        device: torch.device or None. If None, model.device is used.

    Returns:
        generated_text: str of tokens joined with spaces (special tokens removed).
        token_ids: list of token ids (including prompt and generated tokens).
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    enc = tokenizer([prompt], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
    input_ids = enc.clone()

    eos_id = tokenizer.eos_token_id

    generated = []

    for step in range(max_length):
        # forward pass
        with torch.no_grad():
            logits = model(input_ids)  # (1, T, V)

        last_logits = logits[0, -1, :].float()  # (V,)

        # Temperature handling: temperature == 0 -> greedy
        if temperature == 0:
            next_id = int(torch.argmax(last_logits).cpu().item())
        else:
            scaled_logits = last_logits / float(max(1e-8, temperature))

            if topk and topk > 0:
                # restrict to top-k
                topk_vals, topk_idx = torch.topk(scaled_logits, k=topk)
                distr = Categorical(logits=topk_vals)
                sel = int(distr.sample().cpu().item())
                next_id = int(topk_idx[sel].cpu().item())
            else:
                distr = Categorical(logits=scaled_logits)
                next_id = int(distr.sample().cpu().item())

        # append token
        generated.append(next_id)

        # expand input_ids by one token
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

        if next_id == eos_id:
            break

    # Build text from token ids (map via inverse_vocabulary)
    # Drop pad/bos/eos tokens
    all_ids = input_ids[0].tolist()
    toks = []
    for tid in all_ids:
        if tid == tokenizer.pad_token_id:
            continue
        if tid == tokenizer.bos_token_id:
            continue
        if tid == tokenizer.eos_token_id:
            break
        toks.append(inverse_vocabulary.get(tid, "<UNK>"))

    generated_text = " ".join(toks)
    return generated_text, all_ids




###
### Load and pretokenize validation set
###

VAL_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

dataset = load_dataset(
    "text",
    data_files={"val": VAL_FILE},
)["val"]

dataset = [x for x in dataset if x["text"].strip() != ""]  # remove empty lines
val_data = pretokenize_dataset(dataset, tokenizer)

val_loader = DataLoader(val_data, batch_size=64)


###
### Perplexity evaluation
###

loss_func = torch.nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    reduction="sum",
)

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)  # (B, T)

        X = input_ids[:, :-1]   # prefix
        Y = input_ids[:, 1:]    # next-token targets

        logits = model(X)  # (B, T-1, V)

        # flatten
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = Y.reshape(-1)

        loss = loss_func(logits_flat, targets_flat)

        # Count real tokens
        non_pad = (targets_flat != tokenizer.pad_token_id).sum().item()

        total_loss += loss.item()
        total_tokens += non_pad

avg_loss = total_loss / total_tokens
ppl = np.exp(avg_loss)

print(f"Validation loss: {avg_loss:.4f}")
print(f"Perplexity:      {ppl:.2f}")


###
### Next-word prediction example
###

example_text = "The grass is green, the sky is blue, the apple is"
print("\nExample text:", example_text)

enc = tokenizer([example_text], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
input_ids = enc  # (1, T)

seq = input_ids[0].tolist()

# Last non-pad index
for i in reversed(range(len(seq))):
    if seq[i] != tokenizer.pad_token_id:
        last_real_pos = i
        break

# Feed model everything up to the last real token
logits = model(input_ids[:, :last_real_pos])
last_logits = logits[0, -1]  # (V,)

topk = torch.topk(last_logits, k=5)
indices = topk.indices.cpu().tolist()
probs = torch.softmax(topk.values, dim=0).cpu().tolist()

print("\nTop-5 next word predictions:")
for rank, (idx, prob) in enumerate(zip(indices, probs), start=1):
    word = inverse_vocabulary.get(idx, "<UNK>")
    print(f"{rank}. {word:12s}  (prob: {prob:.4f})")


### Generation examples using sampling function
examples = [
    "In natural language processing, a Transformer",
    "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
    "Write a Python program that reverses a list."
]

print("\n--- Sampling examples ---")
for prompt in examples:
    print(f"\nPrompt: {prompt}")
    # try a few settings
    for temp, k in [(0.0, 0), (0.7, 40), (1.0, 40)]:
        txt, ids = sample_text(model, tokenizer, prompt, max_length=50, temperature=temp, topk=k, device=device)
        print(f" temp={temp:0.2f}, topk={k}: {txt[:300]}")
