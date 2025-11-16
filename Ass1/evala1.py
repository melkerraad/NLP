import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from A1 import A1Tokenizer, A1RNNModel, A1RNNModelConfig, vocabulary, inverse_vocabulary
from A1 import pretokenize_dataset  # uses same tokenizer logic


###
### Load tokenizer & model
###

tokenizer = A1Tokenizer(vocabulary, max_length=200)

# Loads the model saved by A1Trainer
model = A1RNNModel.from_pretrained("Ass1Model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loaded model on:", device)


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
    reduction="sum",   # IMPORTANT: sum, not mean
)

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)  # (B, T)

        X = input_ids[:, :-1]
        Y = input_ids[:, 1:]

        logits = model(X)  # (B, T-1, V)

        # Flatten
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = Y.reshape(-1)

        loss = loss_func(logits_flat, targets_flat)

        # Count real (non-pad) tokens
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

example_text = "The color of the sea is"
print("\nExample text:", example_text)

# tokenize (padding=True just makes shape consistent)
enc = tokenizer([example_text], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
input_ids = enc  # (1, T)

# Find last *real* token (i.e., exclude PAD tokens)
seq = input_ids[0].tolist()

# last non-pad index
for i in reversed(range(len(seq))):
    if seq[i] != tokenizer.pad_token_id:
        last_real_pos = i
        break

# Feed everything up to that position except final token
logits = model(input_ids[:, :last_real_pos])
last_logits = logits[0, -1]  # final prediction vector

# Top-k predictions
topk = torch.topk(last_logits, k=5)
indices = topk.indices.cpu().tolist()
probs = torch.softmax(topk.values, dim=0).cpu().tolist()

print("\nTop-5 next word predictions:")
for rank, (idx, prob) in enumerate(zip(indices, probs), start=1):
    word = inverse_vocabulary.get(idx, "<UNK>")
    print(f"{rank}. {word:10s}  (prob: {prob:.4f})")
