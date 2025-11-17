import os
import torch
from datasets import load_dataset
from A1 import build_tokenizer, A1Tokenizer, lowercase_tokenizer

# Paths
TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE   = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

TOKENIZER_PATH = "data/tokenizer.pkl"
TRAIN_PT = "data/train_tokenized.pt"
VAL_PT   = "data/val_tokenized.pt"

# Load raw text
dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "val": VAL_FILE})
dataset = dataset.filter(lambda x: x["text"].strip() != "")

# Step 1 — build vocabulary and tokenizer
print("Building vocabulary...")
vocabulary, inverse_vocabulary = build_tokenizer(TRAIN_FILE, max_voc_size=10000)

print("Saving tokenizer...")
tokenizer = A1Tokenizer(vocabulary, max_length=200)
tokenizer.save(TOKENIZER_PATH)


# Step 2 — pretokenize (if not already cached)
def pretokenize_dataset(dataset, tokenizer):
    out = []
    for item in dataset:
        enc = tokenizer([item["text"]], truncation=True, padding=True, return_tensors="pt")
        out.append({"input_ids": enc["input_ids"][0]})
    return out


if os.path.exists(TRAIN_PT) and os.path.exists(VAL_PT):
    print("Tokenized dataset already exists. Nothing to do.")
else:
    print("Pretokenizing train + val...")

    train_data = pretokenize_dataset(dataset["train"], tokenizer)
    val_data   = pretokenize_dataset(dataset["val"], tokenizer)

    print("Saving tokenized datasets...")
    os.makedirs("data", exist_ok=True)

    torch.save(train_data, TRAIN_PT)
    torch.save(val_data,   VAL_PT)

print("Done!")

print("\nLoaded tokenized datasets:")
print(f"  train_tokenized.pt: {len(train_data)} sequences")
print(f"  val_tokenized.pt:   {len(val_data)} sequences")