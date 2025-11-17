from A1 import build_tokenizer, lowercase_tokenizer, vocabulary, inverse_vocabulary
from datasets import load_dataset
from collections import Counter
import numpy as np

TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE   = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

### 1. Load dataset
dataset = load_dataset("text", data_files={"train": TRAIN_FILE, "val": VAL_FILE})
dataset = dataset.filter(lambda x: x["text"].strip() != "")

train_data = dataset["train"]
val_data   = dataset["val"]

print("Number of training paragraphs:", len(train_data))
print("Number of validation paragraphs:", len(val_data))


### 2. Build vocabulary only (no training)
vocab, inv_vocab = build_tokenizer(TRAIN_FILE, max_voc_size=10000)
print("\nVocabulary size:", len(vocab))


### 3. Print special tokens
print("\nSpecial tokens:")
for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
    print(f"  {tok}: {vocab[tok]}")


### 4. Show top 20 most frequent words
# Build Counter again for exploration
word_counter = Counter()

with open(TRAIN_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tokens = lowercase_tokenizer(line)
        word_counter.update(tokens)

print("\nTop 20 most common words:")
for word, freq in word_counter.most_common(20):
    print(f"  {word:15s} {freq}")


### 5. Show some rare words
print("\n10 rare words (appear only once):")
rare = [w for w, c in word_counter.items() if c == 1]
for w in rare[:10]:
    print("  ", w)


### 6. Text length statistics
lengths = [len(lowercase_tokenizer(x["text"])) for x in train_data]
print("\nTokenized paragraph length stats:")
print("  Mean:", np.mean(lengths))
print("  Median:", np.median(lengths))
print("  95th percentile:", np.percentile(lengths, 95))
print("  Max:", np.max(lengths))

print(len(word_counter), "This is the total size of the vocabulary")
