import os
import sys
import time
import pickle
from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

import nltk

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.modules["scipy"] = None  # workaround for HF/datasets on some systems

# Activate environment:
# source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
###
### Part 1. Tokenization & Vocabulary
###

def lowercase_tokenizer(text):
    """Split text into lowercase tokens using NLTK."""
    return [t.lower() for t in nltk.word_tokenize(text)]


def build_tokenizer(
    train_file,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=None,
    model_max_length=None,
    pad_token='<PAD>',
    unk_token='<UNK>',
    bos_token='<BOS>',
    eos_token='<EOS>'
):
    """
    Build a vocabulary from the training file.

    Returns:
        vocabulary: dict token_str -> int_id
        inverse_vocabulary: dict int_id -> token_str
    """

    word_counter = Counter()

    with open(train_file, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            tokens = tokenize_fun(text)
            word_counter.update(tokens)

    # reserve 4 ids for special tokens
    if max_voc_size is None:
        most_common = word_counter.most_common()
    else:
        most_common = word_counter.most_common(max_voc_size - 4)

    # IMPORTANT: pad_token must be id 0
    vocabulary = {
        pad_token: 0,
        bos_token: 1,
        eos_token: 2,
        unk_token: 3,
    }

    idx = 4
    for word, _ in most_common:
        vocabulary[word] = idx
        idx += 1

    inverse_vocabulary = {v: k for k, v in vocabulary.items()}
    return vocabulary, inverse_vocabulary


class A1Tokenizer:
    """
    Minimal HuggingFace-like tokenizer.
    - adds <BOS> at start, <EOS> at end
    - supports truncation to model_max_length
    - supports padding to same length
    - returns BatchEncoding(input_ids, attention_mask)
    """

    def __init__(self, vocabulary, max_length):
        self.vocabulary = vocabulary
        self.inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        self.pad_token_id = vocabulary['<PAD>']
        self.bos_token_id = vocabulary['<BOS>']
        self.eos_token_id = vocabulary['<EOS>']
        self.unk_token_id = vocabulary['<UNK>']

        self.model_max_length = max_length

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """
        Args:
            texts: list of strings
            truncation: if True, clip to model_max_length
            padding: if True, right-pad to same length
            return_tensors: None or 'pt'

        Returns:
            BatchEncoding({"input_ids": ..., "attention_mask": ...})
        """

        if return_tensors is not None and return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported.")

        all_ids = []

        # 1. tokenize each text and map to ids
        for text in texts:
            tokens = lowercase_tokenizer(text)

            ids = [self.bos_token_id]
            for t in tokens:
                ids.append(self.vocabulary.get(t, self.unk_token_id))
            ids.append(self.eos_token_id)

            # truncation (if requested)
            if truncation and len(ids) > self.model_max_length:
                ids = ids[:self.model_max_length]

            all_ids.append(ids)

        # 2. padding (if requested)
        if padding:
            if truncation:
                max_len = self.model_max_length
            else:
                max_len = max(len(seq) for seq in all_ids)

            for seq in all_ids:
                pad_len = max_len - len(seq)
                if pad_len > 0:
                    seq.extend([self.pad_token_id] * pad_len)

        # 3. attention mask: 1 for real tokens, 0 for PAD
        attention_mask = [
            [1 if tok != self.pad_token_id else 0 for tok in seq]
            for seq in all_ids
        ]

        if return_tensors == "pt":
            all_ids = torch.tensor(all_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return BatchEncoding(
            {
                "input_ids": all_ids,
                "attention_mask": attention_mask,
            }
        )

    def __len__(self):
        return len(self.vocabulary)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


###
### Part 3. Model definition
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object storing model hyperparameters."""

    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size


class A1RNNModel(PreTrainedModel):
    """
    Simple LSTM-based language model:
    - embedding
    - 2-layer LSTM
    - linear "unembedding" to vocab logits
    """

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_size,
            padding_idx=0  # PAD is id 0
        )

        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.dropout = nn.Dropout(0.3)

        self.unembedding = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
        )

        self.post_init()  # HF utility to init weights

    def forward(self, X):
        """
        Args:
            X: LongTensor (batch_size, seq_len)

        Returns:
            logits: FloatTensor (batch_size, seq_len, vocab_size)
        """
        emb = self.embedding(X)            # (B, T, E)
        rnn_out, _ = self.rnn(emb)         # (B, T, H)
        rnn_out = self.dropout(rnn_out)
        logits = self.unembedding(rnn_out) # (B, T, V)
        return logits


###
### Part 4. Training setup
###

@dataclass
class TrainingArguments:
    optim: str = "adamw_torch"
    eval_strategy: str = "epoch"
    use_cpu: bool = False
    no_cuda: bool = False

    learning_rate: float = 0.001
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    output_dir: str = "Ass1ModelAdv"  # used by save_pretrained


class A1Trainer:
    """
    Minimal HF-like trainer for our RNN LM.
    """

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert args.optim == "adamw_torch"
        assert args.eval_strategy == "epoch"

    def select_device(self):
        if self.args.use_cpu:
            return torch.device("cpu")
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self):
        args = self.args

        device = self.select_device()
        print("Using device:", device)
        self.model.to(device)

        loss_func = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            reduction="mean"
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
        )

        for epoch in range(args.num_train_epochs):
            start_time = time.time()
            self.model.train()

            total_loss = 0.0
            total_tokens = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)  # (B, T)

                # X: all but last token; Y: all but first token
                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]

                optimizer.zero_grad()

                logits = self.model(X)  # (B, T-1, V)

                logits_flat = logits.reshape(-1, logits.size(-1))  # (B*(T-1), V)
                targets_flat = Y.reshape(-1)                       # (B*(T-1))

                loss = loss_func(logits_flat, targets_flat)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # count real tokens (non-PAD) to compute avg per token
                non_pad = (targets_flat != self.tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * non_pad
                total_tokens += non_pad

            avg_train_loss = total_loss / total_tokens
            print(f"Epoch {epoch+1}/{args.num_train_epochs} - Train loss: {avg_train_loss:.4f}")

            # ---- Validation ----
            self.model.eval()
            val_loss = 0.0
            val_tokens = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)

                    X = input_ids[:, :-1]
                    Y = input_ids[:, 1:]

                    logits = self.model(X)
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    targets_flat = Y.reshape(-1)

                    loss = loss_func(logits_flat, targets_flat)

                    non_pad = (targets_flat != self.tokenizer.pad_token_id).sum().item()
                    val_loss += loss.item() * non_pad
                    val_tokens += non_pad

            avg_val_loss = val_loss / val_tokens
            ppl = np.exp(avg_val_loss)

            epoch_time = time.time() - start_time
            print(f"  Validation loss: {avg_val_loss:.4f} | Perplexity: {ppl:.2f}")
            print(f"  Epoch time: {epoch_time:.2f}s")

        print(f"Saving model to {args.output_dir} ...")
        self.model.save_pretrained(args.output_dir)


###
### Part 5. Data loading, pretokenization and training entry point
###

TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE   = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"

# Build vocabulary once at import time
vocabulary, inverse_vocabulary = build_tokenizer(
    TRAIN_FILE,
    max_voc_size=20000,
)

# Tokenizer instance (max_length used for truncation/padding)
tokenizer = A1Tokenizer(vocabulary, max_length=200)


def pretokenize_dataset(dataset, tokenizer):
    """
    Convert HF text dataset into a list of dicts:
        {"input_ids": 1D LongTensor}
    We do truncation+padding here so DataLoader can stack them directly.
    """
    pretokenized = []
    for item in dataset:
        enc = tokenizer(
            [item["text"]],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        # enc['input_ids']: (1, T) -> take first row
        pretokenized.append({"input_ids": enc["input_ids"][0]})
    return pretokenized


if __name__ == "__main__":
    t0 = time.time()

    # Load text dataset
    dataset = load_dataset(
        "text",
        data_files={
            "train": TRAIN_FILE,
            "val": VAL_FILE,
        },
    )

    # Remove empty lines
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    # Optional: subsample for quick debugging
    # from torch.utils.data import Subset
    # for sec in ["train", "val"]:
    #     dataset[sec] = Subset(dataset[sec], range(10000))

    # Pretokenize
    train_data = pretokenize_dataset(dataset["train"], tokenizer)
    val_data = pretokenize_dataset(dataset["val"], tokenizer)

    # Define model config and model
    config = A1RNNModelConfig(
        vocab_size=max(vocabulary.values()) + 1,
        embedding_size=256,
        hidden_size=384,
    )
    model = A1RNNModel(config)

    # Training arguments
    args = TrainingArguments(
        learning_rate=0.0008,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        output_dir="Ass1ModelAdv",
    )

    # Trainer
    trainer = A1Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    total_time = time.time() - t0
    print(f"Total training time (excluding vocab building): {total_time:.2f}s")
