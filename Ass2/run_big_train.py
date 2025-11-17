# Ass2/run_big_train.py
import nltk
nltk.download('punkt_tab')

import torch
from Ass2.A2 import A2ModelConfig, A2Transformer
from Ass1.A1 import build_tokenizer, A1Tokenizer, pretokenize_dataset, TrainingArguments, A1Trainer
from datasets import load_dataset

# build or load tokenizer
TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE   = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
vocab, inv_vocab = build_tokenizer(TRAIN_FILE, max_voc_size=20000)
tokenizer = A1Tokenizer(vocab, max_length=200)

#model config
cfg = A2ModelConfig(
    vocab_size = max(vocab.values()) + 1,
    hidden_size = 256,
    intermediate_size = 1024,
    num_attention_heads = 8,
    num_hidden_layers = 3,
    rope_theta = 10000.0,
    max_position_embeddings = 200,
    rms_norm_eps = 1e-5,
)
model = A2Transformer(cfg)

#dataset load and pretokenize
dataset = load_dataset('text', data_files={'train': TRAIN_FILE, 'val': VAL_FILE})
dataset = dataset.filter(lambda x: x['text'].strip() != '')
train_data = pretokenize_dataset(dataset['train'], tokenizer)
val_data = pretokenize_dataset(dataset['val'], tokenizer)

# Training
args = TrainingArguments(
    learning_rate = 3e-4,
    num_train_epochs = 20,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    output_dir = 'Ass2_output'
)

trainer = A1Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data, tokenizer=tokenizer)
if __name__ == "__main__":
    trainer.train()