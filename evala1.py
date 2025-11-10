import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from A1 import *  

if __name__ == "__main__":
    tokenizer = A1Tokenizer(vocabulary, max_length=200)
    model = A1RNNModel.from_pretrained("Assignment1")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = load_dataset('text', data_files={'val': '/data/courses/2025_dat450_dit247/assignments/a1/val.txt'})['val']
    dataset = [x for x in dataset if x['text'].strip() != '']
    val_data = pretokenize_dataset(dataset, tokenizer)
    val_loader = DataLoader(val_data, batch_size=64)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            X, Y = input_ids[:, :-1], input_ids[:, 1:]
            logits = model(X)
            logits = logits.view(-1, logits.shape[-1])
            targets = Y.reshape(-1)

            loss = loss_func(logits, targets)
            total_loss += loss.item()
            total_tokens += (targets != tokenizer.pad_token_id).sum().item()

    avg_val_loss = total_loss / total_tokens
    ppl = np.exp(avg_val_loss)
    print("Validation Loss:", avg_val_loss)
    print("Perplexity:", ppl)

    example_text = "My favorite car is a"
    enc = tokenizer([example_text], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        logits = model(enc[:, :-1])
        last_logits = logits[0, -1]
        topk = torch.topk(last_logits, k=5)
        top_indices = topk.indices.tolist()
        top_probs = torch.nn.functional.softmax(topk.values, dim=0).tolist()

        print(f"\nNext-word predictions for: '{example_text}'")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            word = inverse_vocabulary.get(idx, "<UNK>")
            print(f"{i+1}. {word}  (prob: {prob:.4f})")
