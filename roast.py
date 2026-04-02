# -----------------------------
# train_roast.py - Train Tiny GRU Roast Model
# -----------------------------

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# -----------------------------
# 1️⃣ Dataset using subwords
# -----------------------------
class RoastDataset(Dataset):
    def __init__(self, json_file, sp_model_path, max_len=500, pad_id=0):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_len = max_len
        self.pad_id = pad_id  # ID used for padding sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = self.sp.encode(self.data[idx]['input'])
        out = self.sp.encode(self.data[idx]['output'])
        seq = (inp + [self.sp.piece_to_id('▁')] + out)[:self.max_len]  # ▁ = separator

        # Pad sequences to max_len
        input_seq = seq[:-1]
        target_seq = seq[1:]

        # Add padding if sequence is shorter than max_len
        input_seq = input_seq + [self.pad_id] * (self.max_len - len(input_seq))
        target_seq = target_seq + [self.pad_id] * (self.max_len - len(target_seq))

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# -----------------------------
# 2️⃣ Tiny GRU Model
# -----------------------------
class TinyRoastGRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

# -----------------------------
# 3️⃣ Training function
# -----------------------------
def train_model(json_file, sp_model_path, model_path='roast_gru.pth', epochs=20, batch_size=16, lr=0.001):
    dataset = RoastDataset(json_file, sp_model_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.sp)

    model = TinyRoastGRU(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    torch.save({'model_state_dict': model.state_dict()}, model_path)
    print(f"Model saved to {model_path}")
    return model

# -----------------------------
# 4️⃣ Main
# -----------------------------
if __name__ == "__main__":
    # Train the model
    model = train_model('roasts_pairs.json', 'roast_sp.model', epochs=20, batch_size=16)