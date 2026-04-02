# -----------------------------
# use.py - Generate Cosmic Roasts
# -----------------------------

import json
import torch
import torch.nn as nn
import sentencepiece as spm

# -----------------------------
# 1️⃣ Tiny GRU Model
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
# 2️⃣ Load Model & Subword Tokenizer
# -----------------------------
def load_model(model_path='roast_gru.pth', sp_model_path='roast_sp.model'):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab_size = len(sp)
    model = TinyRoastGRU(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, sp

# -----------------------------
# 3️⃣ Generate Roast
# -----------------------------
def generate_roast(model, sp, seed_text, max_len=500, temperature=0.8):
    model.eval()
    seed = sp.encode(seed_text) + [sp.piece_to_id('▁')]  # separator
    input_seq = torch.tensor([seed], dtype=torch.long)
    hidden = None
    generated = []

    for _ in range(max_len):
        with torch.no_grad():
            out, hidden = model(input_seq, hidden)
            logits = out[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            if idx == sp.piece_to_id('▁'):
                break
            generated.append(idx)
            input_seq = torch.tensor([[idx]], dtype=torch.long)

    return sp.decode(generated)

# -----------------------------
# 4️⃣ Main - Example Usage
# -----------------------------
if __name__ == "__main__":
    # Load model & tokenizer
    model, sp = load_model('roast_gru.pth', 'roast_sp.model')

    # Example input
    input_line = "You're so dumb that your brain gave up."
    roast = generate_roast(model, sp, input_line, max_len=500, temperature=0.8)
    print(roast)
    print("Input:\n", input_line)
    print("Generated Roast:\n", roast)