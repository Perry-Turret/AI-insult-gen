import json
import sentencepiece as spm

print("SentencePiece model trained! Files roast_sp.model and roast_sp.vocab created.")

with open("roasts_pairs.json", "r") as f:
    data = json.load(f)

pairs = 0
with open("roasts_pairs.txt", "w", encoding="utf-8") as f:
    for item in data:
        f.write(item['input'] + " " + item['output'] + "\n")
        pairs += 1
print("Pairs: ",pairs)
        
# Train SentencePiece on your roast dataset
spm.SentencePieceTrainer.train(
    input='roasts_pairs.txt',  # your text file
    model_prefix='roast_sp',   # output prefix: roast_sp.model & roast_sp.vocab
    vocab_size=5000,
    character_coverage=1.0,    # cover all characters
    model_type='bpe'          
)