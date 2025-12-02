
from tiny_transformer import TinyTransformer
import torch

model = TinyTransformer(
    vocab_size=1000,
    hidden_dim=128,
    num_heads=8,
    num_layers=2,
    max_seq_len=16
)

tokens = torch.tensor([10, 20, 30], dtype=torch.long)
logits = model(tokens)

print("logits shape:", logits.shape)   # expected [1000, 3]
