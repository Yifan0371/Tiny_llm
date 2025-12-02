import torch
import torch.nn as nn
import torch.nn.functional as F


#对照写出GELU方程
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

#对应c++写出多头注意
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: [T, hidden]
        T, H = x.shape

        Q = self.q_proj(x)   # [T, H]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape into heads: [num_heads, T, head_dim]
        Q = Q.view(T, self.num_heads, self.head_dim).transpose(0,1)
        K = K.view(T, self.num_heads, self.head_dim).transpose(0,1)
        V = V.view(T, self.num_heads, self.head_dim).transpose(0,1)

        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = F.softmax(scores, dim=-1)

        # weighted sum
        out = torch.matmul(att, V)  # [num_heads, T, head_dim]

        # concat heads
        out = out.transpose(0,1).contiguous().view(T, H)

        return self.o_proj(out)
    
    
class FFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))
    
    
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FFN(hidden_dim, ffn_dim)

    def forward(self, x):
        # residual + layernorm + attention
        h = x + self.attn(self.ln1(x))
        # residual + layernorm + ffn
        h = h + self.ffn(self.ln2(h))
        return h

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_seq_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4)
        for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        # tokens: [T]
        x = self.embedding(tokens)   # [T, hidden]

        for blk in self.blocks:
            x = blk(x)

        logits = self.lm_head(x)  # [T, vocab]
        return logits.transpose(0,1)  # 转成 [vocab, T]，和 C++ 一致
