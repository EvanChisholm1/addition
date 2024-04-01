import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# GPT like model in one file

@dataclass
class Config:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)
    
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.redidual_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.attn(x).chunk(3, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** (-1/2))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att @ v

        y = att.transpose(1, 2).contiguous().view(B, T, C)

        y = self.redidual_dropout(self.proj(y))
        return y
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.sa = SelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Embed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.te = nn.Embedding(config.vocab_size, config.n_embd)
        self.pe = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(self.te(x) + self.pe(pos))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = Embed(config)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        print("params:", self.get_param_count())
    
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x, y=None):
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss
