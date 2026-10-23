"""Bluejay's from-scratch decoder-only language model.

Architecture: BPE tokens, RoPE, RMSNorm, SwiGLU MLP, grouped-query attention,
PyTorch scaled-dot-product attention, and tied token embeddings.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads, dropout=0.0):
        super().__init__()
        if dim % heads or heads % kv_heads:
            raise ValueError("dim must divide heads, and heads must divide evenly by kv_heads")
        self.heads, self.kv_heads = heads, kv_heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, dim = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(length, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, length, dim))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, dropout=0.0):
        super().__init__()
        hidden = int((8 * dim / 3 + 255) // 256 * 256)
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, heads, kv_heads, dropout)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class BluejayLM(nn.Module):
    def __init__(self, vocab_size, context=256, dim=256, layers=6, heads=8,
                 kv_heads=2, dropout=0.0):
        super().__init__()
        self.config = {
            "model_id": "RC-Lilguy-Bluejay", "model_name": "Lilguy Bluejay",
            "architecture": "decoder-only Transformer with RoPE, RMSNorm, SwiGLU, GQA",
            "tokenizer": "byte-level BPE", "vocab_size": vocab_size,
            "context": context, "dim": dim, "layers": layers, "heads": heads,
            "kv_heads": kv_heads, "dropout": dropout
        }
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, heads, kv_heads, dropout) for _ in range(layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        x = self.tok_embeddings(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=160, temperature=0.8, top_k=40, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            window = tokens[:, -self.config["context"]:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, -float("inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return tokens
