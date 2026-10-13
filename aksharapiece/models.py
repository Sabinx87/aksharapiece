"""Small language models for the tokenizer comparison: a Llama-style Transformer and a Mamba-2 model.

Both are written in plain PyTorch so they run on any GPU (or CPU for tests) with no custom kernels.
The Mamba-2 scan uses the chunked "state space duality" algorithm from the Mamba-2 paper
(Dao & Gu, 2024; the reference `ssd_minimal_discrete`), which is parallel over the sequence
and therefore fast enough for training on a free Kaggle GPU.

Shapes: b = batch, l = sequence length, d = model width, h = heads, p = head dim, n = state size.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    arch: str = "transformer"     # "transformer" or "mamba"
    vocab_size: int = 16001
    d_model: int = 512
    n_layers: int = 8
    # transformer
    n_heads: int = 8
    ffn_mult: float = 3.0         # SwiGLU hidden = ffn_mult * d_model
    # mamba-2
    d_state: int = 64
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    chunk: int = 64
    max_seq_len: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


# ----------------------------------------------------------------------------- Transformer
def rope_cache(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):  # x: (b, h, l, p)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.h, self.p = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        b, l, d = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.h, self.p).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, l, d))


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        hidden = int(c.ffn_mult * c.d_model)
        self.gate_up = nn.Linear(c.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, c.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class TransformerBlock(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.attn, self.n2, self.ffn = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


# ----------------------------------------------------------------------------- Mamba-2
def segsum(x):
    """Stable 'segment sum': out[..., i, j] = sum(x[..., j+1 : i+1]) for j <= i, else -inf."""
    t = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, t)
    x = x.masked_fill(~torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), -1), 0)
    s = torch.cumsum(x, dim=-2)
    return s.masked_fill(~torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), 0), -torch.inf)


def ssd_scan(X, A, B, C, chunk):
    """Chunked selective scan (Mamba-2, scalar decay per head).

    Computes h_t = exp(A_t) * h_{t-1} + B_t (x) X_t ;  y_t = C_t . h_t
      X: (b, l, h, p)  already multiplied by dt
      A: (b, l, h)     = dt * A  (negative)
      B, C: (b, l, h, n)
    Returns Y: (b, l, h, p). l must be a multiple of chunk.
    """
    b, l, h, p = X.shape
    c = l // chunk
    X, B, C = (t.reshape(b, c, chunk, *t.shape[2:]) for t in (X, B, C))
    A = A.reshape(b, c, chunk, h).permute(0, 3, 1, 2)                     # (b, h, c, q)
    A_cum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))                                              # (b, h, c, q, q)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)   # within-chunk
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)                     # (b, h, c, q)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)  # state at each chunk end
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))       # (b, h, c+1, c+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(A_cum))
    return (Y_diag + Y_off).reshape(b, l, h, p)


def ssd_reference(X, A, B, C):
    """Step-by-step recurrence, for testing ssd_scan."""
    b, l, h, p = X.shape
    state = torch.zeros(b, h, p, B.shape[-1], dtype=X.dtype, device=X.device)
    ys = []
    for t in range(l):
        state = state * torch.exp(A[:, t])[:, :, None, None] + X[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
    return torch.stack(ys, dim=1)


class Mamba2Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.d_inner = c.expand * c.d_model
        self.h = self.d_inner // c.headdim
        conv_dim = self.d_inner + 2 * c.d_state
        self.in_proj = nn.Linear(c.d_model, 2 * self.d_inner + 2 * c.d_state + self.h, bias=False)
        self.conv = nn.Conv1d(conv_dim, conv_dim, c.d_conv, groups=conv_dim, padding=c.d_conv - 1)
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.h).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.h))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, c.d_model, bias=False)

    def forward(self, u):
        b, l, _ = u.shape
        c = self.c
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * c.d_state, self.h], dim=-1)
        xBC = F.silu(self.conv(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, c.d_state, c.d_state], dim=-1)
        pad = (-l) % c.chunk
        with torch.autocast(device_type=u.device.type, enabled=False):   # scan in float32
            dt = F.softplus(dt.float() + self.dt_bias)                  # (b, l, h)
            A = -torch.exp(self.A_log.float())
            xs = x.float().view(b, l, self.h, c.headdim)
            Bs = B.float()[:, :, None, :].expand(b, l, self.h, c.d_state)
            Cs = C.float()[:, :, None, :].expand(b, l, self.h, c.d_state)
            X, Ad = xs * dt[..., None], A * dt
            if pad:  # pad the sequence (dim 1) at the end; causal, so earlier outputs are unaffected
                X, Ad, Bs, Cs = (F.pad(t, [0, 0] * (t.dim() - 2) + [0, pad]) for t in (X, Ad, Bs, Cs))
            y = ssd_scan(X, Ad, Bs, Cs, c.chunk)[:, :l]
            y = y + xs * self.D[:, None]
        y = y.reshape(b, l, self.d_inner).type_as(u)
        return self.out_proj(self.norm(y * F.silu(z)))


class MambaLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.norm, self.mixer = RMSNorm(c.d_model), Mamba2Block(c)

    def forward(self, x, *_):
        return x + self.mixer(self.norm(x))


# ----------------------------------------------------------------------------- language model
class LM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        if c.arch == "transformer":
            self.layers = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
            cos, sin = rope_cache(c.max_seq_len, c.d_model // c.n_heads, "cpu")
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif c.arch == "mamba":
            self.layers = nn.ModuleList([MambaLayer(c) for _ in range(c.n_layers)])
            self.cos = self.sin = None
        else:
            raise ValueError(c.arch)
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled init for residual output projections
            if n.endswith(("out.weight", "down.weight", "out_proj.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, idx):
        """Final normalized hidden states (b, l, d): used for generation and classification."""
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        return self.norm(x)

    def logits_from_hidden(self, h):
        return F.linear(h, self.embed.weight)   # tied input/output embeddings

    def forward(self, idx, targets=None):
        logits = self.logits_from_hidden(self.hidden(idx))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), reduction="sum")
        return logits, loss

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb}

    def config_dict(self):
        return asdict(self.c)
