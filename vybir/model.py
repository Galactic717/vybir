"""Torch-native Laya decision model (Windows/Linux/macOS, CPU/CUDA).

Mirrors laya-mlx/model.py layer-for-layer, but in PyTorch so it runs where
MLX cannot (MLX is Apple-silicon-only). Parameter names match the published
upstream checkpoint (convaiinnovations/laya/model.safetensors) exactly, so
weights load with strict=True and no conversion step is needed:

  encoder.embeddings.tok_embeddings / norm
  encoder.layers.i.attn.Wqkv / Wo (+ attn_norm except layer 0)
  encoder.layers.i.mlp.Wi / Wo + mlp_norm
  encoder.final_norm, type_emb, head.layers.*, scorer.*, act_head.*
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- encoder config

class EncoderConfig:
    def __init__(self, cfg: dict):
        if cfg.get("model_type", "modernbert") != "modernbert":
            raise ValueError(f"Unsupported encoder: {cfg.get('model_type')!r}; expected modernbert")
        if cfg.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("Unsupported encoder activation")
        self.vocab_size = cfg["vocab_size"]
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"]
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.norm_eps = cfg.get("norm_eps", 1e-5)
        self.norm_bias = cfg.get("norm_bias", False)
        self.attention_bias = cfg.get("attention_bias", False)
        self.mlp_bias = cfg.get("mlp_bias", False)
        self.local_attention = cfg.get("local_attention", 128)
        self.global_attn_every_n_layers = cfg.get("global_attn_every_n_layers", 3)
        self.global_rope_theta = cfg.get("global_rope_theta", 160000.0)
        self.local_rope_theta = cfg.get("local_rope_theta", 10000.0)
        self.max_position_embeddings = cfg.get("max_position_embeddings", 8192)
        lt = cfg.get("layer_types")
        if lt is None:
            lt = [
                "full_attention" if i % self.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(lt) != self.num_hidden_layers or set(lt) - {"full_attention", "sliding_attention"}:
            raise ValueError("Invalid ModernBERT layer_types")
        self.layer_types = lt
        self.rope_parameters = cfg.get("rope_parameters") or {}
        for kind in set(lt):
            params = self.rope_parameters.get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ValueError("Only default (unscaled) ModernBERT RoPE is supported")
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("ModernBERT requires an even, integral attention head dimension")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind):
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float(self.rope_parameters.get(kind, {}).get("rope_theta", fallback))


# ---------------------------------------------------------------- RoPE (NeoX style, matches mlx.fast.rope traditional=False)

_rope_cache: dict = {}


def _rope_cos_sin(seq_len: int, head_dim: int, base: float, device, dtype):
    # Cache only concrete ints: under torch.export seq_len is a SymInt
    # (unhashable) and each layer must compute its own constants.
    if isinstance(seq_len, int):
        key = (seq_len, head_dim, float(base), str(device), str(dtype))
        hit = _rope_cache.get(key)
        if hit is not None:
            return hit
        out = _rope_compute(seq_len, head_dim, base, device, dtype)
        if len(_rope_cache) > 32:  # bound memory: lengths vary per batch
            _rope_cache.clear()
        _rope_cache[key] = out
        return out
    return _rope_compute(seq_len, head_dim, base, device, dtype)


def _rope_compute(seq_len, head_dim, base, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)  # (L, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
    return (emb.cos().to(dtype), emb.sin().to(dtype))


def _apply_rope(x, cos, sin):
    # x: (B, H, L, D) NeoX rotation: split halves
    d2 = x.size(-1) // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    rot = torch.cat([-x2, x1], dim=-1)
    # cos/sin: (L, D) -> (1, 1, L, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rot * sin


# ---------------------------------------------------------------- layers (names match checkpoint)

class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, ids):
        return self.norm(self.tok_embeddings(ids))


class EncoderAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig, kind: str, impl: str = "sdpa"):
        super().__init__()
        if impl not in ("sdpa", "math"):
            raise ValueError("attn impl must be 'sdpa' or 'math'")
        self.impl = impl
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.base = cfg.rope_base(kind)
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x, mask):
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3)]
        cos, sin = _rope_cos_sin(length, self.head_dim, self.base, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim ** -0.5)
        else:
            # Export-friendly: same math, no SDPA op (ORT has no SDPA kernel here).
            scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            additive = torch.zeros_like(scores)
            additive = additive.masked_fill(~mask, -1e4)
            out = torch.softmax(scores + additive, dim=-1) @ v
        return self.Wo(out.permute(0, 2, 1, 3).reshape(b, length, -1))


class EncoderMLP(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x):
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, index: int, impl: str = "sdpa"):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type, impl)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def forward(self, x, masks):
        x = x + self.attn(self.attn_norm(x), masks[self.attention_type])
        return x + self.mlp(self.mlp_norm(x))


def attention_masks(attention_mask: torch.Tensor, window: int):
    """Boolean key masks. Padded queries see valid keys (avoids all-masked rows)."""
    valid = attention_mask.bool()  # (B, L)
    full = valid[:, None, None, :]  # (B,1,1,L)
    L = valid.shape[1]
    pos = torch.arange(L, device=valid.device)
    local_2d = (pos[:, None] - pos[None, :]).abs() <= window // 2  # (L,L)
    # padded query rows fall back to full mask:
    local = (local_2d[None, None] | ~valid[:, None, :, None]) & full
    return {"full_attention": full, "sliding_attention": local}


class ModernBertTorch(nn.Module):
    def __init__(self, cfg: EncoderConfig, impl: str = "sdpa"):
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = nn.ModuleList([EncoderLayer(cfg, i, impl) for i in range(cfg.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            x = layer(x, masks)
        return self.final_norm(x)


class DecisionModelTorch(nn.Module):
    """Full Laya head. Forward mirrors upstream torch + laya-mlx numerics."""

    def __init__(self, encoder_cfg: EncoderConfig, agent_cfg: dict, attn_impl: str = "sdpa"):
        super().__init__()
        dims = encoder_cfg.hidden_size
        self.encoder = ModernBertTorch(encoder_cfg, impl=attn_impl)
        nhead = max(1, dims // 64)
        layer = nn.TransformerEncoderLayer(
            dims, nhead, 4 * dims, dropout=0.0, batch_first=True, norm_first=True
        )
        hl = int(agent_cfg.get("head_layers", 2))
        self.head = (
            nn.TransformerEncoder(layer, hl, enable_nested_tensor=False) if hl > 0 else None
        )
        self.type_emb = nn.Embedding(3, dims)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dims), nn.Linear(dims, dims), nn.GELU(), nn.Linear(dims, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(dims + 4, 256),
            nn.GELU(),
            nn.Linear(256, len(agent_cfg.get("act_costs", {})) + 1),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                return_hidden: bool = False):
        """Logits per option marker and act logits.

        ``return_hidden=True`` also returns the per-option marker states (B, K, D) and the
        [CLS] state (B, D) that the heads read: features for few-shot adapters.
        """
        h = self.encoder(input_ids, attention_mask)
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        with torch.no_grad():
            p = torch.softmax(logits.detach(), -1)
            k = marker_mask.sum(-1).clamp(min=2).float()
            ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
            if p.size(-1) >= 2:
                top2 = p.topk(2, -1).values
            else:
                top1 = p.topk(1, -1).values
                top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
            feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
            # Features are float32; cast to the head's dtype like laya-mlx does, so
            # float16/bfloat16 weights (CUDA) do not hit a Float-vs-Half matmul error.
            pooled = torch.cat([h[:, 0].float(), feats], -1)
            act_logits = self.act_head(pooled.to(self.act_head[0].weight.dtype))
        if return_hidden:
            return logits, act_logits.float(), m.float(), h[:, 0].float()
        return logits, act_logits.float()


def remap_mlx_to_torch(state: dict) -> dict:
    """Accept pre-converted MLX checkpoints (aac6fef/*-mlx) in the torch loader."""
    out = {}
    for name, value in state.items():
        n = name.replace(".in_proj.weight", ".in_proj_weight").replace(
            ".in_proj.bias", ".in_proj_bias"
        )
        for prefix in ("scorer", "act_head"):
            tag = prefix + ".layers."
            if n.startswith(tag):
                n = prefix + "." + n[len(tag):]
        out[n] = value
    return out
