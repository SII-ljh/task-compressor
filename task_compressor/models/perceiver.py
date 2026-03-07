"""Query-Conditioned Perceiver for Task Compressor.

Standard Perceiver architecture: stacked blocks of
cross-attention -> self-attention -> FFN, all with Pre-LN + residual.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: query attends to key-value."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query:    (B, Nq, D)
            key_value: (B, Nkv, D)
            kv_mask:  (B, Nkv) — 1 for real tokens, 0 for padding
        Returns:
            (B, Nq, D)
        """
        B, Nq, D = query.shape
        _, Nkv, _ = key_value.shape

        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch 2.0 scaled_dot_product_attention when possible
        if kv_mask is not None:
            # (B, Nkv) -> (B, 1, 1, Nkv) for broadcasting
            attn_mask = kv_mask[:, None, None, :].bool()
            # SDPA expects True = attend, which matches our mask convention
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.o_proj(out)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class PerceiverBlock(nn.Module):
    """Single Perceiver block: cross-attn -> self-attn -> FFN, all Pre-LN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        # Cross-attention
        self.cross_attn_ln_q = nn.LayerNorm(hidden_size)
        self.cross_attn_ln_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        # Self-attention
        self.self_attn_ln = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads)

        # FFN
        self.ffn_ln = nn.LayerNorm(hidden_size)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size)
        self.ffn_act = nn.GELU()

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Cross-attention with Pre-LN + residual
        q_normed = self.cross_attn_ln_q(query)
        kv_normed = self.cross_attn_ln_kv(key_value)
        query = query + self.cross_attn(q_normed, kv_normed, kv_mask)

        # Self-attention with Pre-LN + residual
        query = query + self.self_attn(self.self_attn_ln(query))

        # FFN with Pre-LN + residual
        h = self.ffn_ln(query)
        query = query + self.ffn_down(self.ffn_act(self.ffn_up(h)))

        return query


class QueryConditionedPerceiver(nn.Module):
    """Stack of Perceiver blocks with final layer norm.

    Query = [prompt_latents (n_p), context_tokens (n_c)], total k tokens.
    KV = encoder hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PerceiverBlock(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, k, D) — concatenation of prompt latents + context tokens
            key_value: (B, L, D) — encoder hidden states
            kv_mask:   (B, L) — encoder attention mask
        Returns:
            (B, k, D) — compressed representation (soft prompts)
        """
        for layer in self.layers:
            query = layer(query, key_value, kv_mask)
        return self.final_ln(query)
