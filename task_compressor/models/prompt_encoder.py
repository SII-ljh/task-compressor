"""Prompt Encoder for Task Compressor.

Compresses variable-length user prompts into a fixed number (n_p) of latent
vectors via a single cross-attention layer over frozen prompt embeddings.
"""

import torch
import torch.nn as nn

from .perceiver import MultiHeadCrossAttention


class PromptEncoder(nn.Module):
    """Encode user prompt into n_p fixed-length latent vectors.

    Uses frozen Qwen embedding layer to get prompt token embeddings, then
    n_p learnable latent tokens attend to them via cross-attention.
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int,
        num_heads: int,
        n_prompt_tokens: int = 16,
    ):
        super().__init__()
        self.embedding = embedding_layer  # shared & frozen
        self.n_prompt_tokens = n_prompt_tokens

        # Learnable latent query tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(n_prompt_tokens, hidden_size) * 0.02
        )

        # Single cross-attention layer with Pre-LN
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_ids:  (B, L_p) token ids
            prompt_mask: (B, L_p) attention mask
        Returns:
            (B, n_p, D) prompt latent vectors
        """
        # Frozen embedding — no gradient through embedding weights
        with torch.no_grad():
            prompt_embeds = self.embedding(prompt_ids)  # (B, L_p, D)

        B = prompt_ids.shape[0]
        latents = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_p, D)

        # Cross-attention with Pre-LN + residual
        q_normed = self.ln_q(latents)
        kv_normed = self.ln_kv(prompt_embeds)
        output = latents + self.cross_attn(q_normed, kv_normed, prompt_mask)

        return output  # (B, n_p, D)
