import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.interfaces import AttentionBase, RelativePositionalEncodingBase

logger = logging.getLogger(__name__)


class MultiHeadAttention(AttentionBase):
    """Multi-head attention with optional masking."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        embed_bias: bool = True,
        q_dim: int = None,
        ctx_dim: int = None,
        dropout: float = 0.0,
        relative_pos_encoding: Optional[RelativePositionalEncodingBase] = None,
    ):
        """Initialize multi-head attention.

        Args:
            dim: Token dimensionality.
            n_heads: Number of attention heads.
            embed_bias: Whether to use bias in linear projections.
            q_dim: Query dimensionality (defaults to dim).
            ctx_dim: Context dimensionality (defaults to dim).
            dropout: Dropout rate.
            relative_pos_encoding: Relative positional encoding (defaults to None)
        """
        super().__init__()

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.n_heads = n_heads
        q_dim = q_dim or dim
        ctx_dim = ctx_dim or dim
        dim_head = dim // n_heads
        self.scale = dim_head**-0.5

        self.query = torch.nn.Linear(q_dim, dim, bias=embed_bias)
        self.key = torch.nn.Linear(ctx_dim, dim, bias=embed_bias)
        self.value = torch.nn.Linear(ctx_dim, dim, bias=embed_bias)
        self.output = torch.nn.Linear(dim, q_dim, bias=embed_bias)

        self.dropout = nn.Dropout(dropout)
        self.relative_pos_encoding = relative_pos_encoding

    def forward(
        self,
        q: Tensor,
        ctx: Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # Apply projection
        q, k, v = map(lambda f, x: f(x), (self.query, self.key, self.value), (q, ctx, ctx))

        # Split into 'n_heads' heads
        q, k, v = map(lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.n_heads), (q, k, v))

        # Apply relative positional encoding if possible
        if self.relative_pos_encoding is not None:
            q, k = self.relative_pos_encoding(q, k)

        # Compute q @ k^T / sqrt(d)
        scores = torch.einsum("b h s d, b h t d -> b h s t", q, k) * self.scale

        if mask is not None:
            # Expand mask to match scores shape: (b, s, t) -> (b, 1, s, t)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, float("-inf"))

        # Apparently all we need
        attn = scores.softmax(dim=-1)

        # Softmax(q @ k^T / sqrt(d)) @ v
        out = torch.einsum("b h s t, b h t d -> b h s d", attn, v)

        # Combine heads
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.output(out)

        out = self.dropout(out)

        return out, attn
