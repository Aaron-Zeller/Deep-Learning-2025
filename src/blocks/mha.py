import logging

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor

from src.interfaces import AttentionBase
from src.utils import get_forward_metadata

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
    ):
        """Initialize multi-head attention.

        Args:
            dim: Token dimensionality.
            n_heads: Number of attention heads.
            embed_bias: Whether to use bias in linear projections.
            q_dim: Query dimensionality (defaults to dim).
            ctx_dim: Context dimensionality (defaults to dim).
            dropout: Dropout rate.
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
        self.rope_h = RotaryEmbedding(dim=dim_head // 2)
        self.rope_w = RotaryEmbedding(dim=dim_head // 2)

    def forward(
        self,
        q: Tensor,
        ctx: Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        orig_shape = get_forward_metadata("orig_shape")
        _, h, w = orig_shape

        # Apply projection
        q, k, v = map(lambda f, x: f(x), (self.query, self.key, self.value), (q, ctx, ctx))

        # Split into 'n_heads' heads
        q, k, v = map(lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.n_heads), (q, k, v))

        # Apply 2D rotational encoding
        q = self._apply_2d_rope(q, h, w)
        k = self._apply_2d_rope(k, h, w)

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

    def _apply_2d_rope(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply 2D RoPE encoding to x. The features of x are being splitted into
        x_h and x_w, and on those two different rotational encoding are being applied.
        After the encoding, x_h and x_w are again merged into the correct shape.

        Args:
            x: (b h s d) input tokens (without registers)
            h: height of the grid
            w: width of the grid
        """
        b, n_heads, s, d_full = x.shape
        d_half = d_full // 2

        # Remove the head specific registers
        n_registers = s - (h * w)
        registers = x[..., :n_registers, :]
        x = x[..., n_registers:, :]

        x = rearrange(x, "b h (gh gw) d -> b h gh gw d", gh=h, gw=w)

        # Split x in both dimensions
        x_h = x[..., :d_half]
        x_w = x[..., d_half:]

        # Apply rope to height
        x_h = rearrange(x_h, "b h gh gw d -> b (h gw) gh d")
        x_h = self.rope_h.rotate_queries_or_keys(x_h)
        x_h = rearrange(x_h, "b (h gw) gh d -> b h (gh gw) d", h=n_heads, gw=w)

        # Apply rope to width
        x_w = rearrange(x_w, "b h gh gw d -> b (h gh) gw d")
        x_w = self.rope_h.rotate_queries_or_keys(x_w)
        x_w = rearrange(x_w, "b (h gh) gw d -> b h (gh gw) d", h=n_heads, gh=h)

        # Concatenate back together
        x = torch.cat([x_h, x_w], dim=-1)

        # Add registers back
        x = torch.cat([registers, x], dim=2)
        return x
