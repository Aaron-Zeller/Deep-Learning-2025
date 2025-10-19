import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention block. Supports masking.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        embed_bias: bool = False,
        q_dim: int = None,
        ctx_dim: int = None,
    ):
        """
        :param dim: Token dimensionality
        :param n_heads: Number of attention heads
        :param embed_bias: Whether to include bias terms in the linear projections
        :param q_dim: Dimensionality of the Query (if different from dim)
        :param ctx_dim: Dimensionality of the Context (if different from dim)
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

    def forward(
        self,
        q: Tensor,
        ctx: Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        :param q: (b, s, dq) Query
        :param ctx: (b, t, dc) Context (corresponds to Key and Value)
        :param mask: (b, s, t) Attention mask, convention is 0 for visible, 1 for masked
        :return: ((b, s, dq) Outputs, (b, h, s, t) Attention weights for each head)
        """

        # Compute embeddings
        q, k, v = map(lambda f, x: f(x), (self.query, self.key, self.value), (q, ctx, ctx))

        # Reshape into multi-head
        q, k, v = map(lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.n_heads), (q, k, v))

        # Compute (masked) attention
        scores = torch.einsum("b h s d, b h t d -> b h s t", q, k) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float("-inf"))
        attn = scores.softmax(dim=-1)

        # Compute values per head
        out = torch.einsum("b h s t, b h t d -> b h s d", attn, v)

        # Concat heads and project
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.output(out)

        return out, attn
