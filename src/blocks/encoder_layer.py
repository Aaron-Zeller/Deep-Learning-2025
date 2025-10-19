import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.mha import MultiHeadAttention
from src.blocks.ff import FeedForward


class EncoderLayer(nn.Module):
    """
    An Encoder Layer block consisting of Multi-Head Self-Attention and a Feed-Forward network.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        src_dim: int = None,
        mha_embed_bias: bool = False,
        ff_expand_factor: int = 2,
        ff_gated: bool = True,
        ff_dropout: float = 0.0,
    ):
        """
        :param dim: See `src.blocks.mha.MultiHeadAttention`
        :param n_heads: See `src.blocks.mha.MultiHeadAttention`
        :param src_dim: Dimensionality of the Source sequence (if different from dim)
        :param mha_embed_bias: See `src.blocks.mha.MultiHeadAttention`
        :param ff_expand_factor: See `src.blocks.ff.FeedForward`
        :param ff_gated: See `src.blocks.ff.FeedForward`
        :param ff_dropout: See `src.blocks.ff.FeedForward`
        """

        super().__init__()

        src_dim = src_dim or dim

        self.mha = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            embed_bias=mha_embed_bias,
            q_dim=src_dim,
            ctx_dim=src_dim,
        )
        self.mha_norm = nn.LayerNorm(dim)

        self.ff = FeedForward(
            dim=dim,
            expand_factor=ff_expand_factor,
            gated=ff_gated,
            dropout=ff_dropout,
        )
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        :param x: (b, s, d) Source sequence
        :return: ((b, s, d) Outputs, (b, h, s, s) Self-Attention weights)
        """

        dx, attn = self.mha(x, x)
        x = self.mha_norm(x + dx)
        x = self.ff_norm(x + self.ff(x))

        return x, attn
