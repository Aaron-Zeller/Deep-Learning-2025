import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.mha import MultiHeadAttention
from src.blocks.ff import FeedForward


class DecoderLayer(nn.Module):
    """
    A Decoder Layer block consisting of Multi-Head Self-/Cross-Attention and a Feed-Forward network.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        src_dim: int = None,
        tgt_dim: int = None,
        mha_embed_bias: bool = False,
        ff_expand_factor: int = 2,
        ff_gated: bool = True,
        ff_dropout: float = 0.0,
    ):
        """
        :param dim: See `src.blocks.mha.MultiHeadAttention`
        :param n_heads: See `src.blocks.mha.MultiHeadAttention`
        :param src_dim: Dimensionality of the Source sequence (if different from dim)
        :param tgt_dim: Dimensionality of the Target sequence (if different from dim)
        :param mha_embed_bias: See `src.blocks.mha.MultiHeadAttention`
        :param ff_expand_factor: See `src.blocks.ff.FeedForward`
        :param ff_gated: See `src.blocks.ff.FeedForward`
        :param ff_dropout: See `src.blocks.ff.FeedForward`
        """

        super().__init__()

        src_dim = src_dim or dim
        tgt_dim = tgt_dim or dim

        self.self_mha = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            embed_bias=mha_embed_bias,
            q_dim=tgt_dim,
            ctx_dim=tgt_dim,
        )
        self.self_mha_norm = nn.LayerNorm(dim)

        self.cross_mha = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            embed_bias=mha_embed_bias,
            q_dim=tgt_dim,
            ctx_dim=src_dim,
        )
        self.cross_mha_norm = nn.LayerNorm(dim)

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
        y: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param x: (b, s, ds) Source sequence (Context in cross-attention)
        :param y: (b, t, dt) Target sequence
        :param mask: (b, t, t) Self-Attention mask, convention is 0 for visible, 1 for masked
        :return: ((b, t, dt) Outputs, (b, h, t, t) Self-Attention weights, (b, h, t, s) Cross-Attention weights)
        """

        dy, self_attn = self.self_mha(y, y, mask=mask)
        y = self.self_mha_norm(y + dy)

        dy, cross_attn = self.cross_mha(y, x)
        y = self.cross_mha_norm(y + dy)

        y = self.ff_norm(y + self.ff(y))

        return y, self_attn, cross_attn
