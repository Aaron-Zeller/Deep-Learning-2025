import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    """
    A Transformer Encoder block consisting of multiple Encoder Layers.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        src_dim: int = None,
        mha_embed_bias: bool = False,
        ff_expand_factor: int = 2,
        ff_gated: bool = True,
        ff_dropout: float = 0.0,
    ):
        """
        :param n_layers: Number of Encoder Layers
        :param dim: See `src.blocks.mha.MultiHeadAttention`
        :param n_heads: See `src.blocks.mha.MultiHeadAttention`
        :param src_dim: See `src.blocks.encoder_layer.EncoderLayer`
        :param mha_embed_bias: See `src.blocks.mha.MultiHeadAttention`
        :param ff_expand_factor: See `src.blocks.ff.FeedForward`
        :param ff_gated: See `src.blocks.ff.FeedForward`
        :param ff_dropout: See `src.blocks.ff.FeedForward`
        """

        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    dim=dim,
                    n_heads=n_heads,
                    mha_embed_bias=mha_embed_bias,
                    src_dim=src_dim,
                    ff_expand_factor=ff_expand_factor,
                    ff_gated=ff_gated,
                    ff_dropout=ff_dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        :param x: (b, s, d) Source sequence
        :return: ((b, s, d) Outputs, (num_layers, b, h, s, s) Self-Attention weights)
        """

        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)

        attns = torch.stack(attns, dim=0)

        return x, attns
