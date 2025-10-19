import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    """
    A Transformer Decoder block consisting of multiple Decoder Layers.
    """

    def __init__(
        self,
        n_layers: int,
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
        :param n_layers: Number of Encoder Layers
        :param dim: See `src.blocks.mha.MultiHeadAttention`
        :param n_heads: See `src.blocks.mha.MultiHeadAttention`
        :param src_dim: See `src.blocks.encoder_layer.DecoderLayer`
        :param tgt_dim: See `src.blocks.encoder_layer.DecoderLayer`
        :param mha_embed_bias: See `src.blocks.mha.MultiHeadAttention`
        :param ff_expand_factor: See `src.blocks.ff.FeedForward`
        :param ff_gated: See `src.blocks.ff.FeedForward`
        :param ff_dropout: See `src.blocks.ff.FeedForward`
        """

        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=dim,
                    n_heads=n_heads,
                    mha_embed_bias=mha_embed_bias,
                    src_dim=src_dim,
                    tgt_dim=tgt_dim,
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
        y: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param x: (b, s, ds) Source sequence (Context in cross-attention)
        :param y: (b, t, dt) Target sequence
        :param mask: (b, t, t) Self-Attention mask, convention is 0 for visible, 1 for masked
        :return: ((b, s, d) Outputs, (num_layers, b, h, t, t) Self-Attention weights, (num_layers, b, h, t, s) Cross-Attention weights)
        """

        self_attns = []
        cross_attns = []
        for layer in self.layers:
            y, self_attn, cross_attn = layer(x, y, mask=mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)

        self_attns = torch.stack(self_attns, dim=0)
        cross_attns = torch.stack(cross_attns, dim=0)

        return y, self_attns, cross_attns
