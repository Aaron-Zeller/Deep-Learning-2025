import torch
import torch.nn as nn
from torch import Tensor

from src.blocks.encoder import TransformerEncoder
from src.blocks.decoder import TransformerDecoder


class VanillaTransformer(nn.Module):
    """
    A vanilla Transformer model.
    """

    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        n_tokens: int,
        src_dim: int = None,
        tgt_dim: int = None,
        mha_embed_bias: bool = False,
        ff_expand_factor: int = 2,
        ff_gated: bool = True,
        ff_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            src_dim=src_dim,
            mha_embed_bias=mha_embed_bias,
            ff_expand_factor=ff_expand_factor,
            ff_gated=ff_gated,
            ff_dropout=ff_dropout,
        )

        self.decoder = TransformerDecoder(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            src_dim=src_dim,
            tgt_dim=tgt_dim,
            mha_embed_bias=mha_embed_bias,
            ff_expand_factor=ff_expand_factor,
            ff_gated=ff_gated,
            ff_dropout=ff_dropout,
        )

        self.prober = nn.Linear(dim, n_tokens)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        :param src: (b, s, ds) Source sequence
        :param tgt: (b, t, dt) Target sequence
        :param mask: (b, t, t) Target mask
        :return: ((b, t, n_tokens) Probabilities,
                  (b, t, n_tokens) Logits,
                  (b, s, ds) Encoder outputs, (b, t, dt) Decoder outputs,
                  (num_layers, b, h, s, s) Encoder Self-Attention weights,
                  (num_layers, b, h, t, t) Decoder Self-Attention weights,
                  (num_layers, b, h, t, s) Decoder Cross-Attention weights)
        """
        src_enc, enc_attns = self.encoder(src)
        tgt_dec, dec_self_attns, dec_cross_attns = self.decoder(src_enc, tgt, mask=mask)

        logits = self.prober(tgt_dec)
        probs = logits.softmax(dim=-1)

        return probs, logits, src_enc, tgt_dec, enc_attns, dec_self_attns, dec_cross_attns
