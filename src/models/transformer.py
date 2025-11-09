from typing import Optional

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.interfaces import PositionalEncodingBase, TransformerBase, TransformerDecoderBase, TransformerEncoderBase


class Transformer(TransformerBase):
    """Transformer with encoder-decoder architecture."""

    def __init__(
        self,
        pos_encoding: PositionalEncodingBase,
        encoder: TransformerEncoderBase,
        decoder: TransformerDecoderBase,
        vocab_size: int,
        dim: int,
    ):
        """Initialize transformer.

        Args:
            pos_encoding: Positional encoding module.
            encoder: Transformer encoder module.
            decoder: Transformer decoder module.
            vocab_size: Vocabulary size.
            dim: Model dimensionality.
        """
        super().__init__()

        self.src_embed = nn.Embedding(vocab_size, dim)
        self.tgt_embed = nn.Embedding(vocab_size, dim)
        self.pos_encoding = pos_encoding
        self.encoder = encoder
        self.decoder = decoder
        self.prober = nn.Linear(dim, vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        src_orig_size = src.shape
        tgt_orig_size = tgt.shape

        # Flatten 2D grids to 1D sequences: (b, h, w) -> (b, h*w)
        src_seq = rearrange(src, "b h w -> b (h w)")
        tgt_seq = rearrange(tgt, "b h w -> b (h w)")

        # Embed tokens: (b, s) -> (b, s, d)
        src_emb = self.src_embed(src_seq)
        tgt_emb = self.tgt_embed(tgt_seq)

        # Add positional encoding
        src_enc = self.pos_encoding(src_emb, src_orig_size)
        tgt_enc = self.pos_encoding(tgt_emb, tgt_orig_size)

        # Apply encoder-decoder
        memory, enc_attns = self.encoder(src_enc, mask=src_mask)
        tgt_dec, dec_self_attns, dec_cross_attns = self.decoder(
            tgt=tgt_enc, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        # Project to vocabulary
        logits = self.prober(tgt_dec)
        probs = logits.softmax(dim=-1)

        return probs, logits, memory, tgt_dec, enc_attns, dec_self_attns, dec_cross_attns
