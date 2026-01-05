import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from einops import rearrange, repeat

from src.interfaces import (
    DatasetBase,
    PositionalEncodingBase,
    TransformerBase,
    TransformerDecoderBase,
    TransformerEncoderBase,
    TransformerHeadBase,
)

logger = logging.getLogger(__name__)


class Transformer(TransformerBase):
    """Transformer with encoder-decoder architecture."""

    def __init__(
        self,
        head: TransformerHeadBase,
        pos_encoding: PositionalEncodingBase,
        encoder: Optional[TransformerEncoderBase],
        decoder: Optional[TransformerDecoderBase],
        dataset: DatasetBase,
        dim: int,
        n_action_tokens: int = 0,
    ):
        """Initialize transformer.

        Args:
            head: Transformer head module.
            pos_encoding: Positional encoding module.
            encoder: Transformer encoder module.
            decoder: Transformer decoder module.
            dataset: Dataset instance.
            dim: Model dimensionality.
        """
        super().__init__()

        vocab_size = dataset.vocab_size()
        self.src_embed = nn.Embedding(vocab_size, dim)
        self.tgt_embed = nn.Embedding(vocab_size, dim)
        self.pos_encoding = pos_encoding
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

        if encoder is not None and decoder is None:
            logger.info("Transformer is Encoder-Only.")
        elif encoder is None and decoder is not None:
            logger.info("Transformer is Decoder-Only.")
        else:
            logger.info("Transformer is Encoder-Decoder.")

        self._dim = dim

        self.n_action_tokens = n_action_tokens
        self.action_tokens = nn.Parameter(torch.randn(1, n_action_tokens, dim)) if n_action_tokens > 0 else None

    def dim(self) -> int:
        """Get model dimensionality.

        Returns:
            Model dimensionality.
        """
        return self._dim

    def prepare_tokens(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> tuple[Tensor, Tensor]:
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

        return src_enc, tgt_enc

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[dict]
    ]:

        src, tgt = self.head.inject(src, tgt, self.pos_encoding, src.shape, tgt.shape)

        # Add the action tokens if any
        if self.action_tokens is not None:
            b = src.shape[0]
            action_tokens = repeat(self.action_tokens, "1 n d -> b n d", b=b)
            src = torch.cat([action_tokens, src], dim=1)

        # Apply encoder-decoder
        memory, enc_attns = self.encoder(src, mask=src_mask)

        if self.decoder is None:  # Encoder-Only, todo: Decoder-Only
            return memory, None, enc_attns, None, None, {}

        tgt_dec, dec_self_attns, dec_cross_attns = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )

        return memory, tgt_dec, enc_attns, dec_self_attns, dec_cross_attns, {}
