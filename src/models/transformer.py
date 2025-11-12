from typing import Optional

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.interfaces import PositionalEncodingBase, TransformerBase, TransformerEncoderBase


class Transformer(TransformerBase):
    def __init__(
        self,
        pos_encoding: PositionalEncodingBase,
        encoder: TransformerEncoderBase,
        vocab_size: int,
        dim: int,
    ):
        super().__init__()

        self.src_embed = nn.Embedding(vocab_size, dim)
        self.pos_encoding = pos_encoding
        self.encoder = encoder
        self.prober = nn.Linear(dim, vocab_size)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src_orig_size = src.shape

        # Flatten 2D grids to 1D sequences: (b, h, w) -> (b, h*w)
        src_seq = rearrange(src, "b h w -> b (h w)")

        # Embed tokens: (b, s) -> (b, s, d)
        src_emb = self.src_embed(src_seq)

        # Add positional encoding
        src_enc = self.pos_encoding(src_emb, src_orig_size)

        # Apply encoder
        memory, enc_attns = self.encoder(src_enc, mask=src_mask)

        # Project to vocabulary
        logits = self.prober(memory)
        probs = logits.softmax(dim=-1)

        return probs, logits, memory, enc_attns
