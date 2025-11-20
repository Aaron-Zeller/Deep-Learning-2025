from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.interfaces import AttentionBase, FeedForwardBase, TransformerDecoderLayerBase


class DecoderLayer(TransformerDecoderLayerBase):
    """Decoder layer with self-attention, cross-attention, and feed-forward network."""

    def __init__(
        self,
        self_attn: AttentionBase,
        cross_attn: AttentionBase,
        ff: FeedForwardBase,
        dim: int,
    ):
        """Initialize decoder layer.

        Args:
            self_attn: Self-attention module.
            cross_attn: Cross-attention module.
            ff: Feed-forward network module.
            dim: Model dimensionality.
        """
        super().__init__()

        self.self_attn = self_attn
        self.self_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = cross_attn
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_norm = nn.LayerNorm(dim)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual, self_attn_weights = self.self_attn(tgt, tgt, mask=tgt_mask)
        tgt = self.self_attn_norm(tgt + residual)

        residual, cross_attn_weights = self.cross_attn(tgt, memory, mask=memory_mask)
        tgt = self.cross_attn_norm(tgt + residual)

        residual = self.ff(tgt)
        tgt = self.ff_norm(tgt + residual)

        return tgt, self_attn_weights, cross_attn_weights
