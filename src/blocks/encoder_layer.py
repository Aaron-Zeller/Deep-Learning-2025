from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.interfaces import AttentionBase, FeedForwardBase, TransformerEncoderLayerBase


class EncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer with self-attention and feed-forward network."""

    def __init__(
        self,
        mha: AttentionBase,
        ff: FeedForwardBase,
        dim: int,
    ):
        """Initialize encoder layer.

        Args:
            mha: Multi-head attention module.
            ff: Feed-forward network module.
            dim: Model dimensionality.
        """
        super().__init__()

        self.mha = mha
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        residual, attn = self.mha(src, src, mask=mask)
        src = self.mha_norm(src + residual)

        residual = self.ff(src)
        src = self.ff_norm(src + residual)

        return src, attn
