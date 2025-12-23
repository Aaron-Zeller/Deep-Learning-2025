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
        pre_ln: bool = False,
    ):
        """Initialize encoder layer.

        Args:
            mha: Multi-head attention module.
            ff: Feed-forward network module.
            dim: Model dimensionality.
            pre_ln: Whether to use pre-layer normalization.
        """
        super().__init__()

        self.mha = mha
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = ff
        self.ff_norm = nn.LayerNorm(dim)

        self.pre_ln = pre_ln

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        if self.pre_ln:
            # Pre-layer normalization
            src_norm = self.mha_norm(src)
            residual, attn = self.mha(src_norm, src_norm, mask=mask)
            src = src + residual

            src_norm = self.ff_norm(src)
            residual = self.ff(src_norm)
            src = src + residual
        else:
            # Post-layer normalization
            residual, attn = self.mha(src, src, mask=mask)
            src = self.mha_norm(src + residual)

            residual = self.ff(src)
            src = self.ff_norm(src + residual)

        return src, attn
