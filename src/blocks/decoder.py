from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.interfaces import TransformerDecoderBase, TransformerDecoderLayerBase


class TransformerDecoder(TransformerDecoderBase):
    """Transformer decoder stacking multiple decoder layers."""

    def __init__(self, n_layers: int, layer: TransformerDecoderLayerBase):
        """Initialize transformer decoder.

        Args:
            n_layers: Number of decoder layers.
            layer: Specifies the decoder layer configuration. Note: this module will be copied n_layers times.
        """
        super().__init__()

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self_attns = []
        cross_attns = []

        layer: TransformerDecoderLayerBase
        for layer in self.layers:
            tgt, self_attn, cross_attn = layer(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)

        self_attns = torch.stack(self_attns, dim=0)
        cross_attns = torch.stack(cross_attns, dim=0)

        return tgt, self_attns, cross_attns
