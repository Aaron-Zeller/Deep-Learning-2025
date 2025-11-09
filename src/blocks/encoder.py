from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from src.interfaces import TransformerEncoderBase, TransformerEncoderLayerBase


class TransformerEncoder(TransformerEncoderBase):
    """Transformer encoder stacking multiple encoder layers."""

    def __init__(self, n_layers: int, layer: TransformerEncoderLayerBase):
        """Initialize transformer encoder.

        Args:
            n_layers: Number of encoder layers.
            layer: Specifies the encoder layer configuration. Note: this module will be copied n_layers times.
        """
        super().__init__()

        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(n_layers)])

    def forward(self, src: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        attns = []

        layer: TransformerEncoderLayerBase
        for layer in self.layers:
            src, attn = layer(src, mask=mask)
            attns.append(attn)

        attns = torch.stack(attns, dim=0)

        return src, attns
