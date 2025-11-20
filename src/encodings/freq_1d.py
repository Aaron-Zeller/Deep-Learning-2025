import math

import torch
from torch import Size, Tensor

from src.interfaces import PositionalEncodingBase


class Frequency1DEncoding(PositionalEncodingBase):
    """Sinusoidal 1D positional encoding (Vaswani et al., 2017)."""

    def __init__(self, dim: int, max_len: int = 5000):
        """Initialize frequency-based positional encoding.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length to precompute.
        """
        super().__init__()

        # Precompute positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, orig_size: Size) -> Tensor:
        seq_len = x.size(1)

        x = x + self.pe[:seq_len, :].unsqueeze(0)

        return x
