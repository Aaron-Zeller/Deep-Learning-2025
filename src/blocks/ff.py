import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GEGLU(nn.Module):
    """
    A Gated Linear Unit with GELU activation.
    https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int):
        """
        :param dim: Input dimensionality
        """

        super().__init__()

        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    A Feed-Forward block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        expand_factor: int = 2,
        gated: bool = True,
        dropout: float = 0.0,
    ):
        """
        :param dim: Input dimensionality
        :param dim_out: Output dimensionality (if None, same as input dimensionality)
        :param expand_factor: Expanding factor for hidden dimension, based on input dimensionality
        :param gated: Whether to use gated projection (using GEGLU)
        :param dropout: Dropout rate
        """

        super().__init__()

        dim_out = dim_out or dim
        hidden_dim = dim * expand_factor

        proj = (
            GEGLU(dim)
            if gated
            else nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
            )
        )
        self.ff = nn.Sequential(
            proj,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)
