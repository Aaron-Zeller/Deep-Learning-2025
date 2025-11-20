import logging

import torch.nn as nn
from torch import Tensor
from torch.nn.functional import gelu

from src.interfaces import FeedForwardBase

logger = logging.getLogger(__name__)


class GatedLU(FeedForwardBase):
    """Gated Linear Unit

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int, dim_out: int = None, act_fn: nn.Module = None):
        """Initialize Gated Linear Unit module.

        Args:
            dim: Input dimensionality.
            dim_out: Output dimensionality (defaults to input dim).
            act_fn: Activation function to use (defaults to GELU).
        """
        super().__init__()

        dim_out = dim_out or dim

        self.proj = nn.Linear(dim, dim_out * 2)
        self.act_fn = act_fn or nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * self.act_fn(gate)


class FeedForward(FeedForwardBase):
    """Feed-forward network with optional gated projection."""

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        expand_factor: int = 2,
        act_fn: nn.Module = None,
        gated: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize feed-forward network.

        Args:
            dim: Input dimensionality.
            dim_out: Output dimensionality (defaults to input dim).
            expand_factor: Hidden dimension expansion factor.
            act_fn: Activation function to use (defaults to GELU).
            gated: Whether to use GEGLU gated projection.
            dropout: Dropout rate.
        """
        super().__init__()

        dim_out = dim_out or dim
        hidden_dim = dim * expand_factor
        act_fn = act_fn or nn.GELU()

        proj = (
            GatedLU(dim, hidden_dim, act_fn=act_fn)
            if gated
            else nn.Sequential(
                nn.Linear(dim, hidden_dim),
                act_fn,
            )
        )

        self.ff = nn.Sequential(
            proj,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)
