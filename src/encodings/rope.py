from typing import Literal

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Size, Tensor

from src.interfaces import RelativePositionalEncodingBase
from src.utils import get_forward_metadata


class RotaryPositionalEncoding(RelativePositionalEncodingBase):
    """Rotary positional encoding (Su et al., 2021)"""

    def __init__(self, dim: int, theta: float = 10_000, mode: Literal["1d", "2d"] = "2d"):
        """Initialize rotary positional encoding.

        Args:
            dim: Token dimensionality.
            n_heads: Number of attention heads.
            theta: Base frequency for rotary embeddings.
        """
        super().__init__()

        # The user is responsible for ensuring that dim is correctly set
        self.dim = dim
        self.rope = RotaryEmbedding(dim=dim, theta=theta)
        self.mode = mode

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # Fetch the grid size
        orig_shape = get_forward_metadata("orig_shape")
        _, h, w = orig_shape

        if self.mode == "1d":
            q = self._apply_1d_rope(q, h, w)
            k = self._apply_2d_rope(k, h, w)
        elif self.mode == "2d":
            q = self._apply_2d_rope(q, h, w)
            k = self._apply_2d_rope(k, h, w)

        return q, k

    def _apply_1d_rope(self, x: Tensor, h: int, w: int) -> Tensor:
        """Apply 1D RoPE encoding to x. The features of x are being treated as a
        sequence of length h*w, and the rotational encoding is being applied
        on that sequence.

        Args:
            x: (b h s d) input tokens (with potential registers)
            h: height of the grid
            w: width of the grid
        """
        b, n_heads, s, d_full = x.shape

        # Remove the head specific registers
        n_registers = s - (h * w)
        registers = x[..., :n_registers, :]
        x = x[..., n_registers:, :]

        # Not all features are used for rope
        x_rope = x[..., : 2 * self.dim]
        x_keep = x[..., 2 * self.dim :]

        # Apply rope to the sequence
        x_rope = rearrange(x_rope, "b h (gh gw) d -> b (h gh gw) d")
        x_rope = self.rope.rotate_queries_or_keys(x_rope)
        x_rope = rearrange(x_rope, "b (h gh gw) d -> b h (gh gw) d", h=n_heads, gh=h, gw=w)

        # Concatenate back together
        x = torch.cat([x_rope, x_keep], dim=-1)

        # Add registers back
        x = torch.cat([registers, x], dim=2)
        return x

    def _apply_2d_rope(self, x: Tensor, h: int, w: int) -> Tensor:
        """Apply 2D RoPE encoding to x. The features of x are being splitted into
        x_h and x_w, and on those two different rotational encoding are being applied.
        After the encoding, x_h and x_w are again merged into the correct shape.

        Args:
            x: (b h s d) input tokens (with potential registers)
            h: height of the grid
            w: width of the grid
        """
        b, n_heads, s, d_full = x.shape

        # Remove the head specific registers
        n_registers = s - (h * w)
        registers = x[..., :n_registers, :]
        x = x[..., n_registers:, :]

        # Not all features are used for rope
        x_rope = x[..., : 2 * self.dim]
        x_keep = x[..., 2 * self.dim :]

        x_rope = rearrange(x_rope, "b h (gh gw) d -> b h gh gw d", gh=h, gw=w)

        # Split x in both dimensions
        x_h = x_rope[..., : self.dim]
        x_w = x_rope[..., self.dim : 2 * self.dim]

        # Apply rope to height
        x_h = rearrange(x_h, "b h gh gw d -> b (h gw) gh d")
        x_h = self.rope.rotate_queries_or_keys(x_h)
        x_h = rearrange(x_h, "b (h gw) gh d -> b h (gh gw) d", h=n_heads, gw=w)

        # Apply rope to width
        x_w = rearrange(x_w, "b h gh gw d -> b (h gh) gw d")
        x_w = self.rope.rotate_queries_or_keys(x_w)
        x_w = rearrange(x_w, "b (h gh) gw d -> b h (gh gw) d", h=n_heads, gh=h)

        # Concatenate back together
        x = torch.cat([x_h, x_w, x_keep], dim=-1)

        # Add registers back
        x = torch.cat([registers, x], dim=2)
        return x
