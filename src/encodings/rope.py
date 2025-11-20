import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Size, Tensor

from src.interfaces import BaseRelativePositionalEncoding
from src.utils import get_forward_metadata


class RotaryPositionalEncoding(BaseRelativePositionalEncoding):
    """Rotary positional encoding (Su et al., 2021)"""

    def __init__(self, dim: int, n_heads: int):
        """Initialize rotary positional encoding.

        Args:
            dim: Token dimensionality.
            n_heads: Number of attention heads.
        """
        super().__init__()

        print(f"initialize with dim: {dim} and n_heads: {n_heads}")

        # We apply the rotary embedding to each token in every head separately.
        dim_head = dim // n_heads  # this better be even
        self.rope_h = RotaryEmbedding(dim=dim_head // 2)
        self.rope_w = RotaryEmbedding(dim=dim_head // 2)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # Fetch the grid size
        orig_shape = get_forward_metadata("orig_shape")
        _, h, w = orig_shape

        q = self._apply_2d_rope(q, h, w)
        k = self._apply_2d_rope(k, h, w)

        return (q, k)

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
        d_half = d_full // 2

        # Remove the head specific registers
        n_registers = s - (h * w)
        registers = x[..., :n_registers, :]
        x = x[..., n_registers:, :]

        x = rearrange(x, "b h (gh gw) d -> b h gh gw d", gh=h, gw=w)

        # Split x in both dimensions
        x_h = x[..., :d_half]
        x_w = x[..., d_half:]

        # Apply rope to height
        x_h = rearrange(x_h, "b h gh gw d -> b (h gw) gh d")
        x_h = self.rope_h.rotate_queries_or_keys(x_h)
        x_h = rearrange(x_h, "b (h gw) gh d -> b h (gh gw) d", h=n_heads, gw=w)

        # Apply rope to width
        x_w = rearrange(x_w, "b h gh gw d -> b (h gh) gw d")
        x_w = self.rope_h.rotate_queries_or_keys(x_w)
        x_w = rearrange(x_w, "b (h gh) gw d -> b h (gh gw) d", h=n_heads, gh=h)

        # Concatenate back together
        x = torch.cat([x_h, x_w], dim=-1)

        # Add registers back
        x = torch.cat([registers, x], dim=2)
        return x
