from torch import Size, Tensor

from src.interfaces import PositionalEncodingBase


class IdentityEncoding(PositionalEncodingBase):
    """Identity encoding"""

    def forward(self, x: Tensor, orig_size: Size) -> Tensor:
        return x
