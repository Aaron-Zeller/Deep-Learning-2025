import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Size, Tensor

from src.interfaces import DatasetBase, PositionalEncodingBase, TransformerBase, TransformerHeadBase


class GlobalActionDistributionHead(TransformerHeadBase):
    """Transformer head for global action distribution prediction."""

    def __init__(
        self,
        n_registers: int,
        dim: int,
        dataset: DatasetBase,
    ):
        """Initialize global action distribution head.

        Args:
            n_registers: Number of register tokens. This helps avoiding contamination of the data tokens.
            dim: Model dimensionality.
            dataset: Dataset instance.
        """
        super().__init__()

        self.n_registers = n_registers
        self.dim = dim
        self.h, self.w = dataset.grid_size()
        self.vocab_size = dataset.vocab_size()

        _registers = torch.randn(self.n_registers, self.dim)
        self.registers = nn.Parameter(_registers)

        self.proj = nn.Conv1d(self.dim, self.vocab_size, kernel_size=1)

    def inject(
        self,
        src: Tensor,
        tgt: Tensor,
        pos_encoding: PositionalEncodingBase,
        src_orig_size: Size,
        tgt_orig_size: Size,
    ) -> tuple[Tensor, Tensor]:
        b = src.shape[0]

        registers = repeat(self.registers[None, ...], "1 n d -> b n d", b=b)
        src = torch.cat([registers, src], dim=1)

        return src, tgt

    def step(self, x_in: Tensor, y_pred: Tensor) -> Tensor:
        b, h, w = x_in.shape

        y_pred_flat = rearrange(y_pred, "b s v -> b (s v)")

        max_indices = y_pred_flat.argmax(dim=1)

        i, j, v = torch.unravel_index(max_indices, (h, w, self.vocab_size))

        out = x_in.clone()
        out[torch.arange(b), i, j] = v

        return out

    def forward_loss(self, y_pred: Tensor, x_in: Tensor, x_out: Tensor) -> tuple[Tensor, Tensor]:
        b, h, w = x_in.shape

        diff = x_in != x_out

        loc = torch.where(diff)
        v = x_out[diff]

        y_pred = rearrange(y_pred, "b (h w) v -> b h w v", h=h, w=w, v=self.vocab_size)
        y_target = torch.zeros_like(y_pred)
        y_target[*loc, v] = 1.0

        loss = F.cross_entropy(
            rearrange(y_pred, "b h w v -> b (h w v)"),
            rearrange(y_target, "b h w v -> b (h w v)"),
        )

        accuracy = (y_pred.argmax(dim=-1)[diff] == v).float().mean()

        return loss, accuracy

    def forward(self, x: Tensor) -> Tensor:
        grid = x[:, self.n_registers :, :]  # Drop register tokens
        grid = rearrange(grid, "b s d -> b d s")

        logits = self.proj(grid)
        logits = rearrange(logits, "b d s -> b s d")

        return logits
