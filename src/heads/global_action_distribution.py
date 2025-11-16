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
        n_latents: int,
        transformer: TransformerBase,
        dataset: DatasetBase,
    ):
        """Initialize global action distribution head.

        Args:
            n_latents: Number of latent tokens.
            transformer: Transformer model.
            dataset: Dataset instance.
        """
        super().__init__()

        self.n_latents = n_latents
        self.dim = transformer.dim()
        self.h, self.w = dataset.grid_size()
        self.vocab_size = dataset.vocab_size()

        _latents = torch.randn(self.n_latents, self.dim) / (self.dim**0.5)
        self.latents = nn.Parameter(_latents)

        self.proj = nn.Linear(self.n_latents * self.dim, self.h * self.w * self.vocab_size)

    def inject(
        self,
        src: Tensor,
        tgt: Tensor,
        pos_encoding: PositionalEncodingBase,
        src_orig_size: Size,
        tgt_orig_size: Size,
    ) -> tuple[Tensor, Tensor]:
        b = src.shape[0]

        latents = repeat(self.latents[None, ...], "1 n d -> b n d", b=b)
        src = torch.cat([latents, src], dim=1)

        return src, tgt

    def step(self, x_in: Tensor, y_pred: Tensor) -> Tensor:
        b = x_in.shape[0]

        y_pred_flat = rearrange(y_pred, "b h w v -> b (h w v)")

        max_indices = y_pred_flat.argmax(dim=1)

        i, j, v = torch.unravel_index(max_indices, (self.h, self.w, self.vocab_size))

        out = x_in.clone()
        out[torch.arange(b), i, j] = v

        return out

    def forward_loss(self, y_pred: Tensor, x_in: Tensor, x_out: Tensor) -> tuple[Tensor, Tensor]:
        diff = x_in != x_out

        loc = torch.where(diff)
        v = x_out[diff]

        y_target = torch.zeros_like(y_pred)
        y_target[*loc, v] = 1.0

        loss = F.cross_entropy(
            rearrange(y_pred, "b h w v -> b (h w v)"),
            rearrange(y_target, "b h w v -> b (h w v)"),
        )

        accuracy = (y_pred.argmax(dim=-1)[diff] == v).float().mean()

        return loss, accuracy

    def forward(self, x: Tensor) -> Tensor:
        latents = x[:, : self.n_latents, :]
        latents = rearrange(latents, "b n d -> b (n d)")

        logits = self.proj(latents)
        logits = rearrange(logits, "b (h w v) -> b h w v", v=self.vocab_size, h=self.h, w=self.w)

        return logits
