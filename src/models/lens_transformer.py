import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.interfaces import (
    DatasetBase,
    TransformerBase,
    TransformerEncoderBase,
    TransformerHeadBase,
)

logger = logging.getLogger(__name__)


class LensTransformer(TransformerBase):
    """Lens Transformer with encoder architecture."""

    def __init__(
        self,
        lens: nn.Module,
        encoder: TransformerEncoderBase,
        dataset: DatasetBase,
        dim: int,
        n_locations: int,
        mask_type: Literal["none", "hard", "soft"] = "hard",
    ):
        """Initialize transformer.

        Args:
            lens: Lens module.
            encoder: Transformer encoder module.
            dataset: Dataset instance.
            dim: Model dimensionality.
            n_locations: Number locations that are looked at, at any given moment.
            window_size: Size of the window to look at around each location. A value of 1 means a 3x3 window.
            mask_type: Type of masking to use. One of "none", "hard", or "soft".
        """
        super().__init__()

        vocab_size = dataset.vocab_size()
        self.src_embed = nn.Embedding(vocab_size, dim)
        self.tgt_embed = nn.Embedding(vocab_size, dim)
        self.lens = lens
        logger.info(f"Lens: {self.lens}")
        # todo: figure out why this is different compared to using the hydra config initialization
        # self.lens = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=5, padding=2),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim + 1, kernel_size=1),
        # )
        self.encoder = encoder

        self._dim = dim
        self.n_locations = n_locations
        self.mask_type = mask_type

    def dim(self) -> int:
        """Get model dimensionality.

        Returns:
            Model dimensionality.
        """
        return self._dim

    def prepare_tokens(
        self,
        src: Tensor,
        tgt: Tensor,
        head: TransformerHeadBase,
    ) -> tuple[Tensor, Tensor]:
        # Embed tokens: (b, h, w) -> (b, h, w, d)
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)

        # todo: figure out how to deal with the head, should probably be done in the forward pass
        # src_enc, tgt_enc = head.inject(src_enc, tgt_enc, self.pos_encoding, src_orig_size, tgt_orig_size)

        # No positional encoding needed, since that's taken care of by the lens CNN module

        return src_emb, tgt_emb

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[dict]
    ]:
        # Run lens to match local patterns and create implicit positional embedding
        src_lens = self.lens(rearrange(src, "b h w d -> b d h w"))
        src_lens = rearrange(src_lens, "b d h w -> b (h w) d")

        # Select n_locations with maximal values
        mask = torch.zeros_like(src_lens[..., :1])

        if self.mask_type == "none":
            mask = 1 - mask
        elif self.mask_type in ["hard", "soft"]:
            for i in range(self.n_locations):
                mask = mask + F.gumbel_softmax(
                    src_lens[..., :1] * (1 - mask), tau=1.0, dim=1, hard=self.mask_type == "hard"
                )
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        # Mask out all non-maximal tokens
        src_selected = src_lens[..., 1:] * mask

        memory, enc_attns = self.encoder(src_selected, mask=src_mask)

        return memory, None, enc_attns, None, None, {"mask": mask}
