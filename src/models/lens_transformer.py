import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
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
        head: TransformerHeadBase,
        lens: nn.Module,
        encoder: TransformerEncoderBase,
        dataset: DatasetBase,
        dim: int,
        n_locations: int,
        mask_type: Literal["none", "hard", "soft"] = "hard",
        n_action_tokens: int = 0,
    ):
        """Initialize transformer.

        Args:
            head: Transformer head module.
            lens: Lens module.
            encoder: Transformer encoder module.
            dataset: Dataset instance.
            dim: Model dimensionality.
            n_locations: Number locations that are looked at, at any given moment.
            mask_type: Type of masking to use. One of "none", "hard", or "soft".
            n_action_tokens: Number of action tokens.
        """
        super().__init__()

        vocab_size = dataset.vocab_size()
        self.src_embed = nn.Embedding(vocab_size, dim)
        self.tgt_embed = nn.Embedding(vocab_size, dim)
        self.lens = lens
        self.head = head
        self.encoder = encoder

        self._dim = dim
        self.n_locations = n_locations
        self.mask_type = mask_type

        self.n_action_tokens = n_action_tokens
        self.action_tokens = nn.Parameter(torch.randn(1, n_action_tokens, dim)) if n_action_tokens > 0 else None

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
    ) -> tuple[Tensor, Tensor]:
        # Embed tokens: (b, h, w) -> (b, h, w, d)
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)

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
        tgt_lens = rearrange(tgt, "b h w d -> b (h w) d")

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

        # Inject head specific tokens
        src_selected, tgt_lens = self.head.inject(src_selected, tgt_lens, None, src.shape, tgt.shape)

        memory, enc_attns = self.encoder(src_selected, mask=src_mask)

        return memory, None, enc_attns, None, None, {"mask": mask}
