import typing
from abc import ABCMeta, abstractmethod
from typing import Optional

import torch.nn as nn
from torch import Size, Tensor
from torch.utils.data import Dataset

# ==================== #
# ===== Datasets ===== #
# ==================== #


class DatasetBase(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Vocabulary size.
        """
        ...

    @abstractmethod
    def __len__(self):
        """Get dataset length.

        Returns:
            Length of the dataset.
        """
        ...


# =================== #
# ===== Modules ===== #
# =================== #


class PositionalEncodingBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for positional encodings."""

    @abstractmethod
    def forward(self, x: Tensor, orig_size: Size) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: (b, s, d) Input sequence.
            orig_size: Original size of the input (for 2D positional encodings).

        Returns:
            (b, s, d) Input with positional encoding added.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class AttentionBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for attention mechanisms."""

    @abstractmethod
    def forward(
        self,
        q: Tensor,
        ctx: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention mechanism.

        Args:
            q: (b, s, dq) Query sequence.
            ctx: (b, t, dc) Context sequence (key/value).
            mask: (b, s, t) Attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (output, attention_weights)

            - output: (b, s, dq) Attention output.
            - attention_weights: (b, h, s, t) Attention weights per head.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class FeedForwardBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for feed-forward networks."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: (b, s, d) Input tensor.

        Returns:
            (b, s, d) Output tensor.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerEncoderLayerBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer encoder layers."""

    @abstractmethod
    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply encoder layer transformation.

        Args:
            src: (b, s, d) Encoder sequence.
            mask: (b, s, s) Self-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (output, self_attention_weights)

            - output: (b, s, d) Layer output.
            - self_attention_weights: (b, h, s, s) Self-attention weights.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerDecoderLayerBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer decoder layers."""

    @abstractmethod
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply decoder layer transformation.

        Args:
            tgt: (b, t, dt) Decoder sequence.
            memory: (b, s, ds) Encoder memory.
            tgt_mask: (b, t, t) Self-attention mask, where 0=visible, 1=masked.
            memory_mask: (b, t, s) Cross-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights)

            - output: (b, t, d) Layer output.
            - self_attn_weights: (b, h, t, t) Self-attention weights.
            - cross_attn_weights: (b, h, t, s) Cross-attention weights.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerEncoderBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer encoders."""

    @abstractmethod
    def forward(
        self,
        src: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply transformer encoder.

        Args:
            src: (b, s, d) Encoder sequence.
            mask: (b, s, s) Self-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (output, attention_weights)

            - output: (b, s, d) Encoder output.
            - attention_weights: (num_layers, b, h, s, s) Attention weights per layer.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerDecoderBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer decoders."""

    @abstractmethod
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply transformer decoder.

        Args:
            tgt: (b, t, dt) Decoder sequence.
            memory: (b, s, ds) Encoder memory.
            tgt_mask: (b, t, t) Self-attention mask, where 0=visible, 1=masked.
            memory_mask: (b, t, s) Cross-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (output, self_attn_weights, cross_attn_weights)

            - output: (b, t, d) Decoder output.
            - self_attn_weights: (num_layers, b, h, t, t) Self-attention weights per layer.
            - cross_attn_weights: (num_layers, b, h, t, s) Cross-attention weights per layer.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer models."""

    @abstractmethod
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply vanilla transformer.

        Args:
            src: (b, s, ds) Source sequence.
            tgt: (b, t, dt) Target sequence.
            src_mask: (b, s, s) Source self-attention mask, where 0=visible, 1=masked.
            tgt_mask: (b, t, t) Target self-attention mask, where 0=visible, 1=masked.
            memory_mask: (b, t, s) Cross-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (probs, logits, encoder_output, decoder_output,
            encoder_attns, decoder_self_attns, decoder_cross_attns)

            - probs: (b, t, n_tokens) Token probabilities.
            - logits: (b, t, n_tokens) Token logits.
            - encoder_output: (b, s, ds) Encoder output.
            - decoder_output: (b, t, dt) Decoder output.
            - encoder_attns: (num_layers, b, h, s, s) Encoder attention weights per layer.
            - decoder_self_attns: (num_layers, b, h, t, t) Decoder self-attention weights per layer.
            - decoder_cross_attns: (num_layers, b, h, t, s) Decoder cross-attention weights per layer.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward
