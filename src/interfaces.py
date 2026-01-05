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
    def grid_size(self) -> tuple[int, int]:
        """Get grid size.

        Returns:
            Grid size as (height, width).
        """
        ...

    @abstractmethod
    def get_example(self) -> Tensor:
        """Get a single example from the dataset.

        Returns:
            (h, w) Example tensor.
        """
        ...

    def to_string(self, x: Tensor) -> str:
        """Convert tensor representation to string.

        Args:
            x: (h, w) Input tensor.

        Returns:
            String representation.
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


class RelativePositionalEncodingBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for relative positional encodings."""

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply relative encoding between q and k. Matrices

        Args:
            q: (b h s d) Query sequence splitted over heads.
            k: (b h s d) Key sequence splitted over heads.

        Returns:
            Tuple of (encoded_q, encoded_k)

            - encoded_q: (b h s d) relative encoded queries
            - encoded_k: (b h s d) relative encoded keys
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


class TransformerHeadBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer heads."""

    def inject(
        self,
        src: Tensor,
        tgt: Tensor,
        pos_encoding: PositionalEncodingBase,
        src_orig_size: Size,
        tgt_orig_size: Size,
    ) -> tuple[Tensor, Tensor]:
        """Inject tokens into source and target sequences for the transformer. This is used for head-specific tokens.

        Args:
            src: (b, s, ds) Source sequence.
            tgt: (b, t, dt) Target sequence.
            pos_encoding: Positional encoding module.
            src_orig_size: Original size of the source (for 2D positional encodings).
            tgt_orig_size: Original size of the target (for 2D positional encodings).
        Returns:
            Tuple of (src, tgt)
            - src: (b, s, ds) Modified source sequence.
            - tgt: (b, t, dt) Modified target sequence.
        """
        return src, tgt

    @abstractmethod
    def step(self, x_in: Tensor, y_pred: Tensor) -> Tensor:
        """Perform a calculation step.

        Args:
            x_in: (b, h, w) Input values.
            y_pred: (b, s, vocab_size) Predicted output from the head.
        Returns:
            y_sampled: (b, h, w) Sampled output.
        """
        ...

    @abstractmethod
    def forward_loss(self, y_pred: Tensor, x_in: Tensor, x_out: Tensor) -> tuple[Tensor, Tensor]:
        """Compute loss for the transformer head.

        Args:
            y_pred: (b, n, vocab_size) Predicted output from the head.
            x_in: (b, h, w) Input values.
            x_out: (b, h, w) Target output values.
        Returns:
            Tuple of (loss, accuracy)
            - loss: (1,) Loss.
            - accuracy: (1,) Accuracy.
        """
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> list[Tensor]:
        """Apply transformer head.

        Args:
            x: (b, s, ds) Latent sequence.
        Returns:
            Head specific outputs.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward


class TransformerBase(nn.Module, metaclass=ABCMeta):
    """Abstract class for transformer models."""

    @abstractmethod
    def dim(self) -> int:
        """Get model dimensionality.

        Returns:
            Model dimensionality.
        """
        ...

    @abstractmethod
    def prepare_tokens(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Prepare source and target tokens for the transformer.

        Args:
            src: (b, h, w) Source input.
            tgt: (b, h, w) Target input.
        Returns:
            Tuple of (src, tgt)
            - src: (b, s, d) Prepared source sequence.
            - tgt: (b, t, d) Prepared target sequence.
        """

    @abstractmethod
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
        """Apply vanilla transformer.

        Args:
            src: (b, s, ds) Source sequence.
            tgt: (b, t, dt) Target sequence.
            src_mask: (b, s, s) Source self-attention mask, where 0=visible, 1=masked.
            tgt_mask: (b, t, t) Target self-attention mask, where 0=visible, 1=masked.
            memory_mask: (b, t, s) Cross-attention mask, where 0=visible, 1=masked.

        Returns:
            Tuple of (encoder_output, decoder_output, encoder_attns, decoder_self_attns, decoder_cross_attns)

            - encoder_output: (b, s, ds) Encoder output.
            - decoder_output: (b, t, dt) Decoder output.
            - encoder_attns: (num_layers, b, h, s, s) Encoder attention weights per layer.
            - decoder_self_attns: (num_layers, b, h, t, t) Decoder self-attention weights per layer.
            - decoder_cross_attns: (num_layers, b, h, t, s) Decoder cross-attention weights per layer.
            - extra_outputs: Optional dictionary for any extra outputs.
        """
        ...

    if typing.TYPE_CHECKING:
        __call__ = forward
