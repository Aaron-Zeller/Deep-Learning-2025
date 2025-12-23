import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Size, Tensor

from src.interfaces import DatasetBase, PositionalEncodingBase, TransformerBase, TransformerHeadBase


class CalculationHead(nn.Module):
    def __init__(self, n_action_tokens: int, dim: int, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.n_action_tokens = n_action_tokens
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.input_dim = n_action_tokens * dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, vocab_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict value logits.

        Args:
            x: Input tensor of shape (b, n_action_tokens, d).

        Returns:
            logits: Value logits of shape (b, vocab_size).
        """
        b = x.shape[0]
        x = rearrange(x, "b n d -> b (n d)")
        logits = self.layers(x)
        return logits


class LocationHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.proj = nn.Conv1d(self.dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict location logits.

        Args:
            x: Input tensor of shape (b, (h w), d).

        Returns:
            logits: Location logits of shape (b, (h w)).
        """
        x = rearrange(x, "b s d -> b d s")
        logits = self.proj(x)
        logits = rearrange(logits, "b 1 s -> b s")
        return logits


class SplitActionDistributionHead(TransformerHeadBase):
    """Transformer head with separate location and value predictions."""

    def __init__(
        self,
        n_registers: int,
        n_action_tokens: int,
        dim: int,
        dataset: DatasetBase,
        alpha: float = 1.0,
        beta: float = 4.0,
    ):
        """Initialize global action distribution head.

        Args:
            n_registers: Number of register tokens. This helps avoiding contamination of the data tokens.
            n_action_tokens: Number of action tokens for value prediction.
            dim: Model dimensionality.
            dataset: Dataset instance.
            alpha: Weight for value prediction loss.
            beta: Weight for location prediction loss.
        """
        super().__init__()

        if n_action_tokens <= 0:
            raise ValueError("n_action_tokens must be greater than 0 for SplitActionDistributionHead.")

        self.n_registers = n_registers
        self.vocab_size = dataset.vocab_size()
        self.dim = dim
        self.alpha = alpha
        self.beta = beta

        # Registers used as additional "memory" context
        _registers = torch.randn(self.n_registers, self.dim)
        self.registers = nn.Parameter(_registers)

        # Action tokens used for value prediction
        self.n_action_tokens = n_action_tokens
        _action_tokens = torch.randn(1, n_action_tokens, dim)
        self.action_tokens = nn.Parameter(_action_tokens)

        self.location_head = LocationHead(self.dim)
        self.value_head = CalculationHead(self.n_action_tokens, self.dim, self.vocab_size)

    def inject(
        self,
        src: Tensor,
        tgt: Tensor,
        pos_encoding: PositionalEncodingBase,
        src_orig_size: Size,
        tgt_orig_size: Size,
    ) -> tuple[Tensor, Tensor]:
        b = src.shape[0]

        action_tokens = repeat(self.action_tokens, "1 n d -> b n d", b=b)
        registers = repeat(self.registers[None, ...], "1 n d -> b n d", b=b)

        src = torch.cat([action_tokens, registers, src], dim=1)

        return src, tgt

    def step(self, x_in: Tensor, y_pred: Tensor) -> Tensor:
        b, h, w = x_in.shape

        v_pred, loc_pred = y_pred

        max_indices = loc_pred.argmax(dim=1)

        i, j = torch.unravel_index(max_indices, (h, w))

        out = x_in.clone()
        out[torch.arange(b), i, j] = v_pred.argmax(dim=1)

        return out

    def forward_loss(self, y_pred: tuple[Tensor, Tensor], x_in: Tensor, x_out: Tensor) -> tuple[Tensor, Tensor]:
        # x_in, x_out: (b, h, w)
        v_pred, loc_pred = y_pred  # (b, vocab_size), (b, h*w)
        b, h, w = x_in.shape
        vocab_size = v_pred.shape[-1]

        # Ground truth
        diff = x_in != x_out
        # flattened index of the changed location
        true_loc_idx = diff.view(b, -1).long().argmax(dim=-1)  # (b,)

        # flattened index of the true value in the changed location
        true_v_idx = x_out.view(b, -1).gather(1, true_loc_idx.unsqueeze(1)).squeeze(1)  # (b,)

        # target value distribution
        v_target_dist = torch.zeros_like(v_pred, device=x_in.device, dtype=torch.float)
        v_target_dist[torch.arange(b), true_v_idx] = 1.0

        # target location distribution
        loc_target_dist = torch.zeros((b, h * w), device=x_in.device, dtype=torch.float)
        loc_target_dist[torch.arange(b), true_loc_idx] = 1.0

        # Losses
        v_loss = F.cross_entropy(v_pred, v_target_dist)
        loc_loss = F.cross_entropy(loc_pred, loc_target_dist)
        loss = self.alpha * v_loss + self.beta * loc_loss

        # Accuracy
        pred_loc_idx = torch.argmax(loc_pred, dim=-1)  # (b,)
        pred_v_idx = torch.argmax(v_pred, dim=-1)  # (b,)
        correct_loc = pred_loc_idx == true_loc_idx
        correct_val = pred_v_idx == true_v_idx
        accuracy = (correct_loc & correct_val).float().mean()

        return loss, accuracy

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # currently doesn't have register tokens
        action_tokens = x[:, : self.n_action_tokens, :]
        grid_tokens = x[:, self.n_action_tokens + self.n_registers :, :]  # Drop register tokens

        value = self.value_head(action_tokens)
        positions = self.location_head(grid_tokens)

        return value, positions  # hack to return two tensors
