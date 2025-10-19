import torch
import torch.nn as nn

# ==========================================================
#               Transformer (Just Decoder)
# ==========================================================
#
#             ┌──────────────────────────────┐
#             │        Input Tokens          │
#             └──────────────┬───────────────┘
#                            │
#                            ▼
#                   [Token Embeddings]
#                            │
#                  [Positional Encoding]
#                            │
#                            ▼
#           ┌───────────────────────────────────┐
#           │       N × Transformer Blocks      │
#           │───────────────────────────────────│
#           │                                   │
#           │  ┌─────────────────────────────┐  │
#           │  │  Multi-Head Self-Attention  │  │
#           │  │      (with causal mask) (1) │  │
#           │  └──────────────┬──────────────┘  │
#           │                 │ Residual + Norm │
#           │                 ▼                 │
#           │       Feed-Forward MLP (2)        │
#           │                 │ Residual + Norm │
#           └─────────────────┴─────────────────┘
#                             │
#                             ▼
#                    [Final LayerNorm]
#                             │
#                             ▼
#                [Linear Head → Vocabulary]
#                             │
#                             ▼
#                        Predictions
#


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # Self-Attention sublayer (1)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)  # residual connection in attention

        # Feed-Forward sublayer (2)
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff)  # residual connection in feedforward
        return x
