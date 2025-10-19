import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLu(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-Attention sublayer (1)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop(attn_out))  # residual connection in attention

        # Feed-Forward sublayer (2)
        ff = self.ff(x)
        x = self.norm2(x + self.drop(ff))  # residual connection in feedforward
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # Decoder Direction
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # Flows from Encoder
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLu(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )

    def forward(
        self,
        y: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_pad_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
        cross_attn_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Self-Attention sublayer
        self_attn_out, _ = self.self_attn(y, y, y, attn_mask=self_attn_mask, key_padding_mask=self_attn_pad_mask)
        y = self.norm1(y + self.drop(self_attn_out))

        # Cross-Attention sublayer
        cross_attn_out, _ = self.cross_attn(
            encoder_out, encoder_out, y, attn_mask=cross_attn_mask, key_padding_mask=cross_attn_pad_mask
        )
        y = self.norm2(y + self.drop(cross_attn_out))

        # Feed-Forward sublayer
        ff = self.ff(y)
        y = self.norm3(y + self.drop(ff))


# Ordering information for Attention
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.H = height
        self.W = width
        self.D = d_model
        self.row = nn.Embedding(height, d_model)
        self.col = nn.Embedding(width, d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x is given as [B, T, D] B = Batch Size, T = H * W (flattened), D = Token Dim (Size)
        B, T, D = x.shape

        # Check if assumptions are met - Debugging REMOVE once it works
        if T != self.H * self.W:
            raise ValueError(f"T={T} must equal H*W={self.H*self.W}.")

        device, dtype = x.device, x.dtype
        idx = torch.arange(T, device=device)
        rows = idx // self.W
        cols = idx % self.W
        pos_encoding = self.row(rows) + self.col(cols)
        pos_encoding = pos_encoding.unsqueeze(0).to(dtype=dtype)

        if padding_mask is not None:
            pos_encoding = pos_encoding * (~padding_mask.bool()).unsqueeze(-1)

        # Inject positional encoding to token - allows the information to be propagated through network
        return x + pos_encoding


# Ensures that only previous and current tokens are used -> creates causal structure
def causal_mask(T, device) -> torch.Tensor:
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return m.masked_fill(m == 1, float("-inf"))


class TinyTransformer(nn.Module):
    def __init__(
        self,
        source_vocab: int,
        target_vocab: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        max_len: int,
        dropout: int,
    ):
        super().__init__()

        # Embeddings and Positional Encoding
        self.source_tokens = nn.Embedding(source_vocab, d_model)
        self.target_tokens = nn.Embedding(target_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder Blocks
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

        # Decoder Blocks
        self.decoder = nn.ModuleList(
            [TransformerDecoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )

        # Output Linear Layer + Norm (Softmax removed - only in loss)
        self.ln_ff = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, target_vocab)

    def encode(self, source_ids: torch.Tensor, source_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.pos_encoding(self.source_tokens(source_ids))

        for encoder_block in self.encoder:
            x = encoder_block(x, source_key_padding_mask=source_key_padding_mask)
        return x

    def decode(
        self,
        target_ids: torch.Tensor,
        encode_output: torch.Tensor,
        target_mask: torch.Tensor,
        target_key_padding_mask: torch.Tensor,
        encoder_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        y = self.pos_encoding(self.target_tokens(target_ids))

        for decoder_block in self.decoder:
            y = decoder_block(
                y,
                encode_output,
                target_mask=target_mask,
                target_key_padding_mask=target_key_padding_mask,
                encoder_key_padding_mask=encoder_key_padding_mask,
            )

        return y

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_key_padding_mask: torch.Tensor = None,
        target_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        encode_out = self.encode(source_ids, source_key_padding_mask=source_key_padding_mask)
        T = target_ids.size(1)
        mask = causal_mask(T, target_ids.device)

        decode_out = self.decode(
            target_ids,
            encode_out,
            target_mask=mask,
            target_key_padding_mask=target_key_padding_mask,
            encoder_key_padding_mask=source_key_padding_mask,
        )

        decode_out = self.ln_ff(decode_out)
        return self.head(decode_out)
