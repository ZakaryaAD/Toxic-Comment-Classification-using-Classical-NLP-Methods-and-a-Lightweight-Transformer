import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        y = F.relu(self.linear1(x))
        y = self.linear2(self.dropout2(y))
        x = self.norm2(x + y)
        return x

class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 300,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)      # (B, T, d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)            # mean pooling (B, d_model)
        return self.classifier(pooled)    # logits (B, num_labels)
