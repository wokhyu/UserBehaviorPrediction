from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 24) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BehaviorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes_list: List[int],
        max_len: int = 24,
        embedding_dim: int = 256,
        nhead: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        aux_dim: int = 64,
        hidden_dim: int = 256,
        branch_dim: int = 128,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.aux_fc = nn.Linear(6, aux_dim)
        self.shared_fc = nn.Linear(embedding_dim + aux_dim, hidden_dim)
        self.branch_14 = nn.Linear(hidden_dim, branch_dim)

        self.heads = nn.ModuleList(
            [
                nn.Linear(branch_dim if i in [0, 3] else hidden_dim, num_classes_list[i])
                for i in range(6)
            ]
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, aux: torch.Tensor):
        x = self.embedding(seq)
        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=(mask == 0))
        denom = torch.clamp(mask.sum(1, keepdim=True), min=1e-6)
        x = (x * mask.unsqueeze(-1)).sum(1) / denom

        aux = F.relu(self.aux_fc(aux))
        x = torch.cat([x, aux], dim=1)
        x = F.relu(self.shared_fc(x))

        branch14 = F.relu(self.branch_14(x))
        outputs = []
        for head_idx in range(6):
            if head_idx in [0, 3]:
                outputs.append(self.heads[head_idx](branch14))
            else:
                outputs.append(self.heads[head_idx](x))
        return outputs
