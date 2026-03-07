from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pickle

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


class TransformerBehaviorModel(nn.Module):
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


class ImprovedGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = in_dim == out_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(adj, x)
        h = self.linear(h)
        if self.use_residual:
            h = h + x
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class BiLSTMGCNBehaviorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes_list: List[int],
        embed_dim: int = 256,
        lstm_hidden: int = 256,
        gcn_hidden: int = 256,
        stat_dim: int = 6,
        dropout: float = 0.15,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(lstm_hidden * 2)

        self.gcn1 = ImprovedGCNLayer(embed_dim, gcn_hidden, dropout=dropout)
        self.gcn2 = ImprovedGCNLayer(gcn_hidden, gcn_hidden, dropout=dropout)
        self.gcn3 = ImprovedGCNLayer(gcn_hidden, gcn_hidden, dropout=dropout)
        self.gcn_dropout = nn.Dropout(dropout)

        fusion_dim = lstm_hidden * 2 + gcn_hidden + stat_dim
        self.fc_shared = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([nn.Linear(256, c) for c in num_classes_list])

    def build_adj_matrix(self, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.size()
        adj = torch.zeros(batch_size, seq_len, seq_len, device=mask.device)

        for i in range(seq_len - 1):
            adj[:, i, i + 1] = 1
            adj[:, i + 1, i] = 1

        adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        row_sum = adj.sum(dim=-1, keepdim=True) + 1e-8
        return adj / row_sum

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, aux: torch.Tensor):
        emb = self.embedding(seq)
        emb = self.embed_dropout(emb)

        lstm_out, _ = self.lstm(emb)
        lstm_out = lstm_out * mask.unsqueeze(-1)

        if self.use_attention:
            key_padding = (1 - mask).bool()
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding)
            lstm_out = self.attn_norm(lstm_out + attn_out)

        lstm_out = self.lstm_dropout(lstm_out)
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-8)
        lstm_feat = torch.sum(lstm_out, dim=1) / denom

        adj = self.build_adj_matrix(mask)
        gcn_out = self.gcn1(emb, adj)
        gcn_out = self.gcn2(gcn_out, adj)
        gcn_out = self.gcn3(gcn_out, adj)
        gcn_out = gcn_out * mask.unsqueeze(-1)
        gcn_out = self.gcn_dropout(gcn_out)
        gcn_feat = torch.sum(gcn_out, dim=1) / denom

        fused = torch.cat([lstm_feat, gcn_feat, aux], dim=1)
        shared = self.fc_shared(fused)
        return [head(shared) for head in self.heads]


class BehaviorXGBoostModel:
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 350,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        n_jobs: int = -1,
        device: str = "auto",
    ) -> None:
        resolved_device = "cuda" if str(device).lower().startswith("cuda") else "cpu"
        self.params = {
            "random_state": random_state,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "n_jobs": n_jobs,
            "tree_method": "hist",
            "device": resolved_device,
            "eval_metric": "mlogloss",
        }
        self.models: List[Any] = []

    @staticmethod
    def _compute_balanced_sample_weight(y: np.ndarray, num_classes: int) -> np.ndarray:
        class_counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        class_counts[class_counts == 0.0] = 1.0
        class_weights = y.shape[0] / (num_classes * class_counts)
        return class_weights[y].astype(np.float32)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        early_stopping_rounds: int = 0,
        use_class_weights: bool = True,
    ) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for model_type='xgboost'.") from exc

        self.models = []
        for idx in range(labels.shape[1]):
            y = labels[:, idx]
            num_classes = int(np.unique(y).shape[0])

            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                **self.params,
            )

            fit_kwargs: Dict[str, Any] = {}
            if use_class_weights:
                fit_kwargs["sample_weight"] = self._compute_balanced_sample_weight(y, num_classes)

            has_val = val_features is not None and val_labels is not None
            if has_val and early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(val_features, val_labels[:, idx])]
                fit_kwargs["verbose"] = False

                # Compatibility with multiple xgboost versions.
                try:
                    clf.fit(
                        features,
                        y,
                        early_stopping_rounds=early_stopping_rounds,
                        **fit_kwargs,
                    )
                except TypeError:
                    clf.fit(features, y, **fit_kwargs)
            else:
                clf.fit(features, y, **fit_kwargs)

            self.models.append(clf)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("XGBoost model is not fitted.")
        preds = [model.predict(features).astype(np.int64) for model in self.models]
        return np.stack(preds, axis=1)

    def save(self, path: Path, extra: Dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "xgboost",
            "params": self.params,
            "models": self.models,
            "extra": extra or {},
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "BehaviorXGBoostModel":
        with path.open("rb") as f:
            payload = pickle.load(f)

        obj = cls()
        obj.params = payload["params"]
        obj.models = payload["models"]
        return obj


class BehaviorModelFactory:
    @staticmethod
    def create(
        model_type: str,
        vocab_size: int,
        num_classes_list: List[int],
        model_hparams: Dict[str, Any],
    ):
        if model_type == "transformer":
            return TransformerBehaviorModel(
                vocab_size=vocab_size,
                num_classes_list=num_classes_list,
                max_len=model_hparams.get("max_len", 24),
                embedding_dim=model_hparams.get("embedding_dim", 256),
                nhead=model_hparams.get("nhead", 8),
                ff_dim=model_hparams.get("ff_dim", 512),
                num_layers=model_hparams.get("num_layers", 2),
                dropout=model_hparams.get("dropout", 0.3),
                aux_dim=model_hparams.get("aux_dim", 64),
                hidden_dim=model_hparams.get("hidden_dim", 256),
                branch_dim=model_hparams.get("branch_dim", 128),
            )
        if model_type == "bilstm_gcn":
            return BiLSTMGCNBehaviorModel(
                vocab_size=vocab_size,
                num_classes_list=num_classes_list,
                embed_dim=model_hparams.get("embed_dim", 256),
                lstm_hidden=model_hparams.get("lstm_hidden", 256),
                gcn_hidden=model_hparams.get("gcn_hidden", 256),
                stat_dim=model_hparams.get("stat_dim", 6),
                dropout=model_hparams.get("dropout", 0.15),
                use_attention=model_hparams.get("use_attention", False),
            )
        raise ValueError(f"Unsupported torch model_type: {model_type}")


# Backward-compatible alias for older imports.
BehaviorModel = TransformerBehaviorModel
