from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import ATTR_COLUMNS


@dataclass
class PreprocessorArtifacts:
    max_len: int
    feature_columns: List[str]
    action2idx: Dict[int, int]
    label_encoders: Dict[str, Dict[int, int]]


class BehaviorPreprocessor:
    def __init__(self, max_len: int = 24) -> None:
        self.max_len = max_len
        self.feature_columns: List[str] = []
        self.action2idx: Dict[int, int] = {}
        self.label_encoders: Dict[str, Dict[int, int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.action2idx)

    @property
    def num_classes_list(self) -> List[int]:
        return [len(self.label_encoders[col]) for col in ATTR_COLUMNS]

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self.feature_columns = [col for col in x_train.columns if col != "id"]

        flattened = x_train[self.feature_columns].to_numpy().reshape(-1)
        series = pd.to_numeric(pd.Series(flattened), errors="coerce").dropna().astype(np.int64)
        unique_actions = sorted(series.unique().tolist())

        self.action2idx = {action: idx + 1 for idx, action in enumerate(unique_actions)}
        self.label_encoders = {}

        for col in ATTR_COLUMNS:
            unique_vals = sorted(y_train[col].dropna().astype(np.int64).unique().tolist())
            self.label_encoders[col] = {value: idx for idx, value in enumerate(unique_vals)}

    def _clean_actions(self, values: np.ndarray) -> np.ndarray:
        actions = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(np.int64).to_numpy()
        return actions

    def process_sequence(self, values: np.ndarray) -> Tuple[List[int], List[int], int]:
        actions = self._clean_actions(values)
        mapped = [self.action2idx.get(int(action), 0) for action in actions]

        seq_length = len(mapped)
        if seq_length > self.max_len:
            mapped = mapped[-self.max_len :]

        padded = mapped + [0] * (self.max_len - len(mapped))
        mask = [1 if token != 0 else 0 for token in padded]
        return padded, mask, seq_length

    def compute_aux_features(self, values: np.ndarray) -> np.ndarray:
        actions = self._clean_actions(values)

        length = len(actions)
        if length == 0:
            return np.array([0, 0.0, 0.0, 0.0, 0, 0], dtype=np.float32)

        unique_vals, counts = np.unique(actions, return_counts=True)
        unique_count = len(unique_vals)
        unique_ratio = unique_count / length
        repetition_ratio = 1.0 - unique_ratio

        probs = counts / length
        ent = float(-(probs * np.log(probs + 1e-12)).sum())

        first_action = self.action2idx.get(int(actions[0]), 0)
        last_action = self.action2idx.get(int(actions[-1]), 0)

        return np.array(
            [
                float(length),
                float(unique_ratio),
                float(repetition_ratio),
                float(ent),
                float(first_action),
                float(last_action),
            ],
            dtype=np.float32,
        )

    def transform_features(self, x_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.feature_columns or not self.action2idx:
            raise RuntimeError("Preprocessor must be fit before transform_features.")

        ids = x_df["id"].to_numpy()
        seqs, masks, aux_feats = [], [], []

        for _, row in x_df[self.feature_columns].iterrows():
            padded, mask, _ = self.process_sequence(row.values)
            aux = self.compute_aux_features(row.values)

            seqs.append(padded)
            masks.append(mask)
            aux_feats.append(aux)

        return (
            np.asarray(seqs, dtype=np.int64),
            np.asarray(masks, dtype=np.float32),
            np.asarray(aux_feats, dtype=np.float32),
            ids,
        )

    def transform_labels(self, y_df: pd.DataFrame, strict: bool = True) -> np.ndarray:
        if not self.label_encoders:
            raise RuntimeError("Preprocessor must be fit before transform_labels.")

        encoded_cols = []
        for col in ATTR_COLUMNS:
            mapped = y_df[col].astype(np.int64).map(self.label_encoders[col])
            if strict and mapped.isna().any():
                unseen = sorted(y_df.loc[mapped.isna(), col].unique().tolist())
                raise ValueError(
                    f"Unseen labels in column {col}: {unseen[:10]}. "
                    "Validation/test labels must be transform-only."
                )
            encoded_cols.append(mapped.fillna(-1).astype(np.int64).to_numpy())

        return np.stack(encoded_cols, axis=1)

    def decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        reverse_maps = {
            col: {encoded: original for original, encoded in mapping.items()}
            for col, mapping in self.label_encoders.items()
        }

        decoded = np.zeros_like(predictions, dtype=np.int64)
        for attr_idx, col in enumerate(ATTR_COLUMNS):
            mapper = reverse_maps[col]
            decoded[:, attr_idx] = np.array(
                [mapper[int(v)] for v in predictions[:, attr_idx]],
                dtype=np.int64,
            )
        return decoded

    def to_artifacts(self) -> PreprocessorArtifacts:
        return PreprocessorArtifacts(
            max_len=self.max_len,
            feature_columns=self.feature_columns,
            action2idx=self.action2idx,
            label_encoders=self.label_encoders,
        )

    @classmethod
    def from_artifacts(cls, artifacts: PreprocessorArtifacts) -> "BehaviorPreprocessor":
        obj = cls(max_len=artifacts.max_len)
        obj.feature_columns = artifacts.feature_columns
        obj.action2idx = {int(k): int(v) for k, v in artifacts.action2idx.items()}
        obj.label_encoders = {
            col: {int(k): int(v) for k, v in mapping.items()}
            for col, mapping in artifacts.label_encoders.items()
        }
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self.to_artifacts()), path)

    @classmethod
    def load(cls, path: Path) -> "BehaviorPreprocessor":
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found: {path}")
        payload = torch.load(path, map_location="cpu")
        artifacts = PreprocessorArtifacts(**payload)
        return cls.from_artifacts(artifacts)
