from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class BehaviorDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        aux_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
    ) -> None:
        self.sequences = sequences
        self.masks = masks
        self.aux_features = aux_features
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        aux = torch.tensor(self.aux_features[idx], dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return seq, mask, aux, label

        if self.ids is not None:
            return seq, mask, aux, self.ids[idx]

        return seq, mask, aux
