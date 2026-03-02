from dataclasses import dataclass
from pathlib import Path
from typing import List


ATTR_COLUMNS: List[str] = [
    "attr_1",
    "attr_2",
    "attr_3",
    "attr_4",
    "attr_5",
    "attr_6",
]


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")
    model_name: str = "best_model_cpu.pth"
    preprocessor_name: str = "preprocessor.pt"
    max_len: int = 24
    embedding_dim: int = 256
    nhead: int = 8
    ff_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    aux_dim: int = 64
    hidden_dim: int = 256
    branch_dim: int = 128
    train_batch_size: int = 64
    val_batch_size: int = 128
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
