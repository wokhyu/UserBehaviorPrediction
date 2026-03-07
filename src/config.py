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


DEFAULT_MODEL_NAMES = {
    "transformer": "best_model_transformer.pth",
    "bilstm_gcn": "best_model_bilstm_gcn.pth",
    "xgboost": "best_model_xgboost.pkl",
}


def default_model_name_for_type(model_type: str) -> str:
    key = (model_type or "transformer").strip().lower()
    return DEFAULT_MODEL_NAMES.get(key, DEFAULT_MODEL_NAMES["transformer"])


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")
    model_name: str = "best_model_transformer.pth"
    preprocessor_name: str = "preprocessor.pt"
    processed_dir_name: str = "processed"
    model_type: str = "transformer"
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
    device: str = "auto"
    xgb_n_estimators: int = 350
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 8
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_min_child_weight: float = 1.0
    xgb_gamma: float = 0.0
    xgb_n_jobs: int = -1
    xgb_device: str = "auto"
    xgb_early_stopping_rounds: int = 100
    xgb_use_class_weights: bool = True


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
