import random
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    # CUBLAS determinism flag for CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only keeps training running if an op has no deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_torch_device(requested: str = "auto") -> torch.device:
    value = (requested or "auto").strip().lower()
    if value in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "gpu":
        value = "cuda"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def save_checkpoint(path: Path, state_dict: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def save_processed_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            # Convert object arrays (commonly string ids) to fixed-width unicode for safe npz IO.
            try:
                arr = arr.astype("U")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Array '{key}' has object dtype and cannot be serialized safely."
                ) from exc
        normalized[key] = arr

    np.savez_compressed(path, **normalized)


def load_processed_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
