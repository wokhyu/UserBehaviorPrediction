from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import ATTR_COLUMNS, TrainConfig, default_model_name_for_type
from src.dataset import BehaviorDataset
from src.model import BehaviorModelFactory, BehaviorXGBoostModel
from src.preprocessing import BehaviorPreprocessor
from src.utils import load_checkpoint, resolve_torch_device, set_seed


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[float]]:
    model.eval()

    all_preds = [[] for _ in range(6)]
    all_labels = [[] for _ in range(6)]
    exact_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seq, mask, aux, labels in loader:
            seq = seq.to(device)
            mask = mask.to(device)
            aux = aux.to(device)
            labels = labels.to(device)

            outputs = model(seq, mask, aux)
            preds = [torch.argmax(logits, dim=1) for logits in outputs]
            stacked_preds = torch.stack(preds, dim=1)

            exact_correct += (stacked_preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

            for idx in range(6):
                all_preds[idx].extend(preds[idx].cpu().numpy().tolist())
                all_labels[idx].extend(labels[:, idx].cpu().numpy().tolist())

    exact_match = exact_correct / max(total_samples, 1)
    macro_f1 = [
        f1_score(all_labels[idx], all_preds[idx], average="macro", zero_division=0)
        for idx in range(6)
    ]
    return exact_match, macro_f1


def compute_metrics_from_arrays(labels: np.ndarray, preds: np.ndarray) -> Tuple[float, List[float]]:
    exact_match = float((preds == labels).all(axis=1).mean())
    macro_f1 = [
        f1_score(labels[:, idx], preds[:, idx], average="macro", zero_division=0)
        for idx in range(6)
    ]
    return exact_match, macro_f1


def build_model_from_checkpoint(
    checkpoint_payload,
    preprocessor: BehaviorPreprocessor,
    config: TrainConfig,
    device: torch.device,
) -> torch.nn.Module:
    model_hparams = {
        "max_len": config.max_len,
        "embedding_dim": config.embedding_dim,
        "nhead": config.nhead,
        "ff_dim": config.ff_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "aux_dim": config.aux_dim,
        "hidden_dim": config.hidden_dim,
        "branch_dim": config.branch_dim,
    }

    if isinstance(checkpoint_payload, dict) and "model_hparams" in checkpoint_payload:
        model_hparams.update(checkpoint_payload["model_hparams"])

    num_classes_list = preprocessor.num_classes_list
    if isinstance(checkpoint_payload, dict) and "num_classes_list" in checkpoint_payload:
        num_classes_list = checkpoint_payload["num_classes_list"]

    model_type = "transformer"
    if isinstance(checkpoint_payload, dict) and "model_type" in checkpoint_payload:
        model_type = checkpoint_payload["model_type"]

    model = BehaviorModelFactory.create(
        model_type=model_type,
        vocab_size=preprocessor.vocab_size,
        num_classes_list=num_classes_list,
        model_hparams=model_hparams,
    ).to(device)

    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        model.load_state_dict(checkpoint_payload["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_payload)

    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation set")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--preprocessor-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "transformer", "bilstm_gcn", "xgboost"],
    )
    return parser.parse_args()


def resolve_model_type(model_path: Path, checkpoint_payload, user_model_type: str) -> str:
    if user_model_type != "auto":
        return user_model_type
    if model_path.suffix.lower() == ".pkl":
        return "xgboost"
    if isinstance(checkpoint_payload, dict):
        return checkpoint_payload.get("model_type", "transformer")
    return "transformer"


def main() -> None:
    args = parse_args()
    resolved_device = resolve_torch_device(args.device)
    config = TrainConfig(data_dir=args.data_dir, model_dir=args.artifact_dir, device=str(resolved_device))
    set_seed(args.seed)

    preprocessor_path = args.preprocessor_path or (config.model_dir / config.preprocessor_name)
    if args.model_path:
        model_path = args.model_path
    else:
        default_type = "transformer" if args.model_type == "auto" else args.model_type
        model_path = config.model_dir / default_model_name_for_type(default_type)

    x_val = pd.read_csv(config.data_dir / "X_val.csv")
    y_val = pd.read_csv(config.data_dir / "Y_val.csv")

    preprocessor = BehaviorPreprocessor.load(preprocessor_path)
    seqs, masks, aux, _ = preprocessor.transform_features(x_val)
    labels = preprocessor.transform_labels(y_val, strict=True)

    val_dataset = BehaviorDataset(sequences=seqs, masks=masks, aux_features=aux, labels=labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint_payload = None
    if model_path.suffix.lower() != ".pkl":
        checkpoint_payload = load_checkpoint(model_path, map_location=config.device)

    model_type = resolve_model_type(model_path, checkpoint_payload, args.model_type)

    if model_type == "xgboost":
        tab = preprocessor.build_tabular_features(seqs, masks, aux)
        model = BehaviorXGBoostModel.load(model_path)
        preds = model.predict(tab)
        exact_match, macro_f1 = compute_metrics_from_arrays(labels, preds)
    else:
        device = resolve_torch_device(config.device)
        model = build_model_from_checkpoint(checkpoint_payload, preprocessor, config, device)
        exact_match, macro_f1 = evaluate_model(model, val_loader, device)

    print("Validation exact_match:", round(exact_match, 6))
    for idx, score in enumerate(macro_f1, start=1):
        print(f"attr_{idx} macro_f1: {score:.6f}")
    print("avg macro_f1:", round(sum(macro_f1) / len(macro_f1), 6))


if __name__ == "__main__":
    main()
