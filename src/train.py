from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import TrainConfig, default_model_name_for_type, ensure_dirs
from src.dataset import BehaviorDataset
from src.evaluate import evaluate_model
from src.loss import FocalLoss, compute_multitask_loss
from src.model import BehaviorModelFactory, BehaviorXGBoostModel
from src.preprocessing import BehaviorPreprocessor
from src.utils import (
    load_processed_npz,
    resolve_torch_device,
    save_checkpoint,
    save_processed_npz,
    set_seed,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fns,
    weights,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for seq, mask, aux, labels in loader:
        seq = seq.to(device)
        mask = mask.to(device)
        aux = aux.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(seq, mask, aux)
        loss = compute_multitask_loss(outputs, labels, loss_fns, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def compute_exact_and_macro_f1(labels: torch.Tensor, preds: torch.Tensor) -> tuple[float, List[float]]:
    exact = (preds == labels).all(dim=1).float().mean().item()
    macro_f1 = []
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    for idx in range(6):
        macro_f1.append(f1_score(y_true[:, idx], y_pred[:, idx], average="macro", zero_division=0))
    return exact, macro_f1


def export_processed_splits(
    preprocessor: BehaviorPreprocessor,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    processed_dir: Path,
) -> Dict[str, Path]:
    train_seq, train_mask, train_aux, train_ids = preprocessor.transform_features(x_train)
    val_seq, val_mask, val_aux, val_ids = preprocessor.transform_features(x_val)
    train_labels = preprocessor.transform_labels(y_train, strict=True)
    val_labels = preprocessor.transform_labels(y_val, strict=True)

    train_tab = preprocessor.build_tabular_features(train_seq, train_mask, train_aux)
    val_tab = preprocessor.build_tabular_features(val_seq, val_mask, val_aux)

    train_path = processed_dir / "train_processed.npz"
    val_path = processed_dir / "val_processed.npz"

    save_processed_npz(
        train_path,
        seq=train_seq,
        mask=train_mask,
        aux=train_aux,
        labels=train_labels,
        ids=train_ids,
        tab=train_tab,
    )
    save_processed_npz(
        val_path,
        seq=val_seq,
        mask=val_mask,
        aux=val_aux,
        labels=val_labels,
        ids=val_ids,
        tab=val_tab,
    )
    return {"train": train_path, "val": val_path}


def train_torch_model(
    config: TrainConfig,
    model_type: str,
    preprocessor: BehaviorPreprocessor,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
) -> None:
    train_dataset = BehaviorDataset(
        train_data["seq"],
        train_data["mask"],
        train_data["aux"],
        labels=train_data["labels"],
    )
    val_dataset = BehaviorDataset(
        val_data["seq"],
        val_data["mask"],
        val_data["aux"],
        labels=val_data["labels"],
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

    device = resolve_torch_device(config.device)
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
        "embed_dim": config.embedding_dim,
        "lstm_hidden": config.hidden_dim,
        "gcn_hidden": config.hidden_dim,
        "stat_dim": 6,
        "use_attention": False,
    }

    model = BehaviorModelFactory.create(
        model_type=model_type,
        vocab_size=preprocessor.vocab_size,
        num_classes_list=preprocessor.num_classes_list,
        model_hparams=model_hparams,
    ).to(device)

    loss_fns = [
        FocalLoss(gamma=2.0),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        FocalLoss(gamma=2.0),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
    ]
    weights = [1.5, 1.0, 1.0, 1.5, 1.0, 1.0]

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_exact = -1.0
    best_path = config.model_dir / config.model_name

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fns=loss_fns,
            weights=weights,
            grad_clip=config.grad_clip,
            device=device,
        )

        exact_match, macro_f1 = evaluate_model(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"  train_loss: {train_loss:.6f}")
        print(f"  val_exact_match: {exact_match:.6f}")
        print(f"  val_macro_f1: {[round(x, 6) for x in macro_f1]}")

        if exact_match > best_exact:
            best_exact = exact_match
            save_checkpoint(
                best_path,
                {
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "num_classes_list": preprocessor.num_classes_list,
                    "vocab_size": preprocessor.vocab_size,
                    "model_hparams": model_hparams,
                },
            )
            print(f"  saved best model: {best_path}")

    print("Training completed.")
    print("Best validation exact_match:", round(best_exact, 6))


def train_xgboost_model(
    config: TrainConfig,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
) -> None:
    train_features = train_data["tab"].astype("float32")
    val_features = val_data["tab"].astype("float32")
    train_labels = train_data["labels"].astype("int64")
    val_labels = val_data["labels"].astype("int64")

    xgb_device = config.xgb_device
    if xgb_device == "auto":
        xgb_device = "cuda" if str(config.device).startswith("cuda") else "cpu"

    model = BehaviorXGBoostModel(
        random_state=config.seed,
        n_estimators=config.xgb_n_estimators,
        learning_rate=config.xgb_learning_rate,
        max_depth=config.xgb_max_depth,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        reg_lambda=config.xgb_reg_lambda,
        reg_alpha=config.xgb_reg_alpha,
        min_child_weight=config.xgb_min_child_weight,
        gamma=config.xgb_gamma,
        n_jobs=config.xgb_n_jobs,
        device=xgb_device,
    )
    model.fit(
        train_features,
        train_labels,
        val_features=val_features,
        val_labels=val_labels,
        early_stopping_rounds=config.xgb_early_stopping_rounds,
        use_class_weights=config.xgb_use_class_weights,
    )
    val_preds = model.predict(val_features)

    labels_tensor = torch.from_numpy(val_labels)
    preds_tensor = torch.from_numpy(val_preds)
    exact_match, macro_f1 = compute_exact_and_macro_f1(labels_tensor, preds_tensor)

    best_path = config.model_dir / config.model_name
    model.save(
        best_path,
        extra={
            "model_type": "xgboost",
        },
    )

    print("Training completed.")
    print("Saved XGBoost model:", best_path)
    print("Validation exact_match:", round(exact_match, 6))
    print("Validation macro_f1:", [round(x, 6) for x in macro_f1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior prediction model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformer",
        choices=["transformer", "bilstm_gcn", "xgboost"],
    )
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--preprocessor-name", type=str, default="preprocessor.pt")
    parser.add_argument("--processed-dir-name", type=str, default="processed")
    parser.add_argument("--xgb-n-estimators", type=int, default=350)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=8)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--xgb-use-class-weights", dest="xgb_use_class_weights", action="store_true")
    parser.add_argument("--xgb-no-class-weights", dest="xgb_use_class_weights", action="store_false")
    parser.set_defaults(xgb_use_class_weights=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        model_dir=args.artifact_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_len=args.max_len,
        model_name=args.model_name,
        preprocessor_name=args.preprocessor_name,
        processed_dir_name=args.processed_dir_name,
        model_type=args.model_type,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_max_depth=args.xgb_max_depth,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_gamma=args.xgb_gamma,
        xgb_n_jobs=args.xgb_n_jobs,
        xgb_device=args.xgb_device,
        xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
        xgb_use_class_weights=args.xgb_use_class_weights,
    )

    if not config.model_name:
        config.model_name = default_model_name_for_type(config.model_type)

    resolved_device = resolve_torch_device(config.device)
    config.device = str(resolved_device)
    print(f"Using device: {config.device}")

    ensure_dirs(config.model_dir)
    set_seed(config.seed)

    x_train = pd.read_csv(config.data_dir / "X_train.csv")
    y_train = pd.read_csv(config.data_dir / "Y_train.csv")
    x_val = pd.read_csv(config.data_dir / "X_val.csv")
    y_val = pd.read_csv(config.data_dir / "Y_val.csv")

    preprocessor = BehaviorPreprocessor(max_len=config.max_len)
    preprocessor.fit(x_train=x_train, y_train=y_train)
    preprocessor.save(config.model_dir / config.preprocessor_name)

    processed_dir = config.model_dir / config.processed_dir_name
    ensure_dirs(processed_dir)
    exported_paths = export_processed_splits(
        preprocessor=preprocessor,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        processed_dir=processed_dir,
    )
    print(f"Saved processed train data: {exported_paths['train']}")
    print(f"Saved processed val data: {exported_paths['val']}")

    train_processed = load_processed_npz(exported_paths["train"])
    val_processed = load_processed_npz(exported_paths["val"])

    if config.model_type == "xgboost":
        train_xgboost_model(config, train_processed, val_processed)
    else:
        train_torch_model(
            config=config,
            model_type=config.model_type,
            preprocessor=preprocessor,
            train_data=train_processed,
            val_data=val_processed,
        )


if __name__ == "__main__":
    main()
