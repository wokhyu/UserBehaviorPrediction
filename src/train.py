from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import TrainConfig, ensure_dirs
from src.dataset import BehaviorDataset
from src.evaluate import evaluate_model
from src.loss import FocalLoss, compute_multitask_loss
from src.model import BehaviorModel
from src.preprocessing import BehaviorPreprocessor
from src.utils import save_checkpoint, set_seed


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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--model-name", type=str, default="best_model_cpu.pth")
    parser.add_argument("--preprocessor-name", type=str, default="preprocessor.pt")
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
    )

    ensure_dirs(config.model_dir)
    set_seed(config.seed)

    x_train = pd.read_csv(config.data_dir / "X_train.csv")
    y_train = pd.read_csv(config.data_dir / "Y_train.csv")
    x_val = pd.read_csv(config.data_dir / "X_val.csv")
    y_val = pd.read_csv(config.data_dir / "Y_val.csv")

    preprocessor = BehaviorPreprocessor(max_len=config.max_len)
    preprocessor.fit(x_train=x_train, y_train=y_train)
    preprocessor.save(config.model_dir / config.preprocessor_name)

    train_seq, train_mask, train_aux, _ = preprocessor.transform_features(x_train)
    val_seq, val_mask, val_aux, _ = preprocessor.transform_features(x_val)

    train_labels = preprocessor.transform_labels(y_train, strict=True)
    val_labels = preprocessor.transform_labels(y_val, strict=True)

    train_dataset = BehaviorDataset(train_seq, train_mask, train_aux, labels=train_labels)
    val_dataset = BehaviorDataset(val_seq, val_mask, val_aux, labels=val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)

    device = torch.device(config.device)
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
    model = BehaviorModel(
        vocab_size=preprocessor.vocab_size,
        num_classes_list=preprocessor.num_classes_list,
        **model_hparams,
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
                    "model_state_dict": model.state_dict(),
                    "num_classes_list": preprocessor.num_classes_list,
                    "vocab_size": preprocessor.vocab_size,
                    "model_hparams": model_hparams,
                },
            )
            print(f"  saved best model: {best_path}")

    print("Training completed.")
    print("Best validation exact_match:", round(best_exact, 6))


if __name__ == "__main__":
    main()
