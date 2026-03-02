from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ATTR_COLUMNS, TrainConfig
from src.dataset import BehaviorDataset
from src.evaluate import build_model_from_checkpoint
from src.preprocessing import BehaviorPreprocessor
from src.utils import load_checkpoint


def predict_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for seq, mask, aux, ids in loader:
            seq = seq.to(device)
            mask = mask.to(device)
            aux = aux.to(device)

            outputs = model(seq, mask, aux)
            preds = [torch.argmax(logits, dim=1) for logits in outputs]
            stacked = torch.stack(preds, dim=1)

            all_preds.append(stacked.cpu().numpy())
            all_ids.extend(ids)

    pred_array = np.concatenate(all_preds, axis=0)
    return np.asarray(all_ids), pred_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on test set and export submission CSV")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--preprocessor-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("team_name.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(data_dir=args.data_dir, model_dir=args.artifact_dir, device=args.device)

    preprocessor_path = args.preprocessor_path or (config.model_dir / config.preprocessor_name)
    model_path = args.model_path or (config.model_dir / config.model_name)

    x_test = pd.read_csv(config.data_dir / "X_test.csv")

    preprocessor = BehaviorPreprocessor.load(preprocessor_path)
    seqs, masks, aux, ids = preprocessor.transform_features(x_test)

    test_dataset = BehaviorDataset(sequences=seqs, masks=masks, aux_features=aux, ids=ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(config.device)
    checkpoint_payload = load_checkpoint(model_path, map_location=config.device)
    model = build_model_from_checkpoint(checkpoint_payload, preprocessor, config, device)

    pred_ids, pred_encoded = predict_test(model, test_loader, device)
    pred_decoded = preprocessor.decode_predictions(pred_encoded).astype(np.int64)

    submission_df = pd.DataFrame(
        {
            "id": pred_ids,
            "attr_1": pred_decoded[:, 0],
            "attr_2": pred_decoded[:, 1],
            "attr_3": pred_decoded[:, 2],
            "attr_4": pred_decoded[:, 3],
            "attr_5": pred_decoded[:, 4],
            "attr_6": pred_decoded[:, 5],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.output, index=False)
    print(f"Saved prediction file: {args.output}")
    print("Columns:", list(submission_df.columns))


if __name__ == "__main__":
    main()
