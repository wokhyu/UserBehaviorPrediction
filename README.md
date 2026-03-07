# User Behavior Prediction

Pipeline huan luyen va suy luan cho bai toan du doan hanh vi nguoi dung (multi-target classification) tu chuoi su kien, da duoc refactor tu notebook sang cau truc module `src/`.

Du an ho tro 3 model type:

- `transformer` (mac dinh)
- `bilstm_gcn`
- `xgboost`

## 1. Muc tieu bai toan

Input moi mau la mot chuoi hanh dong user (`feature_1..feature_37`) kem `id`.

Output la 6 thuoc tinh can du doan:

- `attr_1`
- `attr_2`
- `attr_3`
- `attr_4`
- `attr_5`
- `attr_6`

Day la bai toan multi-task classification: 1 input, 6 dau ra phan loai.

## 2. Cau truc du lieu

Thu muc `data/`:

```text
data/
├── X_train.csv
├── Y_train.csv
├── X_val.csv
├── Y_val.csv
└── X_test.csv
```

Schema chinh:

- `X_*.csv`: `id`, `feature_1..feature_37`
- `Y_train.csv`, `Y_val.csv`: `id`, `attr_1..attr_6`

## 3. Kien truc du an

```text
src/
├── config.py          # Cau hinh, default model name, ATTR columns
├── preprocessing.py   # Fit/transform sequence + label encoder + aux features
├── dataset.py         # Torch Dataset
├── model.py           # Transformer, BiLSTM+GCN, XGBoost wrapper
├── loss.py            # FocalLoss va multitask loss
├── train.py           # Train pipeline
├── evaluate.py        # Validation metrics
├── inference.py       # Predict test va xuat submission
└── utils.py           # Seed, device resolve, checkpoint, npz I/O

artifacts/
├── preprocessor.pt
├── processed/
│   ├── train_processed.npz
│   └── val_processed.npz
├── best_model_transformer.pth
├── best_model_bilstm_gcn.pth
└── best_model_xgboost.pkl
```

## 4. Pipeline tong quan

1. Doc `X_train/Y_train`, `X_val/Y_val`.
2. `BehaviorPreprocessor.fit(...)` tren train:
- Build vocabulary cho action sequence.
- Build label encoders cho `attr_1..attr_6`.
3. Transform train/val thanh:
- sequence da pad (`max_len`)
- mask
- aux features (6 dac trung thong ke)
- labels encoded
4. Luu processed arrays vao `artifacts/processed/*.npz`.
5. Train model theo `--model-type`.
6. Danh gia tren validation va luu best model.

## 5. Chong data leakage

Du an da tach ro fit/transform de tranh leakage:

- Chi fit preprocessor tren `X_train`, `Y_train`.
- Validation/Test chi transform, khong fit lai vocab/encoder.
- `preprocessor.pt` duoc tai su dung cho evaluate/inference.
- `transform_labels(..., strict=True)` se bao loi neu gap label moi o validation.

## 6. Cai dat moi truong

Yeu cau: Python 3.10+.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. Huan luyen

Train Transformer (mac dinh):

```bash
python -m src.train --data-dir data --artifact-dir artifacts --model-type transformer --device auto
```

Train BiLSTM+GCN:

```bash
python -m src.train --data-dir data --artifact-dir artifacts --model-type bilstm_gcn --device auto
```

Train XGBoost:

```bash
python -m src.train --data-dir data --artifact-dir artifacts --model-type xgboost --device auto
```

Luu y:

- `--device auto` tu dong chon `cuda` neu co, nguoc lai dung `cpu`.
- Voi xgboost, model duoc luu dang `.pkl`.
- Neu khong set `--model-name`, he thong tu chon ten model theo model type.

## 8. Danh gia validation

Danh gia theo model mac dinh (tu `--model-type`):

```bash
python -m src.evaluate --data-dir data --artifact-dir artifacts --model-type transformer --device auto
```

Chi ro model path thu cong:

```bash
python -m src.evaluate \
  --data-dir data \
  --artifact-dir artifacts \
  --model-path artifacts/best_model_xgboost.pkl \
  --preprocessor-path artifacts/preprocessor.pt \
  --model-type xgboost
```

Metrics in ra:

- `exact_match`
- `macro_f1` cho tung `attr_1..attr_6`
- `avg macro_f1`

## 9. Inference va tao submission

```bash
python -m src.inference \
  --data-dir data \
  --artifact-dir artifacts \
  --model-type transformer \
  --output submission.csv \
  --device auto
```

File output co 7 cot dung format:

```text
id,attr_1,attr_2,attr_3,attr_4,attr_5,attr_6
```

## 10. Tham so CLI quan trong

`src.train`:

- `--model-type {transformer,bilstm_gcn,xgboost}`
- `--epochs`
- `--train-batch-size`
- `--val-batch-size`
- `--lr`
- `--max-len`
- `--model-name`
- `--xgb-*` (cac tham so rieng cho xgboost)

`src.evaluate`:

- `--model-type {auto,transformer,bilstm_gcn,xgboost}`
- `--model-path`
- `--preprocessor-path`
- `--batch-size`

`src.inference`:

- `--model-type {auto,transformer,bilstm_gcn,xgboost}`
- `--model-path`
- `--preprocessor-path`
- `--output`
- `--batch-size`

## 11. Reproducibility

Du an co `set_seed(...)` va che do deterministic cho torch de giam sai lech giua cac lan chay.

Neu can ket qua on dinh hon, giu nguyen:

- `--seed` co dinh
- cung model type
- cung phien ban package trong `requirements.txt`

## 12. Troubleshooting nhanh

- Loi `Checkpoint not found`: kiem tra `--model-path` hoac da train model chua.
- Loi `Preprocessor artifact not found`: kiem tra `artifacts/preprocessor.pt`.
- CUDA khong san sang: script tu dong fallback CPU khi `--device auto`.
- Dung model cu (`best_model_cpu.pth`): nen train lai hoac chi ro `--model-path` phu hop.

## 13. Giay phep

MIT License.
