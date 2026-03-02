# User Behavior Prediction (Transformer + Multi-head)

Refactor từ notebook sang cấu trúc module `src/`, đồng thời tách rõ pipeline để tránh leak dữ liệu validation.

## Cấu trúc dự án

```text
src/
 ├── config.py
 ├── preprocessing.py
 ├── dataset.py
 ├── model.py
 ├── loss.py
 ├── train.py
 ├── evaluate.py
 ├── inference.py
 └── utils.py
data/
 ├── X_train.csv
 ├── Y_train.csv
 ├── X_val.csv
 ├── Y_val.csv
 └── X_test.csv
```

## Nguyên tắc chống data leakage

- `BehaviorPreprocessor.fit(...)` chỉ fit trên `X_train` và `Y_train`.
- Validation/Test chỉ gọi `transform(...)`, không fit lại vocab hoặc label encoder.
- Artifacts tiền xử lý được lưu riêng (`artifacts/preprocessor.pt`) và tái sử dụng cho evaluate/inference.

## Cài đặt

Yêu cầu Python 3.10+.

```bash
pip install -U pandas numpy torch scikit-learn
```

## Train model

Từ thư mục gốc project:

```bash
python -m src.train --data-dir data --artifact-dir artifacts --device cpu
```

Kết quả:

- Model tốt nhất: `artifacts/best_model_cpu.pth`
- Artifacts tiền xử lý: `artifacts/preprocessor.pt`

## Evaluate trên validation

```bash
python -m src.evaluate --data-dir data --artifact-dir artifacts --device cpu
```

In ra:

- `exact_match`
- `macro_f1` cho từng `attr_1..attr_6`

## Predict test và xuất CSV submission (7 cột)

```bash
python -m src.inference --data-dir data --artifact-dir artifacts --output submission.csv --device cpu
```

File output `submission.csv` có đúng 7 cột:

```text
id,attr_1,attr_2,attr_3,attr_4,attr_5,attr_6
```

## Tuỳ chọn tham số nhanh

- Train:
  - `--epochs`
  - `--train-batch-size`
  - `--val-batch-size`
  - `--lr`
  - `--max-len`
- Evaluate:
  - `--model-path`
  - `--preprocessor-path`
- Inference:
  - `--model-path`
  - `--preprocessor-path`
  - `--output`
