# LogBERT Pipeline

This folder contains a minimal workflow to prepare Spark executor logs for LogBERT-style modelling, train a classifier, and run predictions.

## Setup

```bash
unzip raw_data/raw_data.zip -d raw_data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The requirements file targets the CUDA 12.1 wheels from the official PyTorch index. Ensure your NVIDIA driver supports CUDA ≥ 12.1. If you need a different CUDA runtime (e.g. cu118) replace the `--extra-index-url` and `torch==…` entries accordingly.

## Data preparation

```bash
source .venv/bin/activate
python scripts/1_prepare_logbert_dataset.py \
    --structured-log raw_data/Spark/Spark_full.log_structured.csv \
    --raw-log raw_data/Spark/Spark_full.log \
    --output-dir data
```

This generates `data/all_jobs.jsonl`, `data/train.jsonl`, `data/test.jsonl`, and a `metadata.json` summary.

## Training

Download a BERT checkpoint (e.g. `bert-base-uncased`) if you are working offline and pass its local path via `--model-name`. Then run:

```bash
python scripts/2_train_logbert.py \
    --train-data data/train.jsonl \
    --eval-data data/test.jsonl \
    --model-name bert-base-uncased \
    --output-dir model
```

Adjust hyperparameters such as `--epochs`, `--batch-size`, or `--max-length` as needed.

## Prediction

```bash
python scripts/3_predict_logbert.py \
    --model-dir model \
    --input-jsonl data/test.jsonl \
    --output-jsonl predictions.jsonl
```

Each output record contains the original metadata, the predicted status, and the failure probability.
