# LogBERT Pipeline

This folder contains a minimal workflow to prepare Spark executor logs for LogBERT-style modelling, train a classifier, and run predictions.

## Ubuntu 22.04 (AWS G6) prerequisites

```bash
sudo apt update
sudo apt install -y git git-lfs python3-venv python3-pip unzip
git lfs install --system
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

git clone https://github.com/0xBADBAC0N/logbert-pipeline.git
cd logbert-pipeline
git lfs pull
```

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
    --output-dir data \
    --k-folds 4
```

This generates `data/all_jobs.jsonl`, `data/train.jsonl`, `data/test.jsonl`, and a `metadata.json` summary.
With `--k-folds <n>` additional stratified folds are written to `data/folds/fold_*/{train,eval}.jsonl`.

## Training

Download the base encoder once (skip when `models/bert-base-uncased` already exists):

```bash
python scripts/0_download_checkpoint.py \
    --model bert-base-uncased \
    --target-dir models/bert-base-uncased \
    --skip-existing
```

Then run the fine-tuning script:

```bash
python scripts/2_train_logbert.py \
    --train-data data/train.jsonl \
    --eval-data data/test.jsonl \
    --model-name models/bert-base-uncased \
    --output-dir model \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --max-length 384 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --class-balance
```

Adjust hyperparameters such as `--epochs`, `--batch-size`, or `--max-length` as needed.
`--class-balance` activates inverse-frequency weighting in the loss so the rare failure class contributes proportionally during training.
Optional flags: `--metrics-file <path>` writes a JSON summary of train/eval metrics, and `--skip-save` prevents checkpoint export (useful for large sweeps).
To train on a specific fold, add `--fold-index <k>` (and optionally `--folds-dir` if you stored them elsewhere).

## Prediction

```bash
python scripts/3_predict_logbert.py \
    --model-dir model \
    --input-jsonl data/test.jsonl \
    --output-jsonl predictions.jsonl
```

Each output record contains the original metadata, the predicted status, and the failure probability.

## Hyperparameter sweeps

```bash
python scripts/4_sweep_logbert.py \
    --gpus 0,1,2,3 \
    --max-parallel 4 \
    --learning-rates 3e-5 2e-5 \
    --max-lengths 384 512 \
    --seeds 42 1337 2025 \
    --folds 0 1 2 3
```

Runs are launched with one GPU each; temporary run folders live under `model/seed*_lr*_bs*_ml*_ep*` and are removed once metrics are collected (retain them with `--keep-artifacts`). Use `--dry-run` to inspect commands only or `--extra-args --gradient-accumulation-steps 2` to pass additional flags through.

Each completed run appends its configuration and train/eval metrics to `model/results.csv`. For a broader search space (thousands of combinations) add `--preset extensive`.

## Cleanup

Remove generated datasets and sweep artefacts to free disk space:

```bash
python scripts/5_cleanup_artifacts.py --dry-run   # inspect first
python scripts/5_cleanup_artifacts.py --keep-results
```

Use `--paths <path1> <path2>` to target specific folders (defaults: `data`, `model`).
