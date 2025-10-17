#!/usr/bin/env python3
"""
Launch multiple LogBERT fine-tuning runs with different hyper-parameters.

The script distributes runs across the provided GPU list by setting
`CUDA_VISIBLE_DEVICES` per process and persists logs/configuration for each run.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyper-parameter sweep launcher for LogBERT training."
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--train-script",
        type=Path,
        default=script_dir / "2_train_logbert.py",
        help="Path to the training script that should be invoked.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=script_dir.parent / "data" / "train.jsonl",
        help="Path to training dataset JSONL.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=script_dir.parent / "data" / "test.jsonl",
        help="Path to evaluation dataset JSONL.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="models/bert-base-uncased",
        help="Model identifier or path passed to the training script.",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=script_dir.parent / "model",
        help="Directory where run subfolders will be created.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g. '0,1,2,3').",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of concurrent runs. Defaults to the number of GPUs.",
    )
    parser.add_argument(
        "--preset",
        choices=["baseline", "extensive"],
        default="baseline",
        help="Parameter grid preset. 'extensive' expands to thousands of combinations.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        nargs="+",
        default=[5.0],
        help="Epoch values to sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="Per-device batch sizes to sweep.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[3e-5, 2e-5],
        help="Learning rates to sweep.",
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=[384, 512],
        help="Maximum token lengths to sweep.",
    )
    parser.add_argument(
        "--warmup-ratios",
        type=float,
        nargs="+",
        default=[0.1],
        help="Warmup ratios to sweep.",
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=[0.01],
        help="Weight decay values to sweep.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 2025],
        help="Random seeds used for different runs.",
    )
    parser.add_argument(
        "--class-balance",
        action="store_true",
        default=True,
        help="Enable class-balanced loss (default: on).",
    )
    parser.add_argument(
        "--no-class-balance",
        dest="class_balance",
        action="store_false",
        help="Disable class-balanced loss override.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended to the training command.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=script_dir.parent / "model" / "results.csv",
        help="CSV file to append run parameters and metrics.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-run output directories (models/logs). Default removes them after recording metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def build_param_grid(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    sweep_iter = itertools.product(
        args.epochs,
        args.batch_sizes,
        args.learning_rates,
        args.max_lengths,
        args.warmup_ratios,
        args.weight_decays,
        args.seeds,
    )

    for (
        epochs,
        batch_size,
        learning_rate,
        max_length,
        warmup_ratio,
        weight_decay,
        seed,
    ) in sweep_iter:
        yield {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "seed": seed,
        }


def format_run_name(params: Dict[str, object]) -> str:
    return "seed{seed}_lr{learning_rate:g}_bs{batch_size}_ml{max_length}_ep{epochs:g}".format(
        **params
    )


def launch_run(
    gpu_id: str,
    train_script: Path,
    train_data: Path,
    eval_data: Path,
    model_name: str,
    base_output: Path,
    params: Dict[str, object],
    class_balance: bool,
    extra_args: Sequence[str],
    dry_run: bool,
) -> Tuple[subprocess.Popen | None, Path, Path, object]:
    run_name = format_run_name(params)
    output_dir = base_output / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train.log"
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.json"

    cmd = [
        sys.executable,
        str(train_script),
        "--train-data",
        str(train_data),
        "--eval-data",
        str(eval_data),
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--max-length",
        str(params["max_length"]),
        "--warmup-ratio",
        str(params["warmup_ratio"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--seed",
        str(params["seed"]),
    ]

    if class_balance:
        cmd.append("--class-balance")

    cmd.extend(["--metrics-file", str(metrics_path), "--skip-save"])

    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        print(f"[DRY-RUN][GPU {gpu_id}] {' '.join(cmd)}")
        return None, output_dir, metrics_path, None

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "gpu": gpu_id,
                "command": cmd,
                "params": params,
                "class_balance": class_balance,
            },
            handle,
            indent=2,
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=train_script.parent,
    )
    return process, output_dir, metrics_path, log_handle


BASE_COLUMNS = [
    "run",
    "gpu",
    "status",
    "timestamp",
    "preset",
]
PARAM_COLUMNS = [
    "param_epochs",
    "param_batch_size",
    "param_learning_rate",
    "param_max_length",
    "param_warmup_ratio",
    "param_weight_decay",
    "param_seed",
]


def metrics_columns() -> List[str]:
    return [
        "train_epoch",
        "train_loss",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_total_flos",
        "train_global_step",
        "train_loss_epoch",
        "eval_epoch",
        "eval_loss",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "metrics_timestamp",
        "best_model_checkpoint",
    ]


def append_results(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = BASE_COLUMNS + PARAM_COLUMNS + metrics_columns()
    for key in row.keys():
        if key not in cols:
            cols.append(key)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in cols})


def main() -> None:
    args = parse_args()
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No GPU IDs provided via --gpus.")

    max_parallel = args.max_parallel or len(gpu_ids)
    max_parallel = min(max_parallel, len(gpu_ids))

    base_output = args.base_output.resolve()
    base_output.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    model_path = Path(model_name)
    if model_path.exists():
        model_name = str(model_path.resolve())
    else:
        candidate = args.train_script.resolve().parent.parent / model_name
        if candidate.exists():
            model_name = str(candidate.resolve())

    if args.preset == "extensive":
        args.epochs = [4.0, 5.0, 6.0]
        args.batch_sizes = [8, 16, 24]
        args.learning_rates = [5e-5, 3e-5, 2e-5, 1e-5]
        args.max_lengths = [256, 320, 384, 448, 512]
        args.warmup_ratios = [0.05, 0.1, 0.15]
        args.weight_decays = [0.0, 0.01, 0.03, 0.05]
        args.seeds = [13, 37, 42, 128, 314, 733, 1337]

    jobs = list(build_param_grid(args))
    if not jobs:
        print("No parameter combinations generated; exiting.")
        return

    print(f"Planned runs: {len(jobs)} across GPUs {gpu_ids} (max parallel: {max_parallel})")

    if args.dry_run:
        for job in jobs:
            launch_run(
                gpu_ids[0],
                args.train_script.resolve(),
                args.train_data.resolve(),
                args.eval_data.resolve(),
                args.model_name,
                base_output,
                job,
                args.class_balance,
                args.extra_args,
                True,
            )
        return

    pending = jobs.copy()
    running: List[Tuple[subprocess.Popen, str, Path, Path, Dict[str, object], object]] = []
    available = gpu_ids.copy()

    try:
        while pending or running:
            while pending and available and len(running) < max_parallel:
                job = pending.pop(0)
                gpu_id = available.pop(0)
                proc, out_dir, metrics_path, log_handle = launch_run(
                    gpu_id,
                    args.train_script.resolve(),
                    args.train_data.resolve(),
                    args.eval_data.resolve(),
                    model_name,
                    base_output,
                    job,
                    args.class_balance,
                    args.extra_args,
                    False,
                )
                if proc is None:
                    available.append(gpu_id)
                    continue
                print(f"[LAUNCH][GPU {gpu_id}] {out_dir.name}")
                running.append((proc, gpu_id, out_dir, metrics_path, job, log_handle))

            time.sleep(2)

            for proc, gpu_id, out_dir, metrics_path, job, log_handle in list(running):
                retcode = proc.poll()
                if retcode is None:
                    continue
                running.remove((proc, gpu_id, out_dir, metrics_path, job, log_handle))
                available.append(gpu_id)
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                status = "OK" if retcode == 0 else f"FAIL ({retcode})"
                metrics_data: Dict[str, object] = {}
                if retcode == 0 and metrics_path.exists():
                    try:
                        with metrics_path.open("r", encoding="utf-8") as handle:
                            metrics_data = json.load(handle)
                    except json.JSONDecodeError as exc:
                        status = f"METRICS_PARSE_ERROR ({exc})"
                elif retcode == 0:
                    status = "METRICS_MISSING"

                row: Dict[str, object] = {
                    "run": out_dir.name,
                    "gpu": gpu_id,
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat(),
                    "preset": args.preset,
                    "param_epochs": job["epochs"],
                    "param_batch_size": job["batch_size"],
                    "param_learning_rate": job["learning_rate"],
                    "param_max_length": job["max_length"],
                    "param_warmup_ratio": job["warmup_ratio"],
                    "param_weight_decay": job["weight_decay"],
                    "param_seed": job["seed"],
                }
                for key, value in metrics_data.items():
                    row[key] = value
                append_results(args.results_csv.resolve(), row)

                if not args.keep_artifacts:
                    try:
                        shutil.rmtree(out_dir)
                    except OSError:
                        pass

                print(f"[DONE][GPU {gpu_id}] {out_dir.name} -> {status}")
    finally:
        for proc, _, out_dir, _, _, log_handle in running:
            if proc.poll() is None:
                proc.terminate()
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
