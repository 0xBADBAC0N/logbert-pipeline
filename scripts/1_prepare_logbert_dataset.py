#!/usr/bin/env python3
"""
Prepare Spark executor logs for LogBERT-style modelling.

The script pairs the raw Spark log with its structured variant, extracts
per-job sequences, and produces stratified train/test splits with labels:
    0 -> job finished successfully
    1 -> job failed

Each sample contains the ordered list of EventTemplates encountered between
“Starting job …” and the matching “Job … finished/failed …” line. The output
is written as JSON Lines files that can be consumed by downstream training
and inference scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sklearn.model_selection import train_test_split


START_JOB_RE = re.compile(r"Starting job:\s*(.+)")
GOT_JOB_RE = re.compile(r"Got job (\d+)")
FINISH_JOB_RE = re.compile(r"Job (\d+) finished")
FAIL_JOB_RE = re.compile(r"Job (\d+) failed")
APP_ATTEMPT_RE = re.compile(r"appattempt_(\d+_\d+)_\d+")
APPLICATION_RE = re.compile(r"(application_\d+_\d+)")
FAIL_REASON_RE = re.compile(
    r"(ExecutorLostFailure.*|Container killed by YARN.*|PythonException:.*|User application exited with status.*|Final app status:.*|Task \d+ .*failed.*|FetchFailed.*|Job aborted due to stage failure:.*)",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}")


class JobContext:
    """Collect state for a single Spark job run."""

    __slots__ = (
        "application_id",
        "job_name",
        "job_id",
        "start_time",
        "start_line",
        "events",
        "status",
        "failure_reason",
        "last_reason_line",
    )

    def __init__(self, *, application_id: Optional[str], job_name: str,
                 start_time: str, start_line: int) -> None:
        self.application_id = application_id or ""
        self.job_name = job_name.strip()
        self.job_id: Optional[str] = None
        self.start_time = start_time
        self.start_line = start_line
        self.events: List[str] = []
        self.status: Optional[str] = None
        self.failure_reason: str = ""
        self.last_reason_line: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Spark structured logs into LogBERT-ready dataset."
    )
    parser.add_argument(
        "--structured-log",
        type=Path,
        default=Path("raw_data/Spark/Spark_full.log_structured.csv"),
        help="CSV file with columns LineId,Content,EventId,EventTemplate.",
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=Path("raw_data/Spark/Spark_full.log"),
        help="Original Spark log containing timestamps (used for segmentation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logbert_pipeline/data"),
        help="Target folder for generated dataset files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples assigned to the training split (stratified).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=256,
        help="Maximum number of EventTemplates kept per job sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def extract_timestamp(line: str) -> str:
    match = TIMESTAMP_RE.match(line)
    return match.group(0) if match else ""


def normalize_reason(reason_line: str) -> str:
    # Remove log level prefix for cleaner labels.
    if ": " in reason_line:
        return reason_line.split(": ", maxsplit=1)[1].strip()
    return reason_line.strip()


def read_structured_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield row


def main() -> None:
    args = parse_args()
    if not args.structured_log.exists():
        raise FileNotFoundError(f"Structured log not found: {args.structured_log}")
    if not args.raw_log.exists():
        raise FileNotFoundError(f"Raw log not found: {args.raw_log}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    structured_iter = iter(read_structured_rows(args.structured_log))
    jobs: List[Dict[str, object]] = []
    current_job: Optional[JobContext] = None
    current_application: Optional[str] = None

    with args.raw_log.open("r", encoding="utf-8", errors="ignore") as raw_file:
        for raw_line in raw_file:
            try:
                structured_row = next(structured_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Structured log exhausted before raw log; ensure files are aligned."
                ) from exc
            raw_line = raw_line.rstrip("\n")
            line_id = int(structured_row["LineId"])
            event_template = structured_row["EventTemplate"] or structured_row["Content"]

            # Track the most recent application identifier.
            attempt_match = APP_ATTEMPT_RE.search(raw_line)
            if attempt_match:
                current_application = f"application_{attempt_match.group(1)}"
            else:
                app_match = APPLICATION_RE.search(raw_line)
                if app_match:
                    current_application = app_match.group(1)

            if current_job is None:
                start_match = START_JOB_RE.search(raw_line)
                if start_match:
                    current_job = JobContext(
                        application_id=current_application,
                        job_name=start_match.group(1),
                        start_time=extract_timestamp(raw_line),
                        start_line=line_id,
                    )
                    current_job.events.append(event_template)
                continue

            # Append template to the active job, respecting max length.
            if len(current_job.events) < args.max_events:
                current_job.events.append(event_template)

            # Update job identifier as soon as it becomes available.
            if current_job.job_id is None:
                got_match = GOT_JOB_RE.search(raw_line)
                if got_match:
                    current_job.job_id = got_match.group(1)

            # Capture potential failure explanations.
            fail_match = FAIL_REASON_RE.search(raw_line)
            if fail_match:
                current_job.last_reason_line = fail_match.group(0)

            finish_match = FINISH_JOB_RE.search(raw_line)
            fail_job_match = FAIL_JOB_RE.search(raw_line)

            if finish_match or fail_job_match:
                job_id = (finish_match or fail_job_match).group(1)
                if current_job.job_id is None:
                    current_job.job_id = job_id

                if fail_job_match:
                    current_job.status = "failed"
                    current_job.failure_reason = normalize_reason(current_job.last_reason_line)
                    label = 1
                else:
                    current_job.status = "finished"
                    label = 0

                job_entry = {
                    "sequence_id": len(jobs),
                    "application_id": current_job.application_id,
                    "job_id": current_job.job_id,
                    "job_name": current_job.job_name,
                    "status": current_job.status,
                    "label": label,
                    "start_time": current_job.start_time,
                    "end_time": extract_timestamp(raw_line),
                    "start_line": current_job.start_line,
                    "end_line": line_id,
                    "num_events": len(current_job.events),
                    "events": current_job.events,
                    "failure_reason": current_job.failure_reason,
                }
                jobs.append(job_entry)
                current_job = None

    if current_job is not None:
        print(
            f"Warning: dropping incomplete job starting at line {current_job.start_line}",
            file=sys.stderr,
        )

    if not jobs:
        raise RuntimeError("No jobs were extracted – check log format or paths.")

    labels = [entry["label"] for entry in jobs]
    fail_count = sum(labels)
    success_count = len(jobs) - fail_count

    if len(set(labels)) < 2:
        raise RuntimeError("Dataset lacks both successful and failed samples.")

    test_ratio = 1.0 - args.train_ratio
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    indices = list(range(len(jobs)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    def write_jsonl(path: Path, data_indices: Iterable[int]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for idx in data_indices:
                json.dump(jobs[idx], handle)
                handle.write("\n")

    all_path = args.output_dir / "all_jobs.jsonl"
    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    write_jsonl(all_path, indices)
    write_jsonl(train_path, train_indices)
    write_jsonl(test_path, test_indices)

    metadata = {
        "num_samples": len(jobs),
        "num_success": success_count,
        "num_failed": fail_count,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "train_ratio": args.train_ratio,
        "max_events": args.max_events,
        "label_mapping": {"finished": 0, "failed": 1},
    }
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"Wrote {len(jobs)} sequences to {args.output_dir}")
    print(f"  Success: {success_count}, Failed: {fail_count}")
    print(f"  Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")


if __name__ == "__main__":
    main()
