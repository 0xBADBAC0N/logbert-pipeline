#!/usr/bin/env python3
"""
Download a Hugging Face Transformers checkpoint for offline use.

Usage example:
    python download_checkpoint.py \
        --model bert-base-uncased \
        --target-dir ../models/bert-base-uncased

Once finished, point the training script to the target directory via
    --model-name <target-dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


LOGGER = logging.getLogger("download-checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a model checkpoint from Hugging Face for local training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Model repo ID on Hugging Face (e.g. bert-base-uncased).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("../models/bert-base-uncased"),
        help="Directory to store the downloaded snapshot.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision (tag, branch, or commit hash).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token. If omitted, uses cached login or anonymous access.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if the target directory already exists.",
    )
    parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=None,
        help="Optional list of glob patterns to limit downloaded files.",
    )
    parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=None,
        help="Optional list of glob patterns to exclude files.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )


def ensure_login(token: Optional[str]) -> None:
    if token:
        LOGGER.info("Logging into Hugging Face Hub with provided token.")
        HfApi().set_access_token(token)


def main() -> None:
    args = parse_args()
    configure_logging()

    target_dir = args.target_dir.resolve()
    if target_dir.exists() and args.skip_existing:
        LOGGER.info("Target directory %s already exists; skipping download.", target_dir)
        return

    ensure_login(args.token)

    LOGGER.info("Downloading %s to %s", args.model, target_dir)
    try:
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=args.include_patterns,
            ignore_patterns=args.exclude_patterns,
        )
    except HfHubHTTPError as exc:
        LOGGER.error("Download failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Checkpoint downloaded successfully. Directory ready for training.")


if __name__ == "__main__":
    main()
