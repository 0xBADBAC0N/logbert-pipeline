#!/usr/bin/env python3
"""
Remove generated artifacts (datasets, model outputs, sweep results) to free disk space.

By default the script deletes directories under:
  - data/ (excluding raw_data.zip)
  - model/
  - logs and temporary run folders

Use --dry-run to inspect what would be removed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_PATHS = [
    Path("data"),
    Path("model"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated artifacts.")
    parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        default=None,
        help="Specific paths to remove. Defaults to common build directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths without deleting.",
    )
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Preserve model/results.csv even if model/ is deleted.",
    )
    return parser.parse_args()


def gather_targets(base_dir: Path, paths: Iterable[Path]) -> list[Path]:
    resolved_targets = []
    for path in paths:
        target = (base_dir / path).resolve()
        if target.exists():
            resolved_targets.append(target)
    return resolved_targets


def remove_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would remove {path}")
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)
    print(f"Removed {path}")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    defaults = [p for p in DEFAULT_PATHS if not p.name.startswith("raw_data")]
    paths = args.paths if args.paths is not None else defaults
    targets = gather_targets(base_dir, paths)

    if args.keep_results:
        for target in list(targets):
            if target.name == "model":
                results = target / "results.csv"
                if results.exists():
                    tmp = results.parent / "results.csv.tmp"
                    if not args.dry_run:
                        results.rename(tmp)
                    targets.append(tmp)

    for target in targets:
        if args.keep_results and target.name == "results.csv.tmp":
            if args.dry_run:
                print(f"[DRY-RUN] Would restore results.csv to {target.parent}")
            else:
                target.rename(target.parent / "results.csv")
            continue
        remove_path(target, args.dry_run)


if __name__ == "__main__":
    main()
