#!/usr/bin/env python3
"""
Run inference with the trained LogBERT classifier.

Reads JSONL sequences (same structure as produced by prepare_logbert_dataset.py),
loads a fine-tuned model, and writes predictions with failure probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


LOGGER = logging.getLogger("logbert-predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Spark job outcomes with a LogBERT model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("logbert_pipeline/model"),
        help="Directory containing the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("logbert_pipeline/data/test.jsonl"),
        help="JSONL file with entries (requires an 'events' field).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("logbert_pipeline/predictions.jsonl"),
        help="Destination for JSONL predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length per sequence.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"No data found in {path}")
    return entries


def events_to_text(events: List[str]) -> str:
    return " [SEP] ".join(event.strip() for event in events if event)


class PredictionDataset(Dataset):
    def __init__(self, entries: List[Dict[str, object]], tokenizer, max_length: int):
        self.entries = entries
        texts = [events_to_text(e["events"]) for e in entries]
        self.encodings = tokenizer(
            texts, truncation=True, padding=False, max_length=max_length
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    entries = load_entries(args.input_jsonl)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    LOGGER.info("Loading model from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = PredictionDataset(entries, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    LOGGER.info("Running inference on %d sequences …", len(dataset))
    predictions: List[Dict[str, object]] = []
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        offset = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = softmax(logits)
            fail_probs = probs[:, 1].cpu().tolist()
            preds = logits.argmax(dim=-1).cpu().tolist()

            for i, (pred_label, fail_prob) in enumerate(zip(preds, fail_probs)):
                entry_idx = offset + i
                source = entries[entry_idx].copy()
                source["predicted_label"] = int(pred_label)
                source["predicted_status"] = "failed" if pred_label == 1 else "finished"
                source["failure_probability"] = float(fail_prob)
                predictions.append(source)
            offset += len(preds)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for item in predictions:
            json.dump(item, handle)
            handle.write("\n")

    LOGGER.info("Saved predictions to %s", args.output_jsonl)


if __name__ == "__main__":
    main()
