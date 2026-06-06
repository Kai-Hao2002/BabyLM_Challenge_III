#!/usr/bin/env python3
import csv
from pathlib import Path
import argparse


def find_best_row(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in reversed(rows):
        if row.get("validation_loss", "").strip() and row.get("checkpoint_path", "").strip():
            return row, "FINAL_CHECKPOINT"

    for row in reversed(rows):
        if row.get("validation_loss", "").strip():
            return row, "LAST_VALIDATION"

    return None, "NO_VALIDATION_ROW"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Output root folder, e.g. outputs/14k_baseline_chunk_10M")
    args = parser.parse_args()

    root = Path(args.root)
    csv_files = sorted(root.glob("*/training_log.csv"))

    if not csv_files:
        print(f"No training_log.csv found under {root}")
        return

    print(
        "experiment,row_type,epoch,global_step,train_loss,validation_loss,"
        "adjusted_seen_tokens_total,checkpoint_path"
    )

    for csv_path in csv_files:
        row, row_type = find_best_row(csv_path)
        exp_name = csv_path.parent.name

        if row is None:
            print(f"{exp_name},{row_type},,,,,,")
            continue

        print(
            f"{exp_name},"
            f"{row_type},"
            f"{row['epoch']},"
            f"{row['global_step']},"
            f"{row['train_loss']},"
            f"{row['validation_loss']},"
            f"{row['adjusted_seen_tokens_total']},"
            f"{row.get('checkpoint_path', '')}"
        )


if __name__ == "__main__":
    main()