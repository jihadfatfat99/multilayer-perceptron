"""
Split dataset into training and validation sets (80/20 split).
Simple shuffling and splitting without any preprocessing.
"""

import argparse
import csv
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split data into train/validation sets")
    parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    parser.add_argument('--training', type=Path, default=Path('training.csv'),
                        help='Output training CSV (default: training.csv)')
    parser.add_argument('--validation', type=Path, default=Path('validation.csv'),
                        help='Output validation CSV (default: validation.csv)')

    args = parser.parse_args()

    # Validate --input argument
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() != '.csv':
        raise ValueError(f"Input file must end with .csv: {args.input}")

    # Validate --training argument
    if args.training.suffix.lower() != '.csv':
        raise ValueError(f"Training output file must end with .csv: {args.training}")

    # Validate --validation argument
    if args.validation.suffix.lower() != '.csv':
        raise ValueError(f"Validation output file must end with .csv: {args.validation}")

    print(f"Loading data from: {args.input}")

    # Load all rows
    rows = []
    with open(args.input, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    total_samples = len(rows)
    print(f"Total samples loaded: {total_samples}")

    if total_samples > 0:
        num_features = len(rows[0]) - 1 if len(rows[0]) < 32 else len(rows[0]) - 2
        print(f"Features per sample: {num_features}")

    # Shuffle
    print("Shuffling data with random permutation...")
    rng = np.random.default_rng(42)
    indices = rng.permutation(total_samples)

    # Split 80/20
    train_size = int(total_samples * 0.8)
    val_size = total_samples - train_size

    print(f"Split ratio: 80% train / 20% validation")
    print(f"Training set size: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    print(f"Validation set size: {val_size} samples ({val_size/total_samples*100:.1f}%)")

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]

    # Save training set
    with open(args.training, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    # Save validation set
    with open(args.validation, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_rows)

    print(f"\nSplit complete:")
    print(f"  Training samples:   {len(train_rows)} (80.0%)")
    print(f"  Validation samples: {len(val_rows)} (20.0%)")
    print(f"\nSaved to:")
    print(f"  {args.training}")
    print(f"  {args.validation}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
