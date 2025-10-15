"""
Standalone training script for neural network with ReLU activation.
Displays epoch-by-epoch results and implements early stopping.
Saves model topology and normalization parameters to JSON.
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from network import NeuralNetwork, TeeOutput
import matplotlib.pyplot as plt
import sys


def load_csv_data(filepath):
    """Load CSV and return features and labels."""
    features, labels = [], []
    with open(filepath, 'r', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 32:
                label_raw, feat_raw = row[-1], row[:-1]
            else:
                label_raw, feat_raw = row[1], row[2:]

            lbl = 1 if str(label_raw).strip().upper() in {'M', '1'} else 0
            features.append([float(x) for x in feat_raw])
            labels.append(lbl)

    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.int64)




def main():
    parser = argparse.ArgumentParser(description="Train neural network with ReLU")
    parser.add_argument('--training', type=Path, required=True, help='Training CSV file')
    parser.add_argument('--validation', type=Path, required=True, help='Validation CSV file')
    parser.add_argument('--layer', nargs='+', type=int, default=[24, 12],
                        help='Hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0256, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--output', type=Path, default=Path('model.json'),
                        help='Output JSON file')
    parser.add_argument('--model_name', type=str, required=True, help='model json file name')

    args = parser.parse_args()

    # Redirect output to both console and file
    tee = TeeOutput('training.txt')
    sys.stdout = tee

    # Validate --training argument
    if not args.training.exists():
        raise FileNotFoundError(f"Training file does not exist: {args.training}")
    if args.training.suffix.lower() != '.csv':
        raise ValueError(f"Training file must end with .csv: {args.training}")

    # Validate --validation argument
    if not args.validation.exists():
        raise FileNotFoundError(f"Validation file does not exist: {args.validation}")
    if args.validation.suffix.lower() != '.csv':
        raise ValueError(f"Validation file must end with .csv: {args.validation}")

    # Validate --layer argument
    if len(args.layer) <= 1:
        raise ValueError(f"At least two hidden layers size must be specified")
    if any(size <= 0 for size in args.layer):
        raise ValueError(f"Hidden layer sizes must be positive integers: {args.layer}")

    # Validate --epochs argument
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be a positive integer: {args.epochs}")

    # Validate --batch_size argument
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer: {args.batch_size}")

    # Validate --learning_rate argument
    if args.learning_rate <= 0:
        raise ValueError(f"Learning rate must be a positive number: {args.learning_rate}")

    # Validate --patience argument
    if args.patience <= 0:
        raise ValueError(f"Patience must be a positive integer: {args.patience}")

    # Validate --output argument
    if args.output.suffix.lower() != '.json':
        raise ValueError(f"Output file must end with .json: {args.output}")

    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    X_train_raw, y_train = load_csv_data(args.training)
    X_valid_raw, y_valid = load_csv_data(args.validation)

    print(f"Training samples:   {X_train_raw.shape[0]:>6}")
    print(f"Validation samples: {X_valid_raw.shape[0]:>6}")
    print(f"Features:           {X_train_raw.shape[1]:>6}")

    # Normalize
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    print("Normalizing features...")
    mu = np.mean(X_train_raw, axis=0, keepdims=True)
    std = np.std(X_train_raw, axis=0, keepdims=True) + 1e-8

    X_train = (X_train_raw - mu) / std
    X_valid = (X_valid_raw - mu) / std

    # One-hot encode
    print("Creating one-hot encoded labels...")
    y_train_oh = np.eye(2)[y_train]
    y_valid_oh = np.eye(2)[y_valid]

    # Build architecture
    input_dim = X_train.shape[1]
    full_arch = [input_dim] + args.layer + [2]

    print("\n" + "=" * 70)
    print("NETWORK ARCHITECTURE")
    print("=" * 70)
    print(f"Architecture: {' -> '.join(map(str, full_arch))}")
    print(f"Total layers: {len(full_arch)}")
    print(f"Hidden layers: {len(args.layer)}")

    # Initialize neural network
    print("\n" + "=" * 70)
    print("INITIALIZING NETWORK")
    print("=" * 70)
    network = NeuralNetwork(architecture=full_arch, random_seed=42)
    print("Network initialized with Xavier/Glorot initialization")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patience:      {args.patience}")
    print("=" * 70)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    rng = np.random.default_rng(42)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_weights = None
    best_biases = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Shuffle and create batches
        indices = rng.permutation(len(X_train))

        for start in range(0, len(X_train), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            network.backward_pass(
                X_train[batch_idx],
                y_train_oh[batch_idx],
                args.learning_rate
            )

        # Evaluate
        train_pred = network.predict(X_train)
        valid_pred = network.predict(X_valid)

        train_loss = network.compute_loss(train_pred, y_train_oh)
        valid_loss = network.compute_loss(valid_pred, y_valid_oh)

        train_acc = network.evaluate_accuracy(X_train, y_train)
        valid_acc = network.evaluate_accuracy(X_valid, y_valid)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)

        # Display
        pad = len(str(args.epochs))
        print(f"Epoch {epoch:>{pad}}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {valid_acc:.3f}")

        # Early stopping logic: update best loss and track patience
        if valid_loss < best_val_loss:
            # New best loss found - update and reset patience counter
            best_val_loss = valid_loss
            best_val_acc = valid_acc
            best_weights = [W.copy() for W in network.weights]
            best_biases = [b.copy() for b in network.biases]
            best_epoch = epoch
            patience_counter = 0
        else:
            # No improvement - increment patience counter
            patience_counter += 1

        # Check if patience limit is reached
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"No improvement for {args.patience} consecutive epochs")
            print(f"Best validation loss: {best_val_loss:.4f} (at epoch {best_epoch})")
            break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Plot training history
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    print("Creating training history visualizations...")
    epochs_range = range(1, len(train_losses) + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1.plot(epochs_range, train_losses, label='Training Loss', linewidth=2, color='#2E86AB')
    ax1.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2, linestyle='--', color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Plot 2: Accuracy curves
    ax2.plot(epochs_range, train_accs, label='Training Accuracy', linewidth=2, color='#2E86AB')
    ax2.plot(epochs_range, val_accs, label='Validation Accuracy', linewidth=2, linestyle='--', color='#A23B72')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()

    # Save model to JSON (using best weights, not final weights)
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"Using weights from epoch {best_epoch} (best validation loss)")

    model_data = {
        'architecture': full_arch,
        'weights': [W.tolist() for W in best_weights],
        'biases': [b.tolist() for b in best_biases],
        'normalization': {
            'mean': mu.tolist(),
            'std': std.tolist()
        }
    }

    with open(args.output, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Save training history for later comparison
    history_data = {
        "model_name": args.model_name,
        "epochs": len(train_losses),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }

    with open(f"history_{args.model_name}.json", "w") as f:
        json.dump(history_data, f, indent=4)

    print("Model saved successfully!")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Training Accuracy:   {train_accs[-1]:.4f} ({train_accs[-1]*100:.2f}%)")
    print(f"Validation Accuracy: {val_accs[-1]:.4f} ({val_accs[-1]*100:.2f}%)")
    print(f"Training Loss:       {train_losses[-1]:.4f}")
    print(f"Validation Loss:     {val_losses[-1]:.4f}")
    print(f"\nBest Validation Loss:     {best_val_loss:.4f} (at epoch {best_epoch})")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) (at epoch {best_epoch})")
    print("=" * 70)

    # Close the tee output and restore stdout
    sys.stdout = tee.terminal
    tee.close()
    print("\nTraining output saved to: training.txt")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
