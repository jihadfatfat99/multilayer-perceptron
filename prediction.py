"""
Standalone prediction script.
Loads model from JSON (including topology and normalization parameters).
Makes predictions and displays results with cross-entropy loss.
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from network import NeuralNetwork, TeeOutput
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
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument('--model', type=Path, required=True, help='Model JSON file')
    parser.add_argument('--input', type=Path, required=True, help='CSV file for prediction')

    args = parser.parse_args()

    # Redirect output to both console and file
    tee = TeeOutput('prediction.txt')
    sys.stdout = tee

    # Validate --model argument
    if not args.model.exists():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")
    if args.model.suffix.lower() != '.json':
        raise ValueError(f"Model file must end with .json: {args.model}")

    # Validate --input argument
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.input.suffix.lower() != '.csv':
        raise ValueError(f"Input file must end with .csv: {args.input}")

    # Load model from JSON
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Model file: {args.model}")

    with open(args.model, 'r') as f:
        model_data = json.load(f)

    # Extract model components
    architecture = model_data['architecture']
    weights = [np.array(w) for w in model_data['weights']]
    biases = [np.array(b) for b in model_data['biases']]
    mean = np.array(model_data['normalization']['mean'])
    std = np.array(model_data['normalization']['std'])

    print(f"Architecture: {' -> '.join(map(str, architecture))}")
    print(f"Total layers: {len(architecture)}")
    print("Model loaded successfully!")

    # Create NeuralNetwork and load weights
    network = NeuralNetwork(architecture=architecture, random_seed=42)
    network.weights = weights
    network.biases = biases

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    print(f"Data file: {args.input}")
    X_raw, y_true = load_csv_data(args.input)

    print(f"Number of samples:  {len(X_raw):>6}")
    print(f"Number of features: {X_raw.shape[1]:>6}")

    # Normalize using training statistics
    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)
    print("Normalizing features using training statistics...")
    X_normalized = (X_raw - mean) / std

    # Make predictions
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    predictions = network.predict(X_normalized)

    predicted_classes = network.predict_classes(X_normalized)
    confidences = np.max(predictions, axis=1)

    # Compute loss
    y_true_onehot = np.eye(2)[y_true]
    loss = network.compute_loss(predictions, y_true_onehot)

    # Compute accuracy
    accuracy = network.evaluate_accuracy(X_normalized, y_true)

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"{'Sample':<8} | {'True Label':<17} | {'Predicted':<17} | {'Confidence':<10} | {'Probabilities':<22} | {'Status'}")
    print("-" * 70)

    label_names = {0: 'Benign (B)', 1: 'Malignant (M)'}

    # Show all predictions
    for i in range(len(predictions)):
        pred_class = predicted_classes[i]
        true_class = y_true[i]
        conf = confidences[i]
        prob_b, prob_m = predictions[i]

        status = "✓" if pred_class == true_class else "✗"

        print(f"{i+1:<8} | {label_names[true_class]:<17} | {label_names[pred_class]:<17} | "
              f"{conf:>10.3f} | B={prob_b:.3f} M={prob_m:.3f} | {status}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total Samples:      {len(predictions)}")
    print(f"Cross-Entropy Loss: {loss:.4f}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    true_positives = np.sum((predicted_classes == 1) & (y_true == 1))
    true_negatives = np.sum((predicted_classes == 0) & (y_true == 0))
    false_positives = np.sum((predicted_classes == 1) & (y_true == 0))
    false_negatives = np.sum((predicted_classes == 0) & (y_true == 1))

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"{'':>20} | Predicted Benign | Predicted Malignant")
    print("-" * 70)
    print(f"{'Actual Benign':>20} | {true_negatives:>16} | {false_positives:>19}")
    print(f"{'Actual Malignant':>20} | {false_negatives:>16} | {true_positives:>19}")

    # Additional metrics
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")

    if true_positives + false_positives > 0 and true_positives + false_negatives > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

    print("=" * 70)

    # Close the tee output and restore stdout
    sys.stdout = tee.terminal
    tee.close()
    print("\nPrediction output saved to: prediction.txt")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")

