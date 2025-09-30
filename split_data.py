#!/usr/bin/env python3
"""
Multilayer Perceptron - Data Splitting Program
Splits the breast cancer dataset into training and validation sets
"""

import numpy as np
import pandas as pd
import sys

def validate_csv_extension(filepath):
    """
    Validate that a file has .csv extension
    
    Args:
        filepath: Path to validate
    """
    if not filepath.lower().endswith('.csv'):
        print(f"Error: File '{filepath}' must have .csv extension")
        sys.exit(1)

def load_data(filepath):
    """
    Load the dataset from CSV file
    
    Args:
        filepath: Path to the input CSV file
        
    Returns:
        numpy array: Loaded data
    """
    try:
        # Load data without headers (CSV has no column names)
        data = pd.read_csv(filepath, header=None)
        print(f"Dataset loaded successfully: {data.shape[0]} samples, {data.shape[1]} columns")
        return data.values
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the dataset
    
    Args:
        data: Raw data array
        
    Returns:
        tuple: (features, labels, ids)
    """
    print("Preprocessing data...")
    
    # Extract components
    # Column 0: ID
    # Column 1: Diagnosis (M/B)
    # Columns 2-31: 30 features
    
    ids = data[:, 0]  # ID column
    labels = data[:, 1]  # Diagnosis column
    features = data[:, 2:32].astype(float)  # Feature columns
    
    # Convert labels M/B to 1/0
    labels_encoded = np.where(labels == 'M', 1, 0)
    
    # Count class distribution
    malignant_count = np.sum(labels_encoded)
    benign_count = len(labels_encoded) - malignant_count
    
    print(f"Class distribution:")
    print(f"  Malignant (M): {malignant_count} samples")
    print(f"  Benign (B): {benign_count} samples")
    
    # Normalize features (standardization: mean=0, std=1)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    features_normalized = (features - mean) / std
    
    print(f"Features normalized: mean ≈ 0, std ≈ 1")
    print(f"Feature matrix shape: {features_normalized.shape}")
    
    return features_normalized, labels_encoded, ids

def split_dataset(features, labels, ids, train_ratio=0.8):
    """
    Split dataset into training and validation sets
    
    Args:
        features: Normalized feature matrix
        labels: Encoded labels
        ids: Sample IDs
        train_ratio: Ratio of training data (default 0.8)
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, train_ids, val_ids)
    """
    print(f"Splitting dataset with ratio {train_ratio:.1%} train, {1-train_ratio:.1%} validation")
    
    # Get dataset size
    n_samples = len(features)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    
    # Calculate split point
    split_idx = int(n_samples * train_ratio)
    
    # Split indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Split data
    X_train = features[train_indices]
    y_train = labels[train_indices]
    train_ids = ids[train_indices]
    
    X_val = features[val_indices]
    y_val = labels[val_indices]
    val_ids = ids[val_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Check class distribution in splits
    train_malignant = np.sum(y_train)
    train_benign = len(y_train) - train_malignant
    val_malignant = np.sum(y_val)
    val_benign = len(y_val) - val_malignant
    
    print(f"Training set distribution: {train_malignant} malignant, {train_benign} benign")
    print(f"Validation set distribution: {val_malignant} malignant, {val_benign} benign")
    
    return X_train, y_train, X_val, y_val, train_ids, val_ids

def save_splits(X_train, y_train, X_val, y_val, train_file, val_file, train_ids, val_ids):
    """
    Save training and validation sets to CSV files with IDs
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        train_file: Training output filename
        val_file: Validation output filename
        train_ids: Training sample IDs
        val_ids: Validation sample IDs
    """
    print("Saving split datasets...")
    
    # Combine IDs, labels, and features
    train_data = np.column_stack((train_ids, y_train, X_train))
    val_data = np.column_stack((val_ids, y_val, X_val))
    
    # Save with mixed formatting: integers for first 2 columns, floats for rest
    fmt = '%d,%d' + ',%.6f' * X_train.shape[1]  # ← CHANGE THIS

    # Save to CSV files (no headers)
    np.savetxt(train_file, train_data, delimiter=',', fmt=fmt)
    np.savetxt(val_file, val_data, delimiter=',', fmt=fmt)
    
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")

def main():
    """Main function"""
    # Check for exactly 3 arguments (excluding program name)
    if len(sys.argv) != 4:
        print("Error: Exactly 3 arguments required")
        print("Usage: python split_data.py <input_file> <training_output_file> <validation_output_file>")
        print("Example: python split_data.py data.csv data_training.csv data_validation.csv")
        sys.exit(1)
    
    # Get arguments
    input_file = sys.argv[1]
    train_output = sys.argv[2]
    val_output = sys.argv[3]

    # Validate CSV extensions for output files
    validate_csv_extension(train_output)
    validate_csv_extension(val_output)
    
    # Fixed parameters
    train_ratio = 0.8
    
    print("=" * 60)
    print("MULTILAYER PERCEPTRON - DATA SPLITTING")
    print("=" * 60)
    
    # Load data
    data = load_data(input_file)
    
    # Preprocess data
    features, labels, ids = preprocess_data(data)
    
    # Split dataset
    X_train, y_train, X_val, y_val, train_ids, val_ids = split_dataset(features, labels, ids, train_ratio)
    
    # Save splits
    save_splits(X_train, y_train, X_val, y_val, train_output, val_output, train_ids, val_ids)
    
    print("=" * 60)
    print("DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Next step: Run training with 'python train_mlp.py'")

if __name__ == "__main__":
    main()