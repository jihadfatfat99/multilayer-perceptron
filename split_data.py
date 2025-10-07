#!/usr/bin/env python3
"""
Multilayer Perceptron - Data Splitting Program
Splits the breast cancer dataset into training and validation sets
"""

import numpy as np
import pandas as pd
import sys
import argparse
import os

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


def split_dataset(features, labels, ids, train_ratio=0.8, seed=42):
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

    # Set seed for reproducibility
    np.random.seed(seed)
    
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
    train_malignant = np.sum(y_train == 'M')
    train_benign = np.sum(y_train == 'B')
    val_malignant = np.sum(y_val == 'M')
    val_benign = np.sum(y_val == 'B')
    
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
    
    # Save with mixed formatting: integer for ID, string for label, floats for features
    fmt = '%d,%s' + ',%g' * X_train.shape[1]

    # Save to CSV files (no headers)
    np.savetxt(train_file, train_data, delimiter=',', fmt=fmt)
    np.savetxt(val_file, val_data, delimiter=',', fmt=fmt)
    
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")

def parse_arguments():
    """
    Parse and validate command line arguments
    
    Returns:
        tuple: (data_filename, training_filename, validation_filename)
    """
    parser = argparse.ArgumentParser(description='Split dataset into training and validation sets')
    parser.add_argument('--data_filename', type=str, help='Input CSV data filename')
    parser.add_argument('--training_filename', type=str, nargs='?', const='', help='Output training CSV filename')
    parser.add_argument('--validation_filename', type=str, nargs='?', const='', help='Output validation CSV filename')
    
    args = parser.parse_args()
    
    # Check for required data_filename
    if not args.data_filename:
        print("Error: --data_filename is required")
        sys.exit(1)
    
    # Check if data file exists
    if not os.path.exists(args.data_filename):
        print(f"Error: File '{args.data_filename}' does not exist")
        sys.exit(1)
    
    # Check if data file ends with .csv
    if not args.data_filename.lower().endswith('.csv'):
        print(f"Error: File '{args.data_filename}' must be a CSV file")
        sys.exit(1)
    
    # Set default filenames if not provided or empty
    training_filename = args.training_filename if args.training_filename else 'data_training.csv'
    validation_filename = args.validation_filename if args.validation_filename else 'data_validation.csv'
    
    # Check for duplicate filenames
    if args.data_filename == training_filename:
        print("Error: Training filename cannot be the same as data filename")
        sys.exit(1)
    
    if args.data_filename == validation_filename:
        print("Error: Validation filename cannot be the same as data filename")
        sys.exit(1)
    
    if training_filename == validation_filename:
        print("Error: Training and validation filenames cannot be the same")
        sys.exit(1)
    
    return args.data_filename, training_filename, validation_filename

def main():
    """Main function"""
    # Parse command line arguments
    input_file, train_output, val_output = parse_arguments()
    
    # Fixed parameters
    train_ratio = 0.8
    
    print("=" * 60)
    print("MULTILAYER PERCEPTRON - DATA SPLITTING")
    print("=" * 60)
    
    # Load data
    data = load_data(input_file)
    
    # Extract components (simplified preprocessing without normalization)
    # Column 0: ID
    # Column 1: Diagnosis (M/B)
    # Columns 2-31: 30 features
    
    ids = data[:, 0]  # ID column
    labels = data[:, 1]  # Diagnosis column (keep as M/B)
    features = data[:, 2:32].astype(float)  # Feature columns
    
    print(f"Dataset loaded: {data.shape[0]} samples, {features.shape[1]} features")
    
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