#!/usr/bin/env python3
"""
Multilayer Perceptron - Data Splitting Program
Splits the breast cancer dataset into training and validation sets
"""

import numpy as np
import pandas as pd
import sys

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
        tuple: (features, labels)
    """
    print("Preprocessing data...")
    
    # Extract components
    # Column 0: ID (ignore)
    # Column 1: Diagnosis (M/B)
    # Columns 2-31: 30 features
    
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
    
    return features_normalized, labels_encoded

def split_dataset(features, labels, train_ratio=0.8):
    """
    Split dataset into training and validation sets
    
    Args:
        features: Normalized feature matrix
        labels: Encoded labels
        train_ratio: Ratio of training data (default 0.8)
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
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
    X_val = features[val_indices]
    y_val = labels[val_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Check class distribution in splits
    train_malignant = np.sum(y_train)
    train_benign = len(y_train) - train_malignant
    val_malignant = np.sum(y_val)
    val_benign = len(y_val) - val_malignant
    
    print(f"Training set distribution: {train_malignant} malignant, {train_benign} benign")
    print(f"Validation set distribution: {val_malignant} malignant, {val_benign} benign")
    
    return X_train, y_train, X_val, y_val

def save_splits(X_train, y_train, X_val, y_val, train_file, val_file):
    """
    Save training and validation sets to CSV files
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        train_file: Training output filename
        val_file: Validation output filename
    """
    print("Saving split datasets...")
    
    # Combine features and labels
    train_data = np.column_stack((y_train, X_train))
    val_data = np.column_stack((y_val, X_val))
    
    # Save to CSV files (no headers)
    np.savetxt(train_file, train_data, delimiter=',', fmt='%.6f')
    np.savetxt(val_file, val_data, delimiter=',', fmt='%.6f')
    
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")

def main():
    """Main function"""
    # Fixed parameters - no command line arguments needed
    input_file = 'data.csv'
    train_ratio = 0.8
    train_output = 'data_training.csv'
    val_output = 'data_validation.csv'
    
    print("=" * 60)
    print("MULTILAYER PERCEPTRON - DATA SPLITTING")
    print("=" * 60)
    
    # Load data
    data = load_data(input_file)
    
    # Preprocess data
    features, labels = preprocess_data(data)
    
    # Split dataset
    X_train, y_train, X_val, y_val = split_dataset(features, labels, train_ratio)
    
    # Save splits
    save_splits(X_train, y_train, X_val, y_val, train_output, val_output)
    
    print("=" * 60)
    print("DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Next step: Run training with 'python train_mlp.py'")

if __name__ == "__main__":
    main()