#!/usr/bin/env python3
"""
Neural Network Utilities
Helper functions for neural network operations including activation functions
"""

import numpy as np
import sys


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data(filename):
    """
    Load data from CSV file

    Args:
        filename: Path to the CSV file

    Returns:
        tuple: (X, Y) where:
            X: Features matrix, shape (m, n_features)
            Y: Labels vector, shape (m,) with values 0 or 1 (B=0, M=1)
    """
    try:
        # Read CSV file
        data = np.loadtxt(filename, delimiter=',', dtype=str)

        # Extract labels from 2nd column (index 1)
        # Convert 'M' to 1 and 'B' to 0
        Y_raw = data[:, 1]
        Y = np.where(Y_raw == 'M', 1, 0).astype(int)

        # Extract features from 3rd column onwards (index 2:)
        X = data[:, 2:].astype(float)

        return X, Y

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)


def get_mean_std(X):
    """
    Calculate mean and standard deviation for normalization

    Args:
        X: Features matrix, shape (m, n_features)

    Returns:
        tuple: (mean, std) where:
            mean: Mean of each feature, shape (n_features,)
            std: Standard deviation of each feature, shape (n_features,)
    """
    # Calculate mean for each feature (column-wise)
    mean = np.mean(X, axis=0)

    # Calculate standard deviation for each feature (column-wise)
    std = np.std(X, axis=0)

    return mean, std


def normalize_data(X, mean, std):
    """
    Normalize data using mean and standard deviation

    Applies z-score normalization: X_normalized = (X - mean) / std

    Args:
        X: Features matrix, shape (m, n_features)
        mean: Mean of each feature, shape (n_features,)
        std: Standard deviation of each feature, shape (n_features,)

    Returns:
        X: Normalized features matrix, shape (m, n_features)
    """
    # Z-score normalization: (X - mean) / std
    X = (X - mean) / std

    return X

def shuffle_and_split_into_mini_batches(X, Y, batch_size):
    """
    Shuffle data and split into mini-batches

    Args:
        X: Features matrix, shape (m, n_features)
        Y: Labels vector, shape (m,) with values 0 or 1
        batch_size: Size of each mini-batch

    Returns:
        list: [[[X1], [Y1]], [[X2], [Y2]], ...] where:
              Xi: features matrix of batch i
              Yi: labels vector of batch i with values 0 or 1
    """
    n_samples = X.shape[0]

    # Shuffle the data (without seed for randomness)
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # Create mini-batches
    num_batches = (n_samples + batch_size - 1) // batch_size
    batches = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]

        batches.append([X_batch, Y_batch])

    return batches


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def sigmoid(x):
    """
    Sigmoid activation function with numerical stability

    Uses different formulas for positive and negative values:
    - For x >= 0: sigmoid(x) = 1 / (1 + e^(-x))
    - For x < 0:  sigmoid(x) = e^x / (e^x + 1)

    This prevents overflow issues in the exponential function.

    Args:
        x: Input matrix/array of any shape

    Returns:
        Output matrix with sigmoid applied element-wise
    """
    # Create output array
    output = np.zeros_like(x, dtype=np.float64)

    # For positive values: sigmoid(x) = 1 / (1 + e^(-x))
    positive_mask = x >= 0
    output[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))

    # For negative values: sigmoid(x) = e^x / (e^x + 1)
    negative_mask = x < 0
    exp_x = np.exp(x[negative_mask])
    output[negative_mask] = exp_x / (exp_x + 1)

    return output


def sigmoid_prime(x):
    """
    Derivative of sigmoid function

    Args:
        x: Input matrix/array of any shape

    Returns:
        Derivative of sigmoid applied element-wise
    """
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)


def softmax(x):
    """
    Softmax activation function

    Args:
        x: Input matrix of shape (batch_size, num_classes) or (num_classes,)

    Returns:
        Output matrix with softmax applied, same shape as input
        Each row sums to 1.0 (probability distribution)
    """
    # Subtract max for numerical stability
    x_stable = x - np.max(x, axis=-1, keepdims=True)

    # Compute exponentials
    exp_x = np.exp(x_stable)

    # Compute softmax
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
