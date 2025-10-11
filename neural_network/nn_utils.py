#!/usr/bin/env python3
"""
Neural Network Utilities
Helper functions for neural network operations including activation functions
"""

import numpy as np


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
