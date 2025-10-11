#!/usr/bin/env python3
"""
Multilayer Perceptron - Object-Oriented Implementation
Flexible neural network with support for arbitrary number of layers
"""

import numpy as np
import json
from .nn_utils import *


# ============================================================================
# LAYER CLASS
# ============================================================================

class Layer:
    """
    Represents a single layer in the neural network

    Attributes:
        neurons: Number of neurons in this layer
        layer_type: Type of layer ('input', 'hidden', 'output')
        activation: Activation function ('sigmoid', 'softmax')
        weights: Weight matrix (initialized during network construction)
        biases: Bias vector (initialized during network construction)
    """

    def __init__(self, neurons, layer_type='hidden', activation='sigmoid'):
        """
        Initialize a layer

        Args:
            neurons: Number of neurons in this layer
            layer_type: Type of layer ('input', 'hidden', 'output')
            activation: Activation function ('sigmoid', 'softmax')
        """
        if neurons <= 0:
            raise ValueError(f"Number of neurons must be positive, got {neurons}")

        if layer_type not in ['input', 'hidden', 'output']:
            raise ValueError(f"Invalid layer type '{layer_type}'. Must be 'input', 'hidden', or 'output'")

        if activation not in ['sigmoid', 'softmax']:
            raise ValueError(f"Invalid activation '{activation}'. Must be 'sigmoid'or 'softmax'")

        self.neurons = neurons
        self.layer_type = layer_type
        self.activation = activation
        self.weights = None
        self.biases = None

        # Cache for forward/backward propagation
        self.Z = None  # Linear output (before activation)
        self.A = None  # Activated output
        self.dW = None  # Weight gradients
        self.db = None  # Bias gradients

    def initialize_weights(self, input_size):
        """
        Initialize weights and biases using Xavier/Glorot initialization

        Args:
            input_size: Number of inputs to this layer (neurons from previous layer)
        """
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + self.neurons))
        self.weights = np.random.uniform(-limit, limit, size=(input_size, self.neurons))
        self.biases = np.zeros(self.neurons)

    def forward(self, X):
        """
        Forward propagation through this layer

        Args:
            X: Input matrix, shape (batch_size, input_features)

        Returns:
            A: Activated output, shape (batch_size, neurons)
        """
        # Linear transformation
        self.Z = X @ self.weights + self.biases

        # Apply activation function
        if self.activation == 'sigmoid':
            self.A = sigmoid(self.Z)
        elif self.activation == 'softmax':
            self.A = softmax(self.Z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        return self.A

# ============================================================================
# NEURAL NETWORK CLASS
# ============================================================================

class NeuralNetwork:
    """
    Flexible Neural Network that supports arbitrary number of layers

    Attributes:
        layers: List of Layer objects
        input_size: Number of input features
    """

    def __init__(self, input_size):
        """
        Initialize neural network

        Args:
            input_size: Number of input features
        """
        if input_size <= 0:
            raise ValueError(f"Input size must be positive, got {input_size}")

        self.input_size = input_size
        self.layers = []

    def add_layer(self, neurons, layer_type='hidden', activation='sigmoid'):
        """
        Add a layer to the network

        Args:
            neurons: Number of neurons in the layer
            layer_type: Type of layer ('input', 'hidden', 'output')
            activation: Activation function ('sigmoid', 'softmax', 'linear')

        Returns:
            self: For method chaining
        """
        layer = Layer(neurons, layer_type, activation)
        self.layers.append(layer)
        return self

    def initialize(self):
        """
        Initialize all weights and biases in the network
        """
        if len(self.layers) == 0:
            raise ValueError("Cannot initialize network with no layers. Add layers first.")

        # Initialize first layer
        self.layers[0].initialize_weights(self.input_size)

        # Initialize subsequent layers
        for i in range(1, len(self.layers)):
            prev_neurons = self.layers[i - 1].neurons
            self.layers[i].initialize_weights(prev_neurons)

    def apply_feedforward(self, X):
        """
        Forward propagation through entire network

        Args:
            X: Input features, shape (batch_size, input_size)

        Returns:
            Output of final layer, shape (batch_size, output_neurons)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def apply_backpropagation(self, X, Y, learning_rate):
        """
        Backward propagation through entire network

        Following the exact equations from train_mlp.py:
        - Layer L: dZ_L = (1/m) * (A_L - Y_one_hot)  [for softmax + cross-entropy]
        - Layer l: dW_l = A_(l-1).T @ dZ_l
        -          db_l = sum(dZ_l, axis=0)
        -          dA_(l-1) = dZ_l @ W_l.T
        -          dZ_(l-1) = dA_(l-1) * sigmoid_prime(Z_(l-1))

        Args:
            X: Input features, shape (batch_size, input_size)
            Y: True labels, shape (batch_size,) with integer values
            learning_rate: Learning rate for gradient descent
        """
        m = X.shape[0]  # Batch size

        # Convert Y to one-hot encoding for output layer
        # Output layer has 2 neurons: [P(M), P(B)]
        # Y=1 (M) → [1,0], Y=0 (B) → [0,1]
        Y_one_hot = np.zeros((m, 2))
        Y_one_hot[range(m), 1 - Y] = 1

        # Compute output layer gradient
        # For softmax + cross-entropy: dZ = (1/m) * (A - Y_one_hot)
        output_layer = self.layers[-1]
        dZ = (output_layer.A - Y_one_hot) / m

        # Backpropagate through all layers (from output to input)
        for i in range(len(self.layers) - 1, -1, -1):
            current_layer = self.layers[i]

            # Get input to this layer (output of previous layer or X)
            if i == 0:
                A_prev = X
            else:
                A_prev = self.layers[i - 1].A

            # Compute gradients for weights and biases
            # dW = A_prev.T @ dZ
            # db = sum(dZ, axis=0)
            current_layer.dW = A_prev.T @ dZ
            current_layer.db = np.sum(dZ, axis=0)

            # Compute gradient for previous layer BEFORE updating weights
            # This is critical: we need current weights for backpropagation
            if i > 0:
                # dA_prev = dZ @ W.T
                dA_prev = dZ @ current_layer.weights.T

                # Apply derivative of activation function
                # dZ_prev = dA_prev * activation_derivative(Z_prev)
                prev_layer = self.layers[i - 1]
                dZ = dA_prev * sigmoid_prime(prev_layer.Z)

            # Now update weights and biases using gradient descent
            # W = W - learning_rate * dW
            # b = b - learning_rate * db
            current_layer.weights -= learning_rate * current_layer.dW
            current_layer.biases -= learning_rate * current_layer.db

    def train_batch(self, X, Y, learning_rate):
        """
        Train on a single batch (forward + backward)

        Args:
            X: Input features, shape (batch_size, input_size)
            Y: True labels, shape (batch_size,)
            learning_rate: Learning rate

        Returns:
            tuple: (loss, accuracy) for this batch
        """
        # Forward propagation
        output = self.apply_feedforward(X)

        # Calculate loss and accuracy
        loss = self.cross_entropy_loss(Y, output)
        accuracy = self.calculate_accuracy(Y, output)

        # Backward propagation
        self.apply_backpropagation(X, Y, learning_rate)

        return loss, accuracy

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss

        Args:
            y_true: True labels, shape (m,) with values 0 or 1
            y_pred: Predicted probabilities, shape (m, 2) from softmax output

        Returns:
            float: Average cross-entropy loss
        """
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Binary classification with M=1, B=0 encoding
        # y_pred[:, 0] = P(M), y_pred[:, 1] = P(B)
        losses = -(y_true * np.log(y_pred_clipped[:, 0]) +
                   (1 - y_true) * np.log(y_pred_clipped[:, 1]))

        return np.mean(losses)

    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy

        Args:
            y_true: True labels, shape (m,) with values 0 or 1
            y_pred: Predicted probabilities, shape (m, 2)

        Returns:
            float: Accuracy between 0.0 and 1.0
        """
        # Binary classification with M=1, B=0 encoding
        # y_pred[:, 0] = P(M), y_pred[:, 1] = P(B)
        argmax_indices = np.argmax(y_pred, axis=1)
        predicted_classes = 1 - argmax_indices

        # Calculate accuracy
        correct = np.sum(predicted_classes == y_true)
        return correct / len(y_true)

    def predict(self, X, Y):
        """
        Make predictions on input data and calculate loss and accuracy

        Args:
            X: Input features, shape (batch_size, input_size)
            Y: True labels, shape (batch_size,) with values 0 or 1

        Returns:
            tuple: (loss, accuracy)
                loss: Cross-entropy loss
                accuracy: Accuracy score between 0.0 and 1.0
        """
        output = self.apply_feedforward(X)

        # Calculate loss and accuracy
        loss = self.cross_entropy_loss(Y, output)
        accuracy = self.calculate_accuracy(Y, output)

        return loss, accuracy

    def save(self, filepath):
        """
        Save model to JSON file

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'input_size': self.input_size,
            'architecture': [],
            'weights': [],
            'biases': []
        }

        for i, layer in enumerate(self.layers):
            model_data['architecture'].append({
                'layer_index': i,
                'neurons': layer.neurons,
                'layer_type': layer.layer_type,
                'activation': layer.activation
            })

            if layer.weights is not None:
                model_data['weights'].append(layer.weights.tolist())
                model_data['biases'].append(layer.biases.tolist())

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load model from JSON file

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        # Clear existing layers
        self.layers = []
        self.input_size = model_data['input_size']

        # Reconstruct architecture
        for layer_config in model_data['architecture']:
            self.add_layer(
                neurons=layer_config['neurons'],
                layer_type=layer_config['layer_type'],
                activation=layer_config['activation']
            )

        # Load weights and biases
        for i, layer in enumerate(self.layers):
            layer.weights = np.array(model_data['weights'][i])
            layer.biases = np.array(model_data['biases'][i])

        print(f"Model loaded from {filepath}")
