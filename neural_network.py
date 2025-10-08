#!/usr/bin/env python3
"""
Multilayer Perceptron - Object-Oriented Implementation
Flexible neural network with support for arbitrary number of layers
"""

import numpy as np
import json
import sys


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def sigmoid(x):
    """
    Sigmoid activation function

    Args:
        x: Input matrix/array of any shape

    Returns:
        Output matrix with sigmoid applied element-wise
    """
    # Clip x to prevent overflow in exp(-x)
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


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


# ============================================================================
# LAYER CLASS
# ============================================================================

class Layer:
    """
    Represents a single layer in the neural network

    Attributes:
        neurons: Number of neurons in this layer
        layer_type: Type of layer ('input', 'hidden', 'output')
        activation: Activation function ('sigmoid', 'softmax', 'linear')
        weights: Weight matrix (initialized during network construction)
        biases: Bias vector (initialized during network construction)
    """

    def __init__(self, neurons, layer_type='hidden', activation='sigmoid'):
        """
        Initialize a layer

        Args:
            neurons: Number of neurons in this layer
            layer_type: Type of layer ('input', 'hidden', 'output')
            activation: Activation function ('sigmoid', 'softmax', 'linear')
        """
        if neurons <= 0:
            raise ValueError(f"Number of neurons must be positive, got {neurons}")

        if layer_type not in ['input', 'hidden', 'output']:
            raise ValueError(f"Invalid layer type '{layer_type}'. Must be 'input', 'hidden', or 'output'")

        if activation not in ['sigmoid', 'softmax', 'linear']:
            raise ValueError(f"Invalid activation '{activation}'. Must be 'sigmoid', 'softmax', or 'linear'")

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

    def initialize_weights(self, input_size, seed=None):
        """
        Initialize weights and biases using Xavier/Glorot initialization

        Args:
            input_size: Number of inputs to this layer (neurons from previous layer)
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)

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
        elif self.activation == 'linear':
            self.A = self.Z
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        return self.A

    def __repr__(self):
        """String representation of the layer"""
        return f"Layer(neurons={self.neurons}, type='{self.layer_type}', activation='{self.activation}')"


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

    def initialize(self, seed=42):
        """
        Initialize all weights and biases in the network

        Args:
            seed: Random seed for reproducibility
        """
        if len(self.layers) == 0:
            raise ValueError("Cannot initialize network with no layers. Add layers first.")

        # Set global seed
        np.random.seed(seed)

        # Initialize first layer
        self.layers[0].initialize_weights(self.input_size, seed=seed)

        # Initialize subsequent layers
        for i in range(1, len(self.layers)):
            prev_neurons = self.layers[i - 1].neurons
            self.layers[i].initialize_weights(prev_neurons, seed=seed + i)

    def forward(self, X):
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

    def backward(self, X, Y, learning_rate):
        """
        Backward propagation through entire network

        Args:
            X: Input features, shape (batch_size, input_size)
            Y: True labels, shape (batch_size,) with integer values
            learning_rate: Learning rate for gradient descent
        """
        m = X.shape[0]  # Batch size

        # Convert Y to one-hot encoding for output layer
        num_classes = self.layers[-1].neurons
        Y_one_hot = np.zeros((m, num_classes))

        # For binary classification: Y=1 (M) → [1,0], Y=0 (B) → [0,1]
        # Network output: A[:, 0] = P(class 0), A[:, 1] = P(class 1)
        if num_classes == 2:
            Y_one_hot[range(m), 1 - Y] = 1  # Flip index for M/B encoding
        else:
            Y_one_hot[range(m), Y] = 1

        # Compute output layer gradient
        output_layer = self.layers[-1]
        if output_layer.activation == 'softmax':
            # Softmax + Cross-entropy derivative
            dZ = (1 / m) * (output_layer.A - Y_one_hot)
        else:
            # For other activations (not typical for classification)
            dA = (1 / m) * (output_layer.A - Y_one_hot)
            if output_layer.activation == 'sigmoid':
                dZ = dA * sigmoid_prime(output_layer.Z)
            else:
                dZ = dA

        # Backpropagate through all layers
        for i in range(len(self.layers) - 1, -1, -1):
            current_layer = self.layers[i]

            # Get input to this layer (output of previous layer or X)
            if i == 0:
                A_prev = X
            else:
                A_prev = self.layers[i - 1].A

            # Compute gradients for weights and biases
            current_layer.dW = A_prev.T @ dZ
            current_layer.db = np.sum(dZ, axis=0)

            # Update weights and biases
            current_layer.weights -= learning_rate * current_layer.dW
            current_layer.biases -= learning_rate * current_layer.db

            # Compute gradient for previous layer (if not first layer)
            if i > 0:
                dA_prev = dZ @ current_layer.weights.T

                # Apply derivative of activation function
                prev_layer = self.layers[i - 1]
                if prev_layer.activation == 'sigmoid':
                    dZ = dA_prev * sigmoid_prime(prev_layer.Z)
                elif prev_layer.activation == 'linear':
                    dZ = dA_prev
                else:
                    dZ = dA_prev

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
        output = self.forward(X)

        # Calculate loss and accuracy
        loss = self.cross_entropy_loss(Y, output)
        accuracy = self.calculate_accuracy(Y, output)

        # Backward propagation
        self.backward(X, Y, learning_rate)

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

        # For binary classification with M=1, B=0 encoding
        if y_pred.shape[1] == 2:
            losses = -(y_true * np.log(y_pred_clipped[:, 0]) +
                       (1 - y_true) * np.log(y_pred_clipped[:, 1]))
        else:
            # Multi-class case
            m = len(y_true)
            losses = -np.log(y_pred_clipped[range(m), y_true])

        return np.mean(losses)

    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy

        Args:
            y_true: True labels, shape (m,)
            y_pred: Predicted probabilities, shape (m, num_classes)

        Returns:
            float: Accuracy between 0.0 and 1.0
        """
        # Get predicted classes
        if y_pred.shape[1] == 2:
            # Binary classification with M=1, B=0 encoding
            argmax_indices = np.argmax(y_pred, axis=1)
            predicted_classes = 1 - argmax_indices
        else:
            # Multi-class
            predicted_classes = np.argmax(y_pred, axis=1)

        # Calculate accuracy
        correct = np.sum(predicted_classes == y_true)
        return correct / len(y_true)

    def predict(self, X):
        """
        Make predictions on input data

        Args:
            X: Input features, shape (batch_size, input_size)

        Returns:
            Predicted class labels, shape (batch_size,)
        """
        output = self.forward(X)

        if output.shape[1] == 2:
            # Binary classification
            argmax_indices = np.argmax(output, axis=1)
            return 1 - argmax_indices
        else:
            # Multi-class
            return np.argmax(output, axis=1)

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

    def summary(self):
        """Print network architecture summary"""
        print("=" * 80)
        print("NEURAL NETWORK ARCHITECTURE")
        print("=" * 80)
        print(f"Input Size: {self.input_size}")
        print("-" * 80)

        total_params = 0
        for i, layer in enumerate(self.layers):
            if layer.weights is not None:
                layer_params = layer.weights.size + layer.biases.size
                total_params += layer_params
                print(f"Layer {i + 1}: {layer.neurons} neurons ({layer.layer_type}, {layer.activation})")
                print(f"         Parameters: {layer_params} (Weights: {layer.weights.shape}, Biases: {layer.biases.shape})")
            else:
                print(f"Layer {i + 1}: {layer.neurons} neurons ({layer.layer_type}, {layer.activation}) [Not initialized]")

        print("-" * 80)
        print(f"Total Parameters: {total_params}")
        print("=" * 80)

    def __repr__(self):
        """String representation of the network"""
        layer_info = [f"{layer.neurons} neurons" for layer in self.layers]
        return f"NeuralNetwork(input={self.input_size}, layers=[{', '.join(layer_info)}])"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example demonstrating how to use the Layer and NeuralNetwork classes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE: Creating a flexible neural network")
    print("=" * 80)

    # Example 1: 3-layer network (like original implementation)
    print("\nExample 1: 3-layer network (30 -> 20 -> 10 -> 2)")
    nn1 = NeuralNetwork(input_size=30)
    nn1.add_layer(neurons=20, layer_type='hidden', activation='sigmoid')
    nn1.add_layer(neurons=10, layer_type='hidden', activation='sigmoid')
    nn1.add_layer(neurons=2, layer_type='output', activation='softmax')
    nn1.initialize(seed=42)
    nn1.summary()

    # Example 2: 5-layer deep network
    print("\nExample 2: 5-layer deep network (30 -> 25 -> 20 -> 15 -> 10 -> 2)")
    nn2 = NeuralNetwork(input_size=30)
    nn2.add_layer(neurons=25, layer_type='hidden', activation='sigmoid')
    nn2.add_layer(neurons=20, layer_type='hidden', activation='sigmoid')
    nn2.add_layer(neurons=15, layer_type='hidden', activation='sigmoid')
    nn2.add_layer(neurons=10, layer_type='hidden', activation='sigmoid')
    nn2.add_layer(neurons=2, layer_type='output', activation='softmax')
    nn2.initialize(seed=42)
    nn2.summary()

    # Example 3: Single hidden layer
    print("\nExample 3: Single hidden layer (30 -> 15 -> 2)")
    nn3 = NeuralNetwork(input_size=30)
    nn3.add_layer(neurons=15, layer_type='hidden', activation='sigmoid')
    nn3.add_layer(neurons=2, layer_type='output', activation='softmax')
    nn3.initialize(seed=42)
    nn3.summary()

    # Test forward propagation with dummy data
    print("\nTesting forward propagation with dummy data...")
    X_dummy = np.random.randn(5, 30)  # 5 samples, 30 features
    output = nn1.forward(X_dummy)
    print(f"Input shape: {X_dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities):\n{output}")
    print(f"Sum of probabilities per sample: {output.sum(axis=1)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    example_usage()
