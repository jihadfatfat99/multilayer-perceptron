"""
Neural Network implementation with ReLU activation functions.
Object-oriented design for multi-layer perceptron.
"""

import numpy as np
import sys
from typing import List, Tuple


class TeeOutput:
    """Class to write output to both console and file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class NeuralNetwork:
    """
    Multi-layer perceptron with ReLU hidden layers and softmax output.
    """

    def __init__(self, architecture: List[int], random_seed: int = 42, optimizer: str = "gd"):
        """
        Initialize network with specified layer sizes.

        Args:
            architecture: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            random_seed: Seed for reproducible weight initialization
        """
        self.architecture = architecture
        self.num_layers = len(architecture) - 1
        self.rng = np.random.default_rng(random_seed)
        self.optimizer = optimizer
        self.t = 0

        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize weights using Xavier/Glorot initialization for ReLU.
        Biases initialized with small positive values to avoid dead ReLU.
        """
        for idx in range(self.num_layers):
            input_size = self.architecture[idx]
            output_size = self.architecture[idx + 1]

            # Xavier initialization scaled for ReLU
            variance = 2.0 / input_size
            weight_matrix = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(variance),
                size=(input_size, output_size)
            )

            # Small positive bias to help avoid dead ReLU neurons
            bias_vector = np.full(output_size, 0.01)

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            self.m_weights.append(np.zeros_like(weight_matrix))
            self.v_weights.append(np.zeros_like(weight_matrix))
            self.m_biases.append(np.zeros_like(bias_vector))
            self.v_biases.append(np.zeros_like(bias_vector))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU activation: f(z) = max(0, z)
        """
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: f'(z) = 1 if z > 0 else 0
        """
        return (z > 0).astype(np.float64)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation for output layer with numerical stability.

        f(z_i) = exp(z_i) / sum(exp(z_j))
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exponentials = np.exp(z_shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform forward propagation through the network.

        Args:
            X: Input data of shape (batch_size, input_dim)

        Returns:
            pre_activations: List of pre-activation values (Z) for each layer
            post_activations: List of post-activation values (A) for each layer
        """
        pre_activations = []
        post_activations = [X]

        current_activation = X

        for layer_idx in range(self.num_layers):
            # Linear transformation: Z = A @ W + b
            z = current_activation @ self.weights[layer_idx] + self.biases[layer_idx]
            pre_activations.append(z)

            # Apply activation function
            if layer_idx == self.num_layers - 1:
                # Output layer: softmax
                current_activation = self.softmax(z)
            else:
                # Hidden layers: ReLU
                current_activation = self.relu(z)

            post_activations.append(current_activation)

        return pre_activations, post_activations

    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute categorical cross-entropy loss.

        L = -1/N * sum(y * log(p))

        Args:
            predictions: Predicted probabilities (batch_size, num_classes)
            targets: One-hot encoded true labels (batch_size, num_classes)

        Returns:
            Average cross-entropy loss
        """
        batch_size = predictions.shape[0]
        epsilon = 1e-12

        # Clip predictions to avoid log(0)
        clipped_predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

        # Compute cross-entropy
        loss = -np.sum(targets * np.log(clipped_predictions)) / batch_size

        return loss

    def backward_pass(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        learning_rate: float
    ) -> None:
        """
        Perform backpropagation and update weights/biases.

        Args:
            X: Input batch (batch_size, input_dim)
            y_true: One-hot encoded labels (batch_size, num_classes)
            learning_rate: Step size for gradient descent
        """
        batch_size = X.shape[0]

        # Forward pass to get activations
        pre_activations, post_activations = self.forward_pass(X)

        # Initialize gradient for output layer
        # For softmax + cross-entropy: dL/dZ = (predictions - true_labels) / batch_size
        gradient = (post_activations[-1] - y_true) / batch_size

        # Backpropagate through layers (from output to input)
        for layer_idx in reversed(range(self.num_layers)):
            # Gradient with respect to weights: dL/dW = A^T @ gradient
            weight_gradient = post_activations[layer_idx].T @ gradient

            # Gradient with respect to biases: dL/db = sum(gradient)
            bias_gradient = np.sum(gradient, axis=0)

            # Update parameters
            if self.optimizer == "gd":
                self.weights[layer_idx] -= learning_rate * weight_gradient
                self.biases[layer_idx] -= learning_rate * bias_gradient
            else:
                # ---------- ADAM OPTIMIZER ----------
                self.t += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8

                # Update biased first moment estimates
                self.m_weights[layer_idx] = beta1 * self.m_weights[layer_idx] + (1 - beta1) * weight_gradient
                self.m_biases[layer_idx]  = beta1 * self.m_biases[layer_idx]  + (1 - beta1) * bias_gradient

                # Update biased second raw moment estimates
                self.v_weights[layer_idx] = beta2 * self.v_weights[layer_idx] + (1 - beta2) * (weight_gradient ** 2)
                self.v_biases[layer_idx]  = beta2 * self.v_biases[layer_idx]  + (1 - beta2) * (bias_gradient ** 2)

                # Compute bias-corrected estimates
                m_w_hat = self.m_weights[layer_idx] / (1 - beta1 ** self.t)
                v_w_hat = self.v_weights[layer_idx] / (1 - beta2 ** self.t)
                m_b_hat = self.m_biases[layer_idx] / (1 - beta1 ** self.t)
                v_b_hat = self.v_biases[layer_idx] / (1 - beta2 ** self.t)

                # Update parameters
                self.weights[layer_idx] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.biases[layer_idx]  -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            # Propagate gradient to previous layer
            if layer_idx > 0:
                # dL/dA_prev = gradient @ W^T
                gradient = gradient @ self.weights[layer_idx].T

                # Apply derivative of activation function (ReLU)
                gradient *= self.relu_derivative(pre_activations[layer_idx - 1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input data (batch_size, input_dim)

        Returns:
            Predicted probabilities (batch_size, num_classes)
        """
        _, post_activations = self.forward_pass(X)
        return post_activations[-1]

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (argmax of probabilities).

        Args:
            X: Input data (batch_size, input_dim)

        Returns:
            Predicted class indices (batch_size,)
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def evaluate_accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input data (batch_size, input_dim)
            y_true: True class labels (batch_size,) - NOT one-hot encoded

        Returns:
            Accuracy as a fraction between 0 and 1
        """
        predictions = self.predict_classes(X)
        accuracy = np.mean(predictions == y_true)
        return accuracy
