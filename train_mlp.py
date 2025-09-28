import numpy as np

def xavier_init(n_in, n_out):
    """
    Xavier/Glorot initialization for weights and biases
    
    Args:
        n_in: Number of input neurons
        n_out: Number of output neurons
        
    Returns:
        tuple: (weights, biases)
            weights: shape (n_in, n_out)
            biases: shape (n_out,)
    """
    # Xavier limit
    limit = np.sqrt(6.0 / (n_in + n_out))
    
    # Initialize weights
    weights = np.random.uniform(-limit, limit, size=(n_in, n_out))
    
    # Initialize biases to zeros
    biases = np.zeros(n_out)
    
    return weights, biases

def initialize_network(input_size=30, hidden1_size=20, hidden2_size=10, output_size=2, seed=42):
    """
    Initialize all weights and biases for the complete network
    
    Args:
        input_size: Number of input features (default: 30)
        hidden1_size: First hidden layer size (default: 20)
        hidden2_size: Second hidden layer size (default: 10)
        output_size: Output layer size (default: 2)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (W1, b1, W2, b2, W3, b3)
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Initialize each layer
    W1, b1 = xavier_init(input_size, hidden1_size)      # 30 -> 20
    W2, b2 = xavier_init(hidden1_size, hidden2_size)    # 20 -> 10
    W3, b3 = xavier_init(hidden2_size, output_size)     # 10 -> 2
    
    return W1, b1, W2, b2, W3, b3

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

def cross_entropy(y_true, y_pred):
    """
    Binary cross-entropy loss for mini-batch
    
    Args:
        y_true: True labels, shape (m,) with values 0 or 1
        y_pred: Predicted probabilities, shape (m, 2) from softmax output
        
    Returns:
        float: Average cross-entropy loss across the mini-batch
    """
    # Get batch size
    m = y_true.shape[0]
    
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate cross-entropy for each sample
    # y_pred[:, 0] = P(malignant), y_pred[:, 1] = P(benign)
    losses = -(y_true * np.log(y_pred_clipped[:, 0]) + 
               (1 - y_true) * np.log(y_pred_clipped[:, 1]))
    
    # Return average loss across mini-batch
    return np.mean(losses)

def argmax(y_pred, axis=-1):
    """
    Return class predictions from softmax probabilities
    Handles M=1, B=0 encoding with [P(malignant), P(benign)] network output
    
    Args:
        y_pred: Predicted probabilities, shape (m, 2)
                y_pred[:, 0] = P(malignant)
                y_pred[:, 1] = P(benign)
        axis: Axis along which to find argmax (default: -1)
        
    Returns:
        Array of predicted classes: 1 for malignant, 0 for benign
    """
    # Get argmax indices (0 or 1)
    argmax_indices = np.argmax(y_pred, axis=axis)
    
    # Convert to your label encoding (M=1, B=0)
    # argmax=0 (malignant has higher prob) → class=1 (M)
    # argmax=1 (benign has higher prob) → class=0 (B)
    predicted_classes = 1 - argmax_indices
    
    return predicted_classes

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy for binary classification
    
    Args:
        y_true: True labels, shape (m,) with values 0 or 1 (B=0, M=1)
        y_pred: Predicted probabilities, shape (m, 2) from softmax
        
    Returns:
        float: Accuracy as a value between 0.0 and 1.0
    """
    # Convert probabilities to predicted classes using argmax
    predicted_classes = argmax(y_pred)
    
    # Count correct predictions
    correct_predictions = np.sum(predicted_classes == y_true)
    
    # Calculate accuracy
    total_predictions = len(y_true)
    accuracy_score = correct_predictions / total_predictions
    
    return accuracy_score