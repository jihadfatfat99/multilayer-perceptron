import numpy as np
import sys
import json
import matplotlib.pyplot as plt

# ============================================================================
# NETWORK INITIALIZATION
# ============================================================================

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

# ============================================================================
# LOSS AND METRICS
# ============================================================================

def cross_entropy(y_true, y_pred):
    """
    Binary cross-entropy loss for mini-batch
    
    Args:
        y_true: True labels, shape (m,) with values 0 or 1
        y_pred: Predicted probabilities, shape (m, 2) from softmax output
        
    Returns:
        float: Average cross-entropy loss across the mini-batch
    """
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

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def get_csv_info(filepath):
    """
    Get the number of data rows and feature columns in a CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        dict: {'lines': number of rows, 'features': number of feature columns (excluding ID and label)}
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
            
            # Get number of columns from first line
            if line_count > 0:
                first_line = lines[0].strip()
                total_columns = len(first_line.split(','))
                # Subtract 2 columns (ID and label) to get feature count
                feature_columns = total_columns - 2
            else:
                feature_columns = 0
        
        return {
            'lines': line_count,
            'features': feature_columns
        }
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def get_training_and_validation_data(training_filename, validation_filename):
    """
    Load training and validation data from CSV files

    Args:
        training_filename: Path to the training CSV file
        validation_filename: Path to the validation CSV file

    Returns:
        list: [[[x_train], [y_train]], [[x_val], [y_val]]] where:
              x_train: features matrix from training file (m_train x features)
              y_train: labels vector from training file with values 0 or 1
              x_val: features matrix from validation file (m_val x features)
              y_val: labels vector from validation file with values 0 or 1
    """
    # Load training data
    try:
        train_data = np.loadtxt(training_filename, delimiter=',')
        # Skip first column (ID), get labels (column 1) and features (columns 2+)
        Y_train = train_data[:, 1].astype(int)  # Label column (1 for M, 0 for B)
        X_train = train_data[:, 2:]              # Feature columns
    except FileNotFoundError:
        print(f"Error: '{training_filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading training file '{training_filename}': {e}")
        sys.exit(1)

    # Load validation data
    try:
        val_data = np.loadtxt(validation_filename, delimiter=',')
        # Skip first column (ID), get labels (column 1) and features (columns 2+)
        Y_val = val_data[:, 1].astype(int)  # Label column (1 for M, 0 for B)
        X_val = val_data[:, 2:]              # Feature columns
    except FileNotFoundError:
        print(f"Error: '{validation_filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading validation file '{validation_filename}': {e}")
        sys.exit(1)

    return [[X_train, Y_train], [X_val, Y_val]]

def shuffle_and_split_into_mini_batches(training_data, batch_size):
    """
    Shuffle data and split into mini-batches

    Args:
        training_data: List containing [x_train, y_train] from get_training_and_validation_data
        batch_size: Size of each mini-batch

    Returns:
        list: [[[X1], [Y1]], [[X2], [Y2]], ...] where:
              Xi: features matrix of batch i with dimension (m x 30)
              Yi: actual results vector of batch i with values 0 or 1
    """
    # Extract features and labels from training_data
    X_train = training_data[0]  # Features matrix
    Y_train = training_data[1]  # Labels vector
    n_samples = X_train.shape[0]

    # Shuffle the data (without seed for randomness)
    indices = np.random.permutation(n_samples)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    # Create mini-batches
    num_batches = (n_samples + batch_size - 1) // batch_size
    batches = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_batch = X_train_shuffled[start:end]  # Shape: (m, 30)
        Y_batch = Y_train_shuffled[start:end]  # Shape: (m,)

        batches.append([X_batch, Y_batch])

    return batches

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parse command line arguments for training configuration
    
    Returns:
        dict: Configuration parameters
    """
    # Default values
    config = {
        'hidden1_size': 22,
        'hidden2_size': 16,
        'epochs': 300,
        'batch_size': 40,
        'learning_rate': 0.058,
    }
    
    # Valid keys
    valid_keys = ['layer', 'epochs', 'batch_size', 'learning_rate']
    
    # Track which keys have been used
    used_keys = set()

    # Get training data dimensions (number of samples and feature columns)
    csv_data = get_csv_info('data_training.csv')
    
    args = sys.argv[1:]  # Skip program name
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        # Check if argument is just "--" with nothing after
        if arg == '--':
            print(f"Error: Invalid argument '--' without key name")
            sys.exit(1)
        
        # Check if argument starts with --
        if not arg.startswith('--'):
            print(f"Error: Invalid argument '{arg}'. All arguments must start with '--'")
            sys.exit(1)
        
        key = arg[2:]  # Remove '--' prefix
        
        # Check if key is empty (e.g., just "--")
        if not key:
            print(f"Error: Invalid argument '--' without key name")
            sys.exit(1)
        
        # Check if key is valid
        if key not in valid_keys:
            print(f"Error: Unknown argument '--{key}'. Valid arguments are: {', '.join(['--' + k for k in valid_keys])}")
            sys.exit(1)
        
        # Check for duplicate keys
        if key in used_keys:
            print(f"Error: Argument '--{key}' specified more than once")
            sys.exit(1)
        
        # Mark key as used
        used_keys.add(key)
        
        # Check if next argument exists and is not a flag
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            value = args[i + 1]
            i += 2  # Move past key and value
        else:
            # No value provided, use default
            print(f"Warning: No value provided for '{arg}', using default")
            i += 1
            continue
        
        # Process each key
        if key == 'layer':
            # Parse hidden layer sizes
            layer_sizes = []
            temp_i = i - 1
            while temp_i < len(args) and not args[temp_i].startswith('--'):
                try:
                    size = int(args[temp_i])
                    if size <= 0:
                        print(f"Error: Layer size must be positive, got {size}")
                        sys.exit(1)
                    layer_sizes.append(size)
                    temp_i += 1
                except ValueError:
                    print(f"Error: Layer size must be an integer, got '{args[temp_i]}'")
                    sys.exit(1)
            
            i = temp_i
            
            if len(layer_sizes) == 1:
                print(f"Error: --layer requires exactly 2 values (hidden layer 1 and hidden layer 2 sizes), got only 1")
                sys.exit(1)
            
            if len(layer_sizes) == 0:
                print(f"Warning: No valid layer sizes provided, using defaults")
            else:
                for idx, size in enumerate(layer_sizes[:2]):
                    if size < 4:
                        print(f"Error: Hidden layer {idx + 1} size must be at least 4, got {size}")
                        sys.exit(1)
                
                # Get feature count from training file
                n_features = csv_data['features']
                
                for idx, size in enumerate(layer_sizes[:2]):
                    if size > n_features:
                        print(f"Error: Hidden layer {idx + 1} size ({size}) cannot exceed number of input features ({n_features})")
                        sys.exit(1)
                
                if len(layer_sizes) >= 2:
                    if layer_sizes[1] > layer_sizes[0]:
                        print(f"Error: Hidden layer 2 size ({layer_sizes[1]}) must be less than or equal to hidden layer 1 size ({layer_sizes[0]})")
                        sys.exit(1)
                
                if len(layer_sizes) >= 1:
                    config['hidden1_size'] = layer_sizes[0]
                if len(layer_sizes) >= 2:
                    config['hidden2_size'] = layer_sizes[1]
                if len(layer_sizes) > 2:
                    print(f"Warning: More than 2 hidden layers specified, using only first 2")
        
        elif key == 'epochs':
            try:
                epochs = int(value)
                if epochs <= 0:
                    print(f"Error: Epochs must be positive, got {epochs}")
                    sys.exit(1)
                if epochs < 10:
                    print(f"Error: Epochs too small (< 10). Network won't have enough time to learn.")
                    sys.exit(1)
                if epochs > 1000:
                    print(f"Error: Epochs too large (> 1000). This will take very long and likely overfit.")
                    sys.exit(1)
                config['epochs'] = epochs
            except ValueError:
                print(f"Error: Epochs must be an integer, got '{value}'")
                sys.exit(1)
        
        elif key == 'batch_size':
            try:
                batch_size = int(value)
                if batch_size <= 0:
                    print(f"Error: Batch size must be positive, got {batch_size}")
                    sys.exit(1)
                
                if batch_size < 4:
                    print(f"Error: Batch size too small (< 4). Use at least 4 samples per batch for stable gradients.")
                    sys.exit(1)
                
                # Get training data size
                n_train_samples = csv_data['lines']
                
                if batch_size > n_train_samples:
                    print(f"Error: Batch size ({batch_size}) cannot be larger than training data size ({n_train_samples})")
                    sys.exit(1)
                
                if batch_size > n_train_samples // 2:
                    print(f"Error: Batch size ({batch_size}) is too large (> {n_train_samples // 2}).")
                    print(f"For effective mini-batch training, use batch size between 4 and {n_train_samples // 2}")
                    sys.exit(1)
                
                config['batch_size'] = batch_size
            
            except ValueError:
                print(f"Error: Batch size must be an integer, got '{value}'")
                sys.exit(1)
        
        elif key == 'learning_rate':
            try:
                lr = float(value)
                if lr <= 0:
                    print(f"Error: Learning rate must be positive, got {lr}")
                    sys.exit(1)
                if lr < 0.0001:
                    print(f"Error: Learning rate too small (< 0.0001). Network will learn too slowly or not at all.")
                    sys.exit(1)
                if lr > 0.5:
                    print(f"Error: Learning rate too large (> 0.5). Network will be unstable and diverge.")
                    sys.exit(1)
                config['learning_rate'] = lr
            except ValueError:
                print(f"Error: Learning rate must be a number, got '{value}'")
                sys.exit(1)
    
    return config

# ============================================================================
# CORE TRAINING FUNCTIONS
# ============================================================================

def apply_feedforward(X, W1, b1, W2, b2, W3, b3):
    """
    Forward propagation through the neural network

    Args:
        X: Input features matrix, shape (m, 30) where m is batch size
        W1: First layer weights, shape (30, hidden1_size)
        b1: First layer biases, shape (hidden1_size,)
        W2: Second layer weights, shape (hidden1_size, hidden2_size)
        b2: Second layer biases, shape (hidden2_size,)
        W3: Third layer weights, shape (hidden2_size, 2)
        b3: Third layer biases, shape (2,)

    Returns:
        tuple: (Z1, A1, Z2, A2, Z3, A3) where:
            Z1: Linear output of layer 1, shape (m, hidden1_size)
            A1: Activated output of layer 1, shape (m, hidden1_size)
            Z2: Linear output of layer 2, shape (m, hidden2_size)
            A2: Activated output of layer 2, shape (m, hidden2_size)
            Z3: Linear output of layer 3, shape (m, 2)
            A3: Final output (softmax probabilities), shape (m, 2)
    """
    # Layer 1: Input -> Hidden1
    Z1 = X @ W1 + b1        # Matrix multiplication: (m, 30) @ (30, h1) = (m, h1)
    A1 = sigmoid(Z1)        # Apply sigmoid activation

    # Layer 2: Hidden1 -> Hidden2
    Z2 = A1 @ W2 + b2       # Matrix multiplication: (m, h1) @ (h1, h2) = (m, h2)
    A2 = sigmoid(Z2)        # Apply sigmoid activation

    # Layer 3: Hidden2 -> Output
    Z3 = A2 @ W3 + b3       # Matrix multiplication: (m, h2) @ (h2, 2) = (m, 2)
    A3 = softmax(Z3)        # Apply softmax activation

    return Z1, A1, Z2, A2, Z3, A3

def apply_backpropagation(X, Y, A1, A2, A3, Z1, Z2, W2, W3):
    """
    Backward propagation to compute gradients

    Args:
        X: Input features matrix, shape (m, 30)
        Y: True labels vector, shape (m,) with values 0 or 1
        A1: Activated output of layer 1, shape (m, hidden1_size)
        A2: Activated output of layer 2, shape (m, hidden2_size)
        A3: Final output (softmax probabilities), shape (m, 2)
        Z1: Linear output of layer 1, shape (m, hidden1_size)
        Z2: Linear output of layer 2, shape (m, hidden2_size)
        W2: Second layer weights, shape (hidden1_size, hidden2_size)
        W3: Third layer weights, shape (hidden2_size, 2)

    Returns:
        tuple: (dW1, db1, dW2, db2, dW3, db3) - Gradients for all weights and biases
    """
    m = X.shape[0]  # Batch size

    # Convert Y to one-hot encoding: shape (m, 2)
    # Network output: A3[:, 0] = P(M), A3[:, 1] = P(B)
    # Y encoding: Y=1 means M, Y=0 means B
    # One-hot: Y=1 (M) → [1,0], Y=0 (B) → [0,1]
    Y_one_hot = np.zeros((m, 2))
    Y_one_hot[range(m), 1 - Y] = 1  # Flip index: Y=1→index 0, Y=0→index 1

    # Output layer gradient (Layer 3)
    # dZ3 = 1/m * (A3 - Y)
    dZ3 = (1 / m) * (A3 - Y_one_hot)  # Shape: (m, 2)

    # Layer 3 gradients
    dW3 = A2.T @ dZ3           # (hidden2_size, m) @ (m, 2) = (hidden2_size, 2)
    db3 = np.sum(dZ3, axis=0)  # Sum over batch, shape (2,)

    # Layer 2 gradient
    dA2 = dZ3 @ W3.T           # (m, 2) @ (2, hidden2_size) = (m, hidden2_size)
    dZ2 = dA2 * sigmoid_prime(Z2)  # Element-wise multiplication with sigmoid derivative

    # Layer 2 gradients
    dW2 = A1.T @ dZ2           # (hidden1_size, m) @ (m, hidden2_size) = (hidden1_size, hidden2_size)
    db2 = np.sum(dZ2, axis=0)  # Sum over batch, shape (hidden2_size,)

    # Layer 1 gradient
    dA1 = dZ2 @ W2.T           # (m, hidden2_size) @ (hidden2_size, hidden1_size) = (m, hidden1_size)
    dZ1 = dA1 * sigmoid_prime(Z1)  # Element-wise multiplication with sigmoid derivative

    # Layer 1 gradients
    dW1 = X.T @ dZ1            # (30, m) @ (m, hidden1_size) = (30, hidden1_size)
    db1 = np.sum(dZ1, axis=0)  # Sum over batch, shape (hidden1_size,)

    return dW1, db1, dW2, db2, dW3, db3

def apply_gradient_descent(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    """
    Update weights and biases using gradient descent

    Args:
        W1: First layer weights, shape (30, hidden1_size)
        b1: First layer biases, shape (hidden1_size,)
        W2: Second layer weights, shape (hidden1_size, hidden2_size)
        b2: Second layer biases, shape (hidden2_size,)
        W3: Third layer weights, shape (hidden2_size, 2)
        b3: Third layer biases, shape (2,)
        dW1: Gradient for W1
        db1: Gradient for b1
        dW2: Gradient for W2
        db2: Gradient for b2
        dW3: Gradient for W3
        db3: Gradient for b3
        learning_rate: Learning rate (alpha)

    Returns:
        tuple: (W1, b1, W2, b2, W3, b3) - Updated weights and biases
    """
    # Update weights and biases
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W1, b1, W2, b2, W3, b3

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    """
    Plot training and validation loss and accuracy curves

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        epochs: Number of epochs
    """
    epoch_range = range(1, epochs + 1)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1.plot(epoch_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epoch_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epoch_range, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epoch_range, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    # plt.show()
    plt.savefig('training_metrics.png')

# ============================================================================
# TRAINING LOOP
# ============================================================================

def training_loop(W1, b1, W2, b2, W3, b3, training_data, validation_data, epochs, batch_size, learning_rate):
    """
    Train the multilayer perceptron using mini-batch gradient descent.

    Parameters:
    -----------
    W1, b1, W2, b2, W3, b3 : numpy.ndarray
        Initial weights and biases for all three layers
    training_data : list
        [X_train, Y_train] from get_training_and_validation_data
    validation_data : list
        [X_val, Y_val] from get_training_and_validation_data
    epochs : int
        Number of training epochs
    batch_size : int
        Size of mini-batches
    learning_rate : float
        Learning rate for gradient descent

    Returns:
    --------
    tuple : (W1, b1, W2, b2, W3, b3)
        Trained weights and biases for all three layers
    """
    # Initialize lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Open log file
    with open('logs.txt', 'w') as log_file:
        msg = "\nStarting training..."
        print(msg)
        log_file.write(msg + '\n')

        msg = "=" * 70
        print(msg)
        log_file.write(msg + '\n')

        for epoch in range(epochs):
            # Training phase
            batches = shuffle_and_split_into_mini_batches(training_data, batch_size)
            total_losses = 0
            total_accuracies = 0

            msg = f"\nEpoch {epoch + 1}/{epochs}"
            print(msg)
            log_file.write(msg + '\n')

            msg = f"  Processing {len(batches)} mini-batches..."
            print(msg)
            log_file.write(msg + '\n')

            for X, Y in batches:
                Z1, A1, Z2, A2, _, A3 = apply_feedforward(X, W1, b1, W2, b2, W3, b3)
                dW1, db1, dW2, db2, dW3, db3 = apply_backpropagation(X, Y, A1, A2, A3, Z1, Z2, W2, W3)
                W1, b1, W2, b2, W3, b3 = apply_gradient_descent(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
                total_losses = total_losses + cross_entropy(Y, A3)
                total_accuracies = total_accuracies + calculate_accuracy(Y, A3)

            # Calculate epoch metrics
            epoch_training_loss = total_losses / len(batches)
            epoch_training_accuracy = total_accuracies / len(batches)
            *_, A3_val = apply_feedforward(validation_data[0], W1, b1, W2, b2, W3, b3)
            epoch_validation_loss = cross_entropy(validation_data[1], A3_val)
            epoch_validation_accuracy = calculate_accuracy(validation_data[1], A3_val)

            # Store metrics for plotting
            train_losses.append(epoch_training_loss)
            train_accuracies.append(epoch_training_accuracy)
            val_losses.append(epoch_validation_loss)
            val_accuracies.append(epoch_validation_accuracy)

            # Print epoch summary
            msg = f"  Training Loss:       {epoch_training_loss:.6f}"
            print(msg)
            log_file.write(msg + '\n')

            msg = f"  Training Accuracy:   {epoch_training_accuracy:.4f} ({epoch_training_accuracy * 100:.2f}%)"
            print(msg)
            log_file.write(msg + '\n')

            msg = f"  Validation Loss:     {epoch_validation_loss:.6f}"
            print(msg)
            log_file.write(msg + '\n')

            msg = f"  Validation Accuracy: {epoch_validation_accuracy:.4f} ({epoch_validation_accuracy * 100:.2f}%)"
            print(msg)
            log_file.write(msg + '\n')

        msg = "\n" + "=" * 70
        print(msg)
        log_file.write(msg + '\n')

        msg = "Training completed!"
        print(msg)
        log_file.write(msg + '\n')

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, epochs)

    return W1, b1, W2, b2, W3, b3

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main training function with argument parsing
    """
    print("=" * 70)
    print("MULTILAYER PERCEPTRON - TRAINING")
    print("=" * 70)
    
    # Parse command line arguments
    config = parse_arguments()
    
    # Display configuration
    print("\nTraining Configuration:")
    print("-" * 70)
    print(f"Hidden Layer 1 Size: {config['hidden1_size']} neurons")
    print(f"Hidden Layer 2 Size: {config['hidden2_size']} neurons")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("-" * 70)
    
    # Load data
    data = get_training_and_validation_data('data_training.csv', 'data_validation.csv')

    # Initialize network with parsed configuration
    W1, b1, W2, b2, W3, b3 = initialize_network(
        input_size=30,
        hidden1_size=config['hidden1_size'],
        hidden2_size=config['hidden2_size'],
        output_size=2
    )

    print(f"\nNetwork initialized successfully!")
    print(f"Total parameters: {W1.size + b1.size + W2.size + b2.size + W3.size + b3.size}")

    # Train the network
    W1, b1, W2, b2, W3, b3 = training_loop(
        W1, b1, W2, b2, W3, b3,
        data[0],  # training_data
        data[1],  # validation_data
        config['epochs'],
        config['batch_size'],
        config['learning_rate']
    )

    # Save trained model to JSON
    model_data = {
        'architecture': {
            'input_size': 30,
            'hidden1_size': config['hidden1_size'],
            'hidden2_size': config['hidden2_size'],
            'output_size': 2
        },
        'weights': {
            'W1': W1.tolist(),
            'W2': W2.tolist(),
            'W3': W3.tolist()
        },
        'biases': {
            'b1': b1.tolist(),
            'b2': b2.tolist(),
            'b3': b3.tolist()
        }
    }

    with open('model.json', 'w') as f:
        json.dump(model_data, f, indent=2)

    print("\n" + "=" * 70)
    print("Training finished! Model saved to model.json")
    print("=" * 70)

if __name__ == "__main__":
    main()