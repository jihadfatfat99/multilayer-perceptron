import numpy as np
import sys
import matplotlib.pyplot as plt
from neural_network.nn_utils import sigmoid, sigmoid_prime, softmax

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
        'layers': [22, 16],  # Default: 2 hidden layers
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

            if len(layer_sizes) == 0:
                print(f"Warning: No valid layer sizes provided, using defaults")
            else:
                config['layers'] = layer_sizes
        
        elif key == 'epochs':
            try:
                epochs = int(value)
                if epochs <= 0:
                    print(f"Error: Epochs must be positive, got {epochs}")
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
    for i, layer_size in enumerate(config['layers'], 1):
        print(f"Hidden Layer {i} Size: {layer_size} neurons")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("-" * 70)

    # Load data
    data = get_training_and_validation_data('data_training.csv', 'data_validation.csv')

    # Initialize network using NeuralNetwork class
    from neural_network.NeuralNetwork import NeuralNetwork

    nn = NeuralNetwork(input_size=30)
    for layer_size in config['layers']:
        nn.add_layer(neurons=layer_size, layer_type='hidden', activation='sigmoid')
    nn.add_layer(neurons=2, layer_type='output', activation='softmax')
    nn.initialize()

    print(f"\nNetwork initialized successfully!")
    nn.summary()

    # Train the network
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    with open('logs.txt', 'w') as log_file:
        msg = "\nStarting training..."
        print(msg)
        log_file.write(msg + '\n')

        msg = "=" * 70
        print(msg)
        log_file.write(msg + '\n')

        for epoch in range(config['epochs']):
            # Training phase
            batches = shuffle_and_split_into_mini_batches(data[0], config['batch_size'])
            total_losses = 0
            total_accuracies = 0

            msg = f"\nEpoch {epoch + 1}/{config['epochs']}"
            print(msg)
            log_file.write(msg + '\n')

            msg = f"  Processing {len(batches)} mini-batches..."
            print(msg)
            log_file.write(msg + '\n')

            for X, Y in batches:
                loss, accuracy = nn.train_batch(X, Y, config['learning_rate'])
                total_losses += loss
                total_accuracies += accuracy

            # Calculate epoch metrics
            epoch_training_loss = total_losses / len(batches)
            epoch_training_accuracy = total_accuracies / len(batches)
            epoch_validation_loss, epoch_validation_accuracy = nn.predict(data[1][0], data[1][1])

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
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, config['epochs'])

    # Save trained model
    nn.save('model.json')

    print("\n" + "=" * 70)
    print("Training finished! Model saved to model.json")
    print("=" * 70)

if __name__ == "__main__":
    main()