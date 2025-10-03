import numpy as np
import sys

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

def parse_arguments():
    """
    Parse command line arguments for training configuration
    
    Returns:
        dict: Configuration parameters
    """
    # Default values
    config = {
        'hidden1_size': 22,
        'hidden2_size': 15,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.01
    }
    
    # Valid keys
    valid_keys = ['layer', 'epochs', 'batch_size', 'learning_rate']
    
    # Track which keys have been used
    used_keys = set()
    
    # Get training data dimensions once at the beginning
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
            # Collect all numbers until next -- argument or end
            temp_i = i - 1  # Back to value position
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
            
            # Update i to skip all parsed layer sizes
            i = temp_i
            
            # Validation: Check number of layer values
            if len(layer_sizes) == 1:
                print(f"Error: --layer requires exactly 2 values (hidden layer 1 and hidden layer 2 sizes), got only 1")
                sys.exit(1)
            
            if len(layer_sizes) == 0:
                print(f"Warning: No valid layer sizes provided, using defaults")
            else:
                # Validation: Check minimum size (must be at least 4 neurons)
                for idx, size in enumerate(layer_sizes[:2]):  # Check only first 2
                    if size < 4:
                        print(f"Error: Hidden layer {idx + 1} size must be at least 4, got {size}")
                        sys.exit(1)
                
                # Validation: Layer sizes must not exceed number of input features
                n_features = csv_data['features']
                for idx, size in enumerate(layer_sizes[:2]):
                    if size > n_features:
                        print(f"Error: Hidden layer {idx + 1} size ({size}) cannot exceed number of input features ({n_features})")
                        sys.exit(1)
                
                # Validation: Second layer must be <= first layer
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
        
                # Basic range validation - minimum
                if batch_size < 4:
                    print(f"Error: Batch size too small (< 4). Use at least 4 samples per batch for stable gradients.")
                    sys.exit(1)
        
                # Maximum validation based on training data size
                n_train_samples = csv_data['lines']
        
                # Batch size should not exceed training data
                if batch_size > n_train_samples:
                    print(f"Error: Batch size ({batch_size}) cannot be larger than training data size ({n_train_samples})")
                    sys.exit(1)
        
                # Batch size should allow at least 2 batches for effective mini-batch training
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
    
    # Load training data to get actual sample count
    print("\nLoading training data...")
    try:
        data = np.loadtxt('data_training.csv', delimiter=',')
        y_train = data[:, 0].astype(int)
        X_train = data[:, 1:]
        n_samples = X_train.shape[0]
        print(f"Training data loaded: {n_samples} samples, {X_train.shape[1]} features")
    except FileNotFoundError:
        print("Error: 'data_training.csv' not found. Please run split_data.py first.")
        sys.exit(1)
    
    # Calculate batch information
    batch_size = config['batch_size']
    num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
    
    print("\nBatch Information:")
    print("-" * 70)
    print(f"Total batches per epoch: {num_batches}")
    
    # Calculate size of each batch
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        batch_length = end - start
        print(f"  Batch {batch_idx + 1}: {batch_length} samples (indices {start}-{end-1})")
    print("-" * 70)
    
    # Initialize network with parsed configuration
    W1, b1, W2, b2, W3, b3 = initialize_network(
        input_size=30,
        hidden1_size=config['hidden1_size'],
        hidden2_size=config['hidden2_size'],
        output_size=2
    )
    
    print("\nNetwork initialized successfully!")
    print(f"Total parameters: {W1.size + b1.size + W2.size + b2.size + W3.size + b3.size}")
    
    # TODO: Load training data and start training loop
    print("\nReady to start training...")
    print("(Training loop implementation coming next)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()