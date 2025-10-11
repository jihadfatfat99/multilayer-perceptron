import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from neural_network.nn_utils import get_data, get_mean_std, normalize_data, shuffle_and_split_into_mini_batches
from neural_network.NeuralNetwork import NeuralNetwork

def parse_arguments():
    """
    Parse command line arguments for training configuration

    Returns:
        dict: Configuration parameters
    """
    # Configuration (no defaults - user must provide all values)
    config = {}

    # Valid and required keys
    valid_keys = ['layer', 'epochs', 'batch_size', 'learning_rate', 'training_filename', 'validation_filename']
    required_keys = {'layer', 'epochs', 'batch_size', 'learning_rate', 'training_filename', 'validation_filename'}

    # Track which keys have been used
    used_keys = set()

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
            # No value provided
            print(f"Error: No value provided for '{arg}'")
            sys.exit(1)
        
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
                print(f"Error: No valid layer sizes provided for '--layer'")
                sys.exit(1)
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

        elif key == 'training_filename':
            # Validate file existence
            if not os.path.exists(value):
                print(f"Error: Training file '{value}' does not exist")
                sys.exit(1)
            # Validate file extension
            if not value.endswith('.csv'):
                print(f"Error: Training filename must end with '.csv', got '{value}'")
                sys.exit(1)
            config['training_filename'] = value

        elif key == 'validation_filename':
            # Validate file existence
            if not os.path.exists(value):
                print(f"Error: Validation file '{value}' does not exist")
                sys.exit(1)
            # Validate file extension
            if not value.endswith('.csv'):
                print(f"Error: Validation filename must end with '.csv', got '{value}'")
                sys.exit(1)
            config['validation_filename'] = value

    # Check if all required keys were provided
    missing_keys = required_keys - used_keys
    if missing_keys:
        print(f"Error: Missing required arguments: {', '.join(['--' + k for k in sorted(missing_keys)])}")
        print(f"Required arguments: {', '.join(['--' + k for k in sorted(required_keys)])}")
        sys.exit(1)

    return config

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
    print(f"Training File: {config['training_filename']}")
    print(f"Validation File: {config['validation_filename']}")
    print("-" * 70)

    # Load data
    # data = get_training_and_validation_data(config['training_filename'], config['validation_filename'])
    training_X, training_Y = get_data(config['training_filename'])
    validation_X, validation_Y = get_data(config['validation_filename'])
    mean, std = get_mean_std(training_X)
    training_X = normalize_data(training_X, mean, std)
    validation_X = normalize_data(validation_X, mean, std)
    print(training_X)
    print(training_Y)
    print(validation_X)
    print(validation_Y)

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
            batches = shuffle_and_split_into_mini_batches(training_X, training_Y, config['batch_size'])
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
            epoch_validation_loss, epoch_validation_accuracy = nn.predict(validation_X, validation_Y)

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