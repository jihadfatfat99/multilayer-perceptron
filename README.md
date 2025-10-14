# Multilayer Perceptron - Binary Classification

A from-scratch implementation of a multi-layer perceptron for binary classification using ReLU activation functions and softmax output layer. Built for the **42 School** curriculum.

## Project Overview

This project implements a complete neural network training and prediction pipeline without using high-level machine learning libraries. The implementation features:

- **ReLU activation** for hidden layers with proper gradient handling
- **Softmax output layer** for binary classification
- **Xavier/Glorot initialization** optimized for ReLU networks
- **Mini-batch gradient descent** with configurable batch sizes
- **Early stopping** with patience-based monitoring
- **JSON model serialization** for human-readable model storage
- **Comprehensive evaluation metrics** and visualization

Built as part of the 42 School machine learning curriculum to understand neural networks from first principles.

## Architecture

```
Input Layer (30 features)
    ↓
Hidden Layer 1 (24 neurons, ReLU)
    ↓
Hidden Layer 2 (16 neurons, ReLU)
    ↓
Output Layer (2 neurons, Softmax)
```

The architecture is fully configurable via command-line arguments.

## Project Structure

```
multilayer-perceptron/
├── network.py          # Neural network class implementation
├── splitter.py         # Data splitting utility (train/validation)
├── trainer.py          # Training script with early stopping
├── prediction.py       # Prediction and evaluation script
└── README.md          # This file
```

## Core Components

### 1. Neural Network Class (`network.py`)

Object-oriented implementation featuring:
- Forward propagation with ReLU and softmax
- Backpropagation with automatic gradient computation
- Cross-entropy loss calculation
- Prediction and accuracy evaluation methods

### 2. Data Splitter (`splitter.py`)

Splits the dataset into training and validation sets:
- 80/20 train/validation split (default)
- Shuffling with configurable random seed
- Preserves original CSV format

### 3. Trainer (`trainer.py`)

Comprehensive training pipeline:
- Mini-batch gradient descent
- Epoch-by-epoch progress display
- **Proper early stopping implementation**:
  - Tracks consecutive epochs without improvement
  - Saves best weights and biases
  - Resets patience counter on improvement
- Saves model to JSON with normalization parameters
- Generates training history plots (loss and accuracy curves)
- Outputs training log to `training.txt`

### 4. Predictor (`prediction.py`)

Model evaluation and prediction:
- Loads trained model from JSON
- Applies saved normalization parameters
- Displays per-sample predictions with confidence scores
- Computes comprehensive metrics (accuracy, precision, recall, F1)
- Shows confusion matrix
- Calculates cross-entropy loss on test data
- Outputs predictions to `prediction.txt`

## Mathematical Foundations

### Forward Propagation

**Hidden Layers (ReLU)**:
```
Z^(l) = A^(l-1) @ W^(l) + b^(l)
A^(l) = max(0, Z^(l))
```

**Output Layer (Softmax)**:
```
Z^(L) = A^(L-1) @ W^(L) + b^(L)
A^(L) = exp(Z^(L) - max(Z^(L))) / sum(exp(Z^(L) - max(Z^(L))))
```
*(Numerically stable with max subtraction)*

### Loss Function

**Cross-Entropy Loss**:
```
L = -1/N * Σ(y * log(p + ε))
```
where ε = 1e-12 prevents log(0)

### Backward Propagation

**Output Layer Gradient** (Softmax + Cross-Entropy):
```
δ^(L) = (predictions - labels) / batch_size
```

**Hidden Layer Gradient** (ReLU):
```
δ^(l) = (δ^(l+1) @ W^(l+1)^T) ⊙ ReLU'(Z^(l))
```
where ReLU'(z) = 1 if z > 0 else 0

**Parameter Updates**:
```
∂W^(l) = A^(l-1)^T @ δ^(l)
∂b^(l) = sum(δ^(l), axis=0)

W^(l) ← W^(l) - η * ∂W^(l)
b^(l) ← b^(l) - η * ∂b^(l)
```

### Weight Initialization

**Xavier/Glorot for ReLU**:
```
W ~ N(0, sqrt(2/n_in))
b = 0.01
```
- Variance scaled by 2 for ReLU activation
- Small positive bias prevents dead ReLU neurons

## Usage Guide

### Installation

```bash
pip install numpy matplotlib
```

### Step 1: Split Data

```bash
python splitter.py --input data.csv --training train.csv --validation valid.csv
```

**Arguments:**
- `--input`: Path to input CSV file
- `--training`: Output path for training set (default: `train.csv`)
- `--validation`: Output path for validation set (default: `valid.csv`)

**Output:**
```
Loading data from: data.csv
Total samples loaded: 569
Features per sample: 30
Shuffling data with random permutation...
Split ratio: 80% train / 20% validation
Training set size: 455 samples (80.0%)
Validation set size: 114 samples (20.0%)

Split complete:
  Training samples:   455 (80.0%)
  Validation samples: 114 (20.0%)

Saved to:
  training.csv
  validation.csv
```

### Step 2: Train Model

```bash
python trainer.py --training training.csv --validation validation.csv --layer 24 12 --epochs 50 --batch_size 16 --learning_rate 0.0256 --patience 15 --output model.json
```

**Arguments:**
- `--training`: Path to training CSV file (required)
- `--validation`: Path to validation CSV file (required)
- `--layer`: Hidden layer sizes (default: `24 12`)
- `--epochs`: Maximum number of epochs (default: 50)
- `--batch_size`: Mini-batch size (default: 16)
- `--learning_rate`: Learning rate for gradient descent (default: 0.0256)
- `--patience`: Early stopping patience (default: 15)
- `--output`: Output model file path (default: `model.json`)

**Output (per epoch):**
```
======================================================================
TRAINING
======================================================================
Epochs:        100
Batch size:    16
Learning rate: 0.001
Patience:      15
======================================================================
Epoch  1/100 | Train Loss: 0.6234 | Val Loss: 0.6012 | Train Acc: 0.687 | Val Acc: 0.702
Epoch  2/100 | Train Loss: 0.4521 | Val Loss: 0.4389 | Train Acc: 0.801 | Val Acc: 0.815
Epoch  3/100 | Train Loss: 0.3156 | Val Loss: 0.3024 | Train Acc: 0.879 | Val Acc: 0.886
...
Epoch 47/100 | Train Loss: 0.0823 | Val Loss: 0.0891 | Train Acc: 0.978 | Val Acc: 0.971

Early stopping at epoch 52
No improvement for 15 consecutive epochs
Best validation loss: 0.0891 (at epoch 37)

======================================================================
FINAL RESULTS
======================================================================
Training Accuracy:   0.9780 (97.80%)
Validation Accuracy: 0.9649 (96.49%)
Training Loss:       0.0856
Validation Loss:     0.0923

Best Validation Loss:     0.0891 (at epoch 37)
Best Validation Accuracy: 0.9711 (97.11%) (at epoch 37)
======================================================================
```

The training script also:
- Generates plots showing loss and accuracy curves over epochs
- Saves all output to `training.txt`
- Saves the best model (from epoch with lowest validation loss) to JSON

### Step 3: Make Predictions

```bash
python prediction.py --model model.json --input validation.csv
```

**Arguments:**
- `--model`: Path to trained model JSON file (required)
- `--data`: Path to test data CSV file (required)

**Output:**
```
======================================================================
PREDICTION RESULTS
======================================================================
Sample   1 | True: Benign    | Pred: Benign    | Conf: 0.987 | P(B)=0.987 P(M)=0.013 ✓
Sample   2 | True: Malignant | Pred: Malignant | Conf: 0.943 | P(B)=0.057 P(M)=0.943 ✓
Sample   3 | True: Benign    | Pred: Benign    | Conf: 0.995 | P(B)=0.995 P(M)=0.005 ✓
...
(First 20 samples shown)

======================================================================
PERFORMANCE SUMMARY
======================================================================
Cross-Entropy Loss: 0.0891
Accuracy:           0.9680 (96.80%)
Precision:          0.9545 (95.45%)
Recall:             0.9767 (97.67%)
F1 Score:           0.9655 (96.55%)

Confusion Matrix:
                    Predicted
                 Benign  Malignant
Actual  Benign      68        2
        Malignant    1       43

True Negatives:  68
False Positives: 2
False Negatives: 1
True Positives:  43
======================================================================
```

All predictions are saved to `prediction.txt`.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--layer` | `24 12` | Hidden layer sizes (space-separated) |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `16` | Mini-batch size for gradient descent |
| `--learning_rate` | `0.0256` | Step size for parameter updates |
| `--patience` | `15` | Consecutive epochs without improvement before stopping |

## Early Stopping Implementation

The trainer implements proper early stopping:

1. **Track best validation loss**: Always save the best loss encountered
2. **Save best model**: Store weights and biases from best epoch
3. **Monitor patience**: Count consecutive epochs without improvement
4. **Reset on improvement**: When validation loss improves, reset patience counter to 0
5. **Stop training**: When patience counter reaches the limit, stop and use best model

This ensures:
- The model saved is from the best epoch (not the last)
- Training stops only after sustained lack of improvement
- No overfitting to the validation set

## Model JSON Format

```json
{
  "architecture": [30, 24, 16, 2],
  "weights": [
    [[...], [...], ...],  // Layer 1 weights
    [[...], [...], ...],  // Layer 2 weights
    [[...], [...], ...]   // Output layer weights
  ],
  "biases": [
    [...],  // Layer 1 biases
    [...],  // Layer 2 biases
    [...]   // Output layer biases
  ],
  "normalization": {
    "mean": [[...]],  // Feature means from training data
    "std": [[...]]    // Feature std devs from training data
  }
}
```

## Evaluation Metrics

The prediction script computes:

- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision**: `TP / (TP + FP)` - How many predicted positives are actually positive
- **Recall**: `TP / (TP + FN)` - How many actual positives were detected
- **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)` - Harmonic mean
- **Cross-Entropy Loss**: Average loss across all test samples

## Dataset Format

**Expected CSV format:**
```csv
id,diagnosis,feature1,feature2,...,feature30
1,M,17.99,10.38,122.8,...
2,B,20.57,17.77,132.9,...
```

- **Column 1**: Sample ID (ignored)
- **Column 2**: Diagnosis (`M` = Malignant/1, `B` = Benign/0)
- **Columns 3-32**: 30 numerical features

Alternative format (no ID column):
```csv
feature1,feature2,...,feature30,diagnosis
17.99,10.38,122.8,...,M
20.57,17.77,132.9,...,B
```

Both formats are automatically handled by the scripts.

## Implementation Details

### Activation Functions

**ReLU (Hidden Layers)**:
- Fast computation
- Reduces vanishing gradient problem
- Can cause "dead neurons" (mitigated with positive bias initialization)

**Softmax (Output Layer)**:
- Produces probability distribution over classes
- Numerically stable implementation with max subtraction
- Combined with cross-entropy for efficient backpropagation

### Normalization

Features are normalized using training set statistics:
```
X_norm = (X - μ_train) / σ_train
```

The same μ and σ are applied to validation and test sets to ensure consistency.

### Gradient Descent

Mini-batch gradient descent:
- Shuffles training data each epoch
- Updates parameters after each batch
- Balances speed and stability (batch_size = 16)

## Troubleshooting

**Issue**: Training accuracy is high but validation accuracy is low
- **Solution**: Reduce model complexity (fewer/smaller layers), increase patience, or get more data

**Issue**: Model predicts same class for all samples
- **Solution**: Check data normalization, reduce learning rate, or verify labels are balanced

**Issue**: Loss is NaN or inf
- **Solution**: Reduce learning rate, check for invalid values in input data

**Issue**: Training is very slow
- **Solution**: Increase batch size, reduce number of layers, or use fewer epochs

## Requirements

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.3+ (for training plots)

```bash
pip install numpy matplotlib
```

## Project Information

**School**: 42 School
**Project**: Multilayer Perceptron
**Objective**: Implement a neural network from scratch to understand backpropagation, gradient descent, and training dynamics without high-level ML frameworks.

## Key Learning Outcomes

- Understanding forward and backward propagation
- Implementing gradient descent optimization
- Managing overfitting with early stopping
- Evaluating model performance with multiple metrics
- Building a complete ML pipeline from data splitting to prediction

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- Xavier Glorot & Yoshua Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks"

## License

This project is part of the 42 School curriculum. Feel free to use for educational purposes.
