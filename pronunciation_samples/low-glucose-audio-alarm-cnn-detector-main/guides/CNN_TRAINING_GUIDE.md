# CNN Training Guide

This guide explains how to use `train_cnn.ipynb` to train a convolutional neural network for glucose alarm detection.

---

## Overview

**Purpose**: Train a CNN to classify 1-second audio windows as containing a glucose alarm or not.

**Input**: Windowed dataset from `prepare_dataset.ipynb`

**Output**: 
- Trained CNN model (`glucose_alarm_cnn.pth`)
- Training metrics and visualizations
- Evaluation results on validation set

**Goal**: Proof-of-learnability - demonstrate that a CNN can learn to distinguish glucose alarm sounds from background audio.

---

## Prerequisites

### Required Python Packages

```bash
pip install numpy pandas librosa torch matplotlib scikit-learn
```

**Note**: PyTorch installation may vary by platform. See [pytorch.org](https://pytorch.org) for platform-specific instructions.

### Input Data

You must have already run `prepare_dataset.ipynb` to create:

```
dataset/
├── train/
│   ├── *.wav (windowed audio files)
├── val/
│   ├── *.wav (windowed audio files)
└── dataset_metadata.csv
```

The metadata CSV must contain:
- `filename`: Window filename
- `label`: Binary label (0 or 1)
- `split`: Dataset split (`train` or `val`)

---

## Configuration

The notebook uses these default parameters:

### Audio Parameters
```python
SAMPLE_RATE = 16000          # Hz
WINDOW_LENGTH = 1.0          # seconds
```

### Spectrogram Parameters
```python
N_MELS = 64                  # Number of mel bands
N_FFT = 1024                 # FFT size
HOP_LENGTH = 256             # Hop length for STFT
```

### Training Parameters
```python
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
```

---

## How It Works

### 1. Dataset Loading

**Custom PyTorch Dataset** (`AudioWindowDataset`):
- Loads WAV files from disk
- Computes log-mel spectrograms on-the-fly
- Returns tensors shaped `(1, n_mels, time_frames)`
- Returns binary labels (0 or 1)

**Why log-mel spectrograms?**
- Mel scale matches human auditory perception
- Log scale compresses dynamic range
- 2D representation suitable for CNNs

### 2. Data Loaders

**Training DataLoader**:
- Batch size: 32
- Shuffle: True (randomize order each epoch)
- Loads spectrograms in batches for efficient training

**Validation DataLoader**:
- Batch size: 32
- Shuffle: False (consistent evaluation order)
- Used for monitoring generalization

### 3. CNN Architecture

**GlucoseAlarmCNN**:
```
Input: (batch_size, 1, 64, time_frames)

Conv Block 1: Conv2D(1→32) → BatchNorm → ReLU → MaxPool
Conv Block 2: Conv2D(32→64) → BatchNorm → ReLU → MaxPool
Conv Block 3: Conv2D(64→128) → BatchNorm → ReLU → MaxPool

Global Average Pooling: (batch_size, 128, H, W) → (batch_size, 128, 1, 1)
Flatten: (batch_size, 128)
Dropout: 0.5
Fully Connected: (batch_size, 128) → (batch_size, 1)

Output: (batch_size, 1) - single logit for binary classification
```

**Key features**:
- 3 convolutional blocks for hierarchical feature learning
- Batch normalization for stable training
- Global average pooling to reduce parameters
- Dropout for regularization

### 4. Training Loop

**Loss function**: Binary Cross-Entropy with Logits
- Combines sigmoid activation and BCE loss
- Numerically stable
- Suitable for binary classification

**Optimizer**: Adam
- Adaptive learning rate
- Momentum-based
- Works well for most problems

**Training process**:
1. Forward pass: Compute predictions
2. Compute loss: Compare predictions to labels
3. Backward pass: Compute gradients
4. Update weights: Apply optimizer

**Validation**:
- Run after each epoch
- No gradient computation (faster)
- Track loss and accuracy

**Model saving**:
- Save model when validation loss improves
- Prevents overfitting to training data

### 5. Evaluation

**Metrics computed**:
- **Confusion Matrix**: True/false positives/negatives
- **Accuracy**: Overall correctness
- **Precision**: Of predicted alarms, how many are correct?
- **Recall**: Of actual alarms, how many are detected?
- **F1 Score**: Harmonic mean of precision and recall

**Visualizations**:
- Training vs validation loss curves
- Validation accuracy over epochs

### 6. Sanity Checks

**Random sample inspection**:
- Select 6 random validation windows
- Plot spectrograms
- Show ground truth vs prediction
- Display prediction probability

**Purpose**: Verify the model is learning meaningful patterns, not memorizing noise.

---

## Usage

### Step 1: Prepare Dataset

Ensure you've run `prepare_dataset.ipynb` and have:
- ~474 training windows
- ~474 validation windows
- Metadata CSV

### Step 2: Run Training Notebook

```bash
jupyter notebook train_cnn.ipynb
```

Run all cells and monitor:
- Training progress (loss decreasing)
- Validation metrics (accuracy increasing)
- No severe overfitting (train loss << val loss)

### Step 3: Interpret Results

**Good signs**:
- Validation accuracy > 70%
- F1 score > 0.7
- Training and validation loss both decreasing
- Sanity check predictions look reasonable

**Warning signs**:
- Validation accuracy ~50% (random guessing)
- Training loss << validation loss (overfitting)
- All predictions are the same class (model collapsed)

---

## Expected Results

### With 4 Sessions (2 per class, 60s each)

**Dataset size**:
- Training: ~474 windows
- Validation: ~474 windows

**Expected performance**:
- Validation accuracy: 60-90% (depends on data quality)
- Training time: 5-15 minutes (CPU), 1-3 minutes (GPU)

**Note**: Performance heavily depends on:
- How distinct the glucose alarm sound is
- Background noise levels
- Session diversity

---

## Troubleshooting

### "FileNotFoundError: Missing files"

Ensure `prepare_dataset.ipynb` was run successfully:
```bash
ls dataset/train/ | wc -l  # Should show ~474
ls dataset/val/ | wc -l    # Should show ~474
```

### "CUDA out of memory"

Reduce batch size:
```python
BATCH_SIZE = 16  # or 8
```

### "Validation accuracy stuck at 50%"

Possible causes:
- Model too simple (increase capacity)
- Data too noisy (collect cleaner samples)
- Classes not distinguishable (verify alarm is audible)

### "Training loss decreases but validation loss increases"

This is overfitting. Try:
- Increase dropout: `self.dropout = nn.Dropout(0.7)`
- Reduce model capacity
- Collect more training data
- Add data augmentation

### "All predictions are 0 (or 1)"

Model collapsed to predicting one class. Try:
- Check class balance (should be ~50/50)
- Reduce learning rate: `LEARNING_RATE = 0.0001`
- Reinitialize model and retrain

---

## Next Steps

### Improve Performance

1. **Collect more data**
   - Record more sessions (10+ per class)
   - Vary environments and contexts

2. **Data augmentation**
   - Time masking
   - Frequency masking
   - Noise injection
   - Time stretching

3. **Architecture tuning**
   - Try deeper networks
   - Experiment with different filter sizes
   - Add residual connections

4. **Hyperparameter tuning**
   - Learning rate
   - Batch size
   - Number of mel bands
   - FFT parameters

### Deploy Model

1. **Real-time inference**
   - Load trained model
   - Process audio in sliding windows
   - Aggregate predictions over time

2. **Temporal smoothing**
   - Use majority voting
   - Apply temporal filters
   - Require N consecutive positive predictions

3. **Production deployment**
   - Export to ONNX or TorchScript
   - Optimize for mobile/edge devices
   - Add confidence thresholds

---

## File Reference

- **`train_cnn.ipynb`**: Main training notebook
- **`glucose_alarm_cnn.pth`**: Saved model weights
- **`dataset/`**: Input dataset directory
- **`dataset_metadata.csv`**: Dataset metadata

