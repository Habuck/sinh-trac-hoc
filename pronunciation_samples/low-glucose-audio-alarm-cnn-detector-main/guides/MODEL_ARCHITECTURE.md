# Model Architecture Reference

Technical reference for the CNN architecture used in glucose alarm detection.

---

## Overview

**Model**: GlucoseAlarmCNN

**Task**: Binary classification (glucose alarm vs no alarm)

**Input**: Log-mel spectrogram of 1-second audio window

**Output**: Single logit (probability after sigmoid)

---

## Input Preprocessing

### Audio to Spectrogram Pipeline

```
Raw Audio (WAV file)
    ↓
Load with librosa (16kHz, mono)
    ↓
Compute Mel Spectrogram
    - n_mels: 64
    - n_fft: 1024
    - hop_length: 256
    ↓
Convert to log scale (dB)
    ↓
Normalize to [0, 1] or [-1, 1]
    ↓
Add channel dimension
    ↓
Tensor: (1, 64, time_frames)
```

### Spectrogram Dimensions

For a 1-second audio window at 16kHz:
- **Samples**: 16,000
- **Time frames**: `(16000 - 1024) / 256 + 1 ≈ 63`
- **Mel bands**: 64
- **Final shape**: `(1, 64, 63)`

---

## Network Architecture

### Layer-by-Layer Breakdown

```
Input: (batch_size, 1, 64, 63)

┌─────────────────────────────────────────────────────────────┐
│ Convolutional Block 1                                       │
├─────────────────────────────────────────────────────────────┤
│ Conv2D(1 → 32, kernel=3x3, padding=1)                       │
│ BatchNorm2D(32)                                             │
│ ReLU                                                        │
│ MaxPool2D(kernel=2x2, stride=2)                             │
│ Output: (batch_size, 32, 32, 31)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Convolutional Block 2                                       │
├─────────────────────────────────────────────────────────────┤
│ Conv2D(32 → 64, kernel=3x3, padding=1)                      │
│ BatchNorm2D(64)                                             │
│ ReLU                                                        │
│ MaxPool2D(kernel=2x2, stride=2)                             │
│ Output: (batch_size, 64, 16, 15)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Convolutional Block 3                                       │
├─────────────────────────────────────────────────────────────┤
│ Conv2D(64 → 128, kernel=3x3, padding=1)                     │
│ BatchNorm2D(128)                                            │
│ ReLU                                                        │
│ MaxPool2D(kernel=2x2, stride=2)                             │
│ Output: (batch_size, 128, 8, 7)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Global Average Pooling                                      │
├─────────────────────────────────────────────────────────────┤
│ AdaptiveAvgPool2D(output_size=(1, 1))                       │
│ Output: (batch_size, 128, 1, 1)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Classifier                                                  │
├─────────────────────────────────────────────────────────────┤
│ Flatten: (batch_size, 128)                                  │
│ Dropout(p=0.5)                                              │
│ Linear(128 → 1)                                             │
│ Output: (batch_size, 1)                                     │
└─────────────────────────────────────────────────────────────┘

Output: Single logit per sample
```

---

## Model Parameters

### Total Parameters

Approximately **~50,000 parameters**:

- **Conv Block 1**: ~300 parameters
  - Conv2D: 1×32×3×3 = 288
  - BatchNorm: 64
  
- **Conv Block 2**: ~18,500 parameters
  - Conv2D: 32×64×3×3 = 18,432
  - BatchNorm: 128
  
- **Conv Block 3**: ~73,800 parameters
  - Conv2D: 64×128×3×3 = 73,728
  - BatchNorm: 256
  
- **Fully Connected**: ~129 parameters
  - Linear: 128×1 + 1 = 129

### Trainable vs Non-Trainable

All parameters are trainable (no frozen layers).

---

## Training Configuration

### Loss Function

**Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`)

```python
loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
```

Where:
- `x` = model output (logit)
- `σ(x)` = sigmoid(x) = 1 / (1 + e^(-x))
- `y` = ground truth label (0 or 1)

**Why BCEWithLogitsLoss?**
- Numerically stable (combines sigmoid + BCE)
- Avoids numerical overflow/underflow
- Standard for binary classification

### Optimizer

**Adam** (Adaptive Moment Estimation)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Hyperparameters**:
- Learning rate: 0.001
- Beta1: 0.9 (default)
- Beta2: 0.999 (default)
- Epsilon: 1e-8 (default)

**Why Adam?**
- Adaptive learning rates per parameter
- Works well with sparse gradients
- Requires minimal tuning

### Regularization

1. **Dropout**: 0.5 before final layer
   - Prevents overfitting
   - Forces redundant representations

2. **Batch Normalization**: After each conv layer
   - Stabilizes training
   - Allows higher learning rates
   - Acts as regularization

---

## Inference

### Forward Pass

```python
# Load model
model = GlucoseAlarmCNN(n_mels=64)
model.load_state_dict(torch.load('glucose_alarm_cnn.pth'))
model.eval()

# Prepare input
spectrogram = compute_log_mel_spectrogram(audio)  # Shape: (1, 64, 63)
input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)  # Add batch dim

# Get prediction
with torch.no_grad():
    logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()
    prediction = 1 if probability > 0.5 else 0
```

### Decision Threshold

**Default**: 0.5

**Adjustable** based on use case:
- **High recall** (catch all alarms): threshold = 0.3
- **High precision** (avoid false alarms): threshold = 0.7

---

## Performance Characteristics

### Computational Cost

**Training**:
- ~5-15 minutes on CPU (20 epochs, ~1000 samples)
- ~1-3 minutes on GPU

**Inference**:
- ~1-5 ms per window on CPU
- ~0.1-0.5 ms per window on GPU

### Memory Requirements

**Model size**: ~200 KB (saved weights)

**Runtime memory**:
- Model: ~1 MB
- Batch (32 samples): ~2 MB
- Total: ~5-10 MB

---

## Design Rationale

### Why This Architecture?

1. **Small and fast**: Suitable for edge deployment
2. **Proven design**: Standard CNN pattern for audio
3. **Global pooling**: Reduces parameters, prevents overfitting
4. **Batch normalization**: Stable training

### Alternatives Considered

1. **Deeper networks**: More capacity but risk overfitting with small dataset
2. **Residual connections**: Better gradient flow but added complexity
3. **Attention mechanisms**: Powerful but overkill for this task
4. **Recurrent layers**: Good for sequences but slower inference

---

## Limitations

1. **Window-level only**: No temporal context across windows
2. **Fixed input size**: Requires exactly 1-second windows
3. **No data augmentation**: Could improve generalization
4. **Simple architecture**: May underfit complex patterns

---

## Future Improvements

1. **Add data augmentation**:
   - Time masking
   - Frequency masking
   - Mixup

2. **Temporal modeling**:
   - Add LSTM after CNN
   - Use attention over time
   - Aggregate predictions

3. **Architecture search**:
   - Try EfficientNet
   - Experiment with ResNet
   - Use neural architecture search

4. **Multi-task learning**:
   - Predict alarm type
   - Estimate urgency level
   - Detect other sounds

---

## References

- **Mel spectrograms**: [librosa documentation](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)
- **CNN for audio**: Hershey et al., "CNN Architectures for Large-Scale Audio Classification" (2017)
- **PyTorch**: [pytorch.org](https://pytorch.org)

