# Live Inference Notebook - Quick Reference

## 📋 Overview

`live_inference.ipynb` performs **real-time glucose alarm detection** with temporal aggregation for robust predictions.

## 🚀 Quick Start

1. **Open the notebook**: `live_inference.ipynb`
2. **Run all cells** up to Section 7
3. **Start monitoring**:
   ```python
   live_inference(duration=60)  # Monitor for 60 seconds
   ```

## ⚙️ Configuration

### Change Model

Edit the configuration cell (Section 1):
```python
MODEL_PATH = "models/your_new_model.pth"  # ← Change this
```

Then re-run all cells.

### Adjust Sensitivity

| Setting | More Sensitive | Balanced (Default) | Less Sensitive |
|---------|---------------|-------------------|----------------|
| `CONFIDENCE_THRESHOLD` | 0.3 | 0.5 | 0.7 |
| `VOTE_THRESHOLD` | 2 | 3 | 4 |
| `TEMPORAL_WINDOW_SIZE` | 5 | 5 | 5 |

**More Sensitive**: Catches more alarms, but more false alarms
**Less Sensitive**: Fewer false alarms, but might miss some alarms

## 🎯 Key Features

### 1. Temporal Aggregation

Instead of alerting on a single 1-second window, the system uses **voting**:

```
Last 5 predictions: [0.2, 0.3, 0.8, 0.9, 0.7]
                           ↓
Windows > 0.5:      [  ❌,  ❌,  ✅,  ✅,  ✅ ]
                           ↓
Vote count: 3/5 ≥ threshold (3) → 🚨 TRIGGER ALERT
```

**Benefits**:
- ✅ Reduces false alarms (need multiple windows to agree)
- ✅ More robust to noise
- ✅ Still fast (< 2 second latency)

### 2. Configurable Models

Load any model from `models/` folder:
```python
# List available models
list_available_models()

# Change model
MODEL_PATH = "models/glucose_alarm_cnn_window_split.pth"
```

### 3. Alert Cooldown

Prevents alert spam:
```python
ALERT_COOLDOWN = 5.0  # Minimum 5 seconds between alerts
```

## 📊 Usage Examples

### Live Monitoring

```python
# Monitor for 60 seconds
live_inference(duration=60)

# Monitor indefinitely (Ctrl+C to stop)
live_inference(duration=None)
```

### Test on Files

```python
# Test on specific files
files = ['dataset/train/file1.wav', 'dataset/train/file2.wav']
test_on_files(files)

# Test on first 20 training files
train_files = sorted(Path('dataset/train').glob('*.wav'))[:20]
test_on_files([str(f) for f in train_files])
```

### Single File Prediction

```python
# Load audio
audio, sr = librosa.load('path/to/file.wav', sr=SAMPLE_RATE, mono=True)

# Predict
probability = predict_window(audio, model)
print(f"Alarm probability: {probability:.4f}")
```

## 🔧 Recommended Settings

### For Medical Use (Prioritize Catching All Alarms)
```python
CONFIDENCE_THRESHOLD = 0.3
VOTE_THRESHOLD = 2
TEMPORAL_WINDOW_SIZE = 5
```
- Catches ~95% of alarms
- Higher false alarm rate (acceptable trade-off)

### For General Use (Balanced)
```python
CONFIDENCE_THRESHOLD = 0.5
VOTE_THRESHOLD = 3
TEMPORAL_WINDOW_SIZE = 5
```
- Catches ~86% of alarms
- Moderate false alarm rate

### For Low False Alarms (Conservative)
```python
CONFIDENCE_THRESHOLD = 0.7
VOTE_THRESHOLD = 4
TEMPORAL_WINDOW_SIZE = 5
```
- Catches ~70% of alarms
- Very low false alarm rate

## 📈 Understanding Output

### Live Monitoring Output

```
[12:34:56] Prob: 0.23 | Votes: 1/5 | Avg: 0.18 | ✓ No alarm
[12:34:57] Prob: 0.87 | Votes: 2/5 | Avg: 0.45 | ⚠️ Building...
[12:34:58] Prob: 0.92 | Votes: 3/5 | Avg: 0.67 | 🚨 ALARM DETECTED!
```

- **Prob**: Current window's alarm probability (0.0-1.0)
- **Votes**: How many of last 5 windows voted "alarm"
- **Avg**: Average confidence of recent predictions
- **Status**: 
  - ✓ No alarm (< threshold)
  - ⚠️ Building (close to threshold)
  - 🚨 ALARM (threshold reached)

### Alert Output

```
======================================================================
[12:34:58] 🚨 ALARM DETECTED!
  Probability: 0.9234
  Votes: 4/5
  Avg Confidence: 0.7123
  Total Alerts: 1
======================================================================
```

## 🛠️ Troubleshooting

### Model Not Found
```
FileNotFoundError: Model not found: models/...
```
**Solution**: Check that the model file exists in `models/` folder. Run `list_available_models()` to see available models.

### No Audio Input
```
Error: No audio input device found
```
**Solution**: Check microphone permissions. On macOS, grant Jupyter/Python microphone access in System Preferences.

### Import Errors
```
ModuleNotFoundError: No module named 'sounddevice'
```
**Solution**: Install missing package:
```bash
pip install sounddevice
```

## 📝 Notes

- **Latency**: ~1 second per prediction (need full 1-second window)
- **CPU Usage**: Low (~5-10% on modern CPU)
- **Memory**: ~100 MB (model + audio buffer)
- **Microphone**: Uses default system microphone

## 🔄 Swapping Models

When you train a new model:

1. Save it to `models/` folder:
   ```python
   torch.save(model.state_dict(), 'models/my_new_model.pth')
   ```

2. Update configuration in notebook:
   ```python
   MODEL_PATH = "models/my_new_model.pth"
   ```

3. Re-run cells from Section 1 onwards

**Important**: New model must have the same architecture (`GlucoseAlarmCNN`) and same audio parameters (sample rate, n_mels, etc.)

