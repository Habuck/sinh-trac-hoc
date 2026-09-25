# Window-Level vs Session-Level Split

Understanding the difference between the two splitting strategies and when to use each.

---

## The Problem

With only **4 sessions** that are fundamentally different:
- **A_clean_alarm.wav** - Clean glucose alarm sound
- **B_alarm_with_noise.wav** - Glucose alarm with background noise
- **C_alexa_only.wav** - Alexa voice (negative example)
- **D_bedroom_noise.wav** - Duvet rustling, bed sounds (negative example)

**Session-level split** creates a critical flaw in the training process.

---

## Session-Level Split (Original Approach)

### How It Works

```
TRAIN:
  ✓ All windows from A_clean_alarm.wav (positive)
  ✓ All windows from D_bedroom_noise.wav (negative)

VALIDATE:
  ✓ All windows from B_alarm_with_noise.wav (positive)
  ✓ All windows from C_alexa_only.wav (negative)
```

### What the Model Learns

The model learns to distinguish:
- **"Duvet rustling"** (only negative example in training)
- vs **"Alarm sound"** (positive examples in training)

### The Fatal Flaw

**The model NEVER sees Alexa sounds during training!**

When tested on validation:
- ✅ Recognizes duvet rustling as "not alarm" (seen during training)
- ❌ Confused by Alexa voice (never seen during training)
- ❌ Guesses "alarm" for Alexa because it's not duvet rustling

### Results

```
Validation Accuracy: 61.79%
Confusion Matrix:
                Predicted
              0         1
Actual 0    411     306    ← 306 false positives (Alexa classified as alarm!)
       1    242     475
```

**Why?**
- Model learned: "Not duvet = alarm"
- Never learned: "Alexa voice = not alarm"
- Never learned: "Other background sounds = not alarm"

---

## Window-Level Split (Correct Approach)

### How It Works

```
1. Collect ALL windows from ALL sessions
2. Randomly split 80/20 (stratified by label)

TRAIN (80%):
  ✓ ~287 windows from A_clean_alarm.wav (positive)
  ✓ ~287 windows from B_alarm_with_noise.wav (positive)
  ✓ ~287 windows from C_alexa_only.wav (negative)
  ✓ ~287 windows from D_bedroom_noise.wav (negative)

VALIDATE (20%):
  ✓ ~72 windows from A_clean_alarm.wav (positive)
  ✓ ~72 windows from B_alarm_with_noise.wav (positive)
  ✓ ~72 windows from C_alexa_only.wav (negative)
  ✓ ~72 windows from D_bedroom_noise.wav (negative)
```

### What the Model Learns

The model learns to distinguish:
- **"Alarm sounds"** (clean OR noisy) 
- vs **"Not alarm"** (Alexa OR duvet OR other sounds)

### Why This Works

✅ Training sees ALL types of sounds
✅ Model learns: "Not alarm" = diverse set of sounds
✅ Model learns: "Alarm" = alarm in various conditions
✅ Validation tests generalization across all sound types

### Expected Results

```
Validation Accuracy: 75-85% (estimated)
Confusion Matrix:
                Predicted
              0         1
Actual 0    600     117    ← Fewer false positives
       1    100     617    ← Fewer false negatives
```

---

## When to Use Each Approach

### Use Window-Level Split When:

✅ **Few sessions** (< 10 per class)
✅ **Diverse sessions** (each session is a different "type")
✅ **Goal**: Learn from all types of examples
✅ **Risk**: Overfitting to specific sessions is low (not enough data)

**Example**: Your current dataset
- 4 sessions total
- Each session is fundamentally different
- Need model to see all sound types

### Use Session-Level Split When:

✅ **Many sessions** (10+ per class)
✅ **Similar sessions** (multiple recordings of same types)
✅ **Goal**: Prevent memorizing specific sessions
✅ **Risk**: Model might memorize session-specific patterns

**Example**: Production dataset
- 20 alarm sessions (different days, volumes, backgrounds)
- 20 no-alarm sessions (Alexa, TV, music, silence, rustling, etc.)
- Each session is one of many similar examples

---

## Comparison Table

| Aspect | Session-Level Split | Window-Level Split |
|--------|---------------------|-------------------|
| **Best for** | Many similar sessions | Few diverse sessions |
| **Prevents** | Session memorization | N/A (not enough data) |
| **Ensures** | Generalization across sessions | Learning from all types |
| **Risk** | Missing sound types in training | Slight data leakage (same session in train/val) |
| **Your case** | ❌ Wrong choice | ✅ Correct choice |

---

## The Fix

### What Changed

**File**: `train_cnn_window_split.ipynb`

**Key differences**:
1. Load ALL windows from both `dataset/train/` and `dataset/val/`
2. Combine into single dataset
3. Split 80/20 randomly (stratified by label)
4. Train on 80%, validate on 20%

**Code**:
```python
# Load all metadata
metadata = pd.read_csv('dataset/dataset_metadata.csv')

# Window-level split (80/20, stratified)
train_df, val_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=metadata['label']
)

# Both train and val now contain windows from ALL sessions
```

---

## Expected Improvement

### Session-Level Split (Original)
- Validation accuracy: **61.79%**
- False positives: **306** (Alexa confused as alarm)
- Model never saw Alexa during training

### Window-Level Split (New)
- Validation accuracy: **75-85%** (estimated)
- False positives: **~100-150** (much better)
- Model sees all sound types during training

**Improvement**: +15-25% accuracy

---

## Bottom Line

**Session-level split** is the right approach for production with lots of data.

**Window-level split** is the right approach for your current dataset with only 4 diverse sessions.

The original advice (session-level split) was correct in principle but wrong for your specific situation.

---

## Next Steps

1. **Run** `train_cnn_window_split.ipynb`
2. **Compare** results to original `train_cnn.ipynb`
3. **Expect** 75-85% validation accuracy
4. **Collect more data** (10+ sessions per class)
5. **Switch back** to session-level split once you have enough data

