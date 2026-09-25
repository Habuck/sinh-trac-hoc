# Dataset Preparation Guide

This guide explains how to use `prepare_dataset.ipynb` to convert session-level audio recordings into window-level training and validation examples.

---

## Overview

**Purpose**: Transform long session recordings into short, overlapping windows suitable for CNN training.

**Input**: Session WAV files in `sessions/` directory

**Output**: 
- Windowed WAV files in `dataset/train/` and `dataset/val/`
- Metadata CSV file (`dataset/dataset_metadata.csv`)

---

## Prerequisites

### Required Python Packages

```bash
pip install numpy librosa soundfile pandas
```

### Input Data

Session files must follow this naming convention:

```
session_<timestamp>__<label>__<context>.wav
```

**Examples:**
- `session_20260111_204751__glucose_alarm__no_background_noise.wav`
- `session_20260111_205217__glucose_alarm__background_noise.wav`
- `session_20260111_205814__no_glucose_alarm__alarms.wav`
- `session_20260111_210208__no_glucose_alarm__background_noise.wav`

**Labels:**
- `glucose_alarm` - Session contains target alarm sound
- `no_glucose_alarm` - Session does NOT contain target alarm sound

---

## Configuration

The notebook uses these default parameters:

```python
SAMPLE_RATE = 16000      # Hz
WINDOW_LENGTH = 1.0      # seconds
HOP_LENGTH = 0.25        # seconds
```

**Window parameters:**
- **Window length**: 1.0 second (16,000 samples)
- **Hop length**: 0.25 seconds (4,000 samples)
- **Overlap**: 75% (0.75 seconds)

**Example**: A 60-second session produces ~237 windows

---

## How It Works

### 1. Load Session Metadata

The notebook scans `sessions/` and parses each filename to extract:
- `session_id`: timestamp (e.g., `20260111_204751`)
- `session_label`: `glucose_alarm` or `no_glucose_alarm`
- `context`: everything after the second `__` (e.g., `background_noise`)

### 2. Session-Level Train/Val Split

**Critical**: The split happens at the session level, NOT the window level.

**Hardcoded split:**
- 1 `glucose_alarm` session → train
- 1 `glucose_alarm` session → val
- 1 `no_glucose_alarm` session → train
- 1 `no_glucose_alarm` session → val

This ensures windows from the same session don't leak between train and validation sets.

### 3. Slice Sessions into Windows

For each session:
1. Load audio (resample to 16kHz, convert to mono)
2. Create overlapping windows:
   - Window length: 1.0s
   - Hop length: 0.25s
3. Track metadata for each window:
   - `session_id`
   - `window_index`
   - `start_time_seconds`

### 4. Assign Window Labels

Labels are inherited from session labels:
- `glucose_alarm` → label = 1
- `no_glucose_alarm` → label = 0

**No filtering or ML is applied** - this is pure dataset construction.

### 5. Export Dataset

**Audio files:**
- Saved to `dataset/train/` and `dataset/val/`
- Filename format: `<session_id>_window_<index>.wav`
- Example: `20260111_204751_window_0042.wav`

**Metadata CSV:**
- Saved to `dataset/dataset_metadata.csv`
- Columns:
  - `filename`: Window filename
  - `session_id`: Source session
  - `window_index`: Window number within session
  - `label`: Binary label (0 or 1)
  - `split`: `train` or `val`
  - `start_time_seconds`: Window start time in source session
  - `context`: Session context (e.g., `background_noise`)

---

## Usage

### Step 1: Prepare Session Files

Ensure you have at least 4 session files in `sessions/`:
- 2 `glucose_alarm` sessions
- 2 `no_glucose_alarm` sessions

### Step 2: Run the Notebook

Open `prepare_dataset.ipynb` in Jupyter and run all cells.

### Step 3: Verify Output

The notebook includes sanity checks:
- File counts per split
- Class balance per split
- Session leakage verification

**Expected output structure:**
```
dataset/
├── train/
│   ├── 20260111_204751_window_0000.wav
│   ├── 20260111_204751_window_0001.wav
│   ├── 20260111_210208_window_0000.wav
│   └── ...
├── val/
│   ├── 20260111_205217_window_0000.wav
│   ├── 20260111_205814_window_0000.wav
│   └── ...
└── dataset_metadata.csv
```

---

## Output Examples

### Metadata CSV Sample

```csv
filename,session_id,window_index,label,split,start_time_seconds,context
20260111_204751_window_0000.wav,20260111_204751,0,1,train,0.0,no_background_noise
20260111_204751_window_0001.wav,20260111_204751,1,1,train,0.25,no_background_noise
20260111_204751_window_0002.wav,20260111_204751,2,1,train,0.5,no_background_noise
```

### Console Output Sample

```
Processing sessions...

Processing 20260111_204751 (glucose_alarm, train)...
  Loaded: 60.00s, 960000 samples
  Created 237 windows

Processing 20260111_205217 (glucose_alarm, val)...
  Loaded: 60.00s, 960000 samples
  Created 237 windows

Total windows created: 948

Class balance per split:
split  label
train  0        237
       1        237
val    0        237
       1        237
```

---

## Important Notes

### Session-Level Split

✅ **Correct**: Windows from session A are all in train, windows from session B are all in val

❌ **Incorrect**: Some windows from session A in train, other windows from session A in val

This prevents data leakage and ensures the model generalizes to new sessions.

### No ML or Filtering

This notebook does NOT:
- Train any models
- Compute spectrograms
- Apply energy thresholds
- Filter windows based on content

It only performs dataset construction. Feature extraction and model training happen in separate notebooks.

### Extending the Split

To use more sessions, modify the hardcoded split logic in Section 3:

```python
# Example: Use first 3 sessions for train, next 2 for val
glucose_train_indices = glucose_alarm_sessions.index[:3]
glucose_val_indices = glucose_alarm_sessions.index[3:5]
```

---

## Next Steps

After running this notebook:

1. **Load windowed audio** for training
2. **Compute features** (e.g., spectrograms, MFCCs)
3. **Train CNN model** using the windowed data
4. **Evaluate** on the validation set

---

## Troubleshooting

### "No glucose_alarm sessions found"

Ensure you have session files with `glucose_alarm` in the filename.

### "Only 1 glucose_alarm session found"

You need at least 2 sessions of each type for the hardcoded split. Add more sessions or modify the split logic.

### Session leakage detected

The sanity check found the same session in both train and val. This should not happen with the default code - check for modifications.

### File count mismatch

If the number of files doesn't match the metadata CSV, check for errors during the export step.

---

## File Reference

- **`prepare_dataset.ipynb`**: Main notebook
- **`sessions/`**: Input directory (session WAV files)
- **`dataset/train/`**: Output directory (training windows)
- **`dataset/val/`**: Output directory (validation windows)
- **`dataset/dataset_metadata.csv`**: Output metadata file

