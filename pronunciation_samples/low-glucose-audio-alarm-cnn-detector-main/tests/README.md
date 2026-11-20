# Testing and Debugging Tools

This directory contains scripts and files used for testing, debugging, and analyzing the alarm detection system.

## Testing Scripts

### `label_training_data.py` ⭐ **RECOMMENDED**
**Interactive tool for manually labeling/relabeling training or validation data.**

**Usage:**
```bash
cd tests
source ../venv/bin/activate
python3 label_training_data.py
```

**Configuration (edit the file to customize):**
- `DATASET_SPLIT = 'train'` - Change to `'val'` to label validation data
- `START_FROM_FILE = None` - Set to a filename to start from a specific file
  - Example: `START_FROM_FILE = "20260111_204751_window_0050.wav"`

**What it does:**
- Shows waveform and spectrogram for each audio file
- Plays the audio automatically
- You press 'y' for alarm, 'n' for no alarm, 'r' to replay, 's' to skip
- Updates `dataset_metadata.csv` automatically
- Saves progress every 10 files (can resume later)
- Shows statistics and progress

**Controls:**
- `y` or `1` = ALARM present
- `n` or `0` = NO alarm (silence/noise)
- `r`       = REPLAY audio (if you want to hear it again)
- `s`       = SKIP this file
- `q`       = QUIT and save progress

**Features:**
- Works on both train and val datasets
- Can start from a specific file (useful for resuming or targeting specific files)
- Resumes from where you left off if interrupted
- Backs up metadata before starting
- Visual feedback with waveform and spectrogram
- Audio playback with replay option
- Progress tracking (separate for train and val)

### `verify_what_microphone_hears.py`
Records audio from the microphone and plays it back to verify that the microphone is capturing sound correctly.

**Usage:**
```bash
python3 verify_what_microphone_hears.py
```

**What it does:**
- Records 3 seconds of audio
- Saves to `captured_audio_TIMESTAMP.wav`
- Plays back the recording
- Shows RMS and peak levels

### `test_alarm_audio.py`
Quick test script for real-time alarm detection using the trained model.

**Usage:**
```bash
python3 test_alarm_audio.py
```

**What it does:**
- Records audio in real-time
- Runs inference with the latest model
- Shows probability and prediction

## Analysis Scripts

### `analyze_all_training_files.py`
Performs frequency analysis on ALL training files to identify mislabeled data.

**Usage:**
```bash
python3 analyze_all_training_files.py
```

**What it does:**
- Analyzes every file labeled as "alarm" in the training set
- Computes peak frequency for each file
- Identifies files with peak < 2000 Hz (likely mislabeled)
- Generates visualization and CSV report
- **Takes several minutes to run (~700 files)**

**Output:**
- `training_frequency_analysis_results.csv` - Detailed results for each file
- `complete_training_analysis.png` - Visualization of frequency distribution

### `fix_mislabeled_files.py`
Fixes mislabeled files based on the frequency analysis results.

**Usage:**
```bash
python3 fix_mislabeled_files.py
```

**Prerequisites:**
- Must run `analyze_all_training_files.py` first

**What it does:**
- Loads results from `training_frequency_analysis_results.csv`
- Shows files to be relabeled
- Asks for confirmation
- Changes label from 1 (alarm) to 0 (non-alarm) for low-frequency files
- Backs up original metadata

### `clean_training_dataset.py`
Alternative cleaning script that analyzes files on-the-fly (slower but more thorough).

**Usage:**
```bash
python3 clean_training_dataset.py
```

**What it does:**
- Analyzes training files in real-time
- Identifies mislabeled files
- Offers options to relabel or move files
- **Takes longer than the two-step approach above**

## Analysis Results

### `training_frequency_analysis_results.csv`
Complete frequency analysis results for all training files labeled as "alarm".

**Columns:**
- `index` - Row index in metadata
- `filename` - Audio file name
- `session_id` - Recording session ID
- `window_index` - Window number in session
- `peak_hz` - Peak frequency in Hz
- `rms` - RMS energy level
- `is_mislabeled` - True if peak < 2000 Hz

### `complete_training_analysis.png`
Visualization showing:
- Distribution of peak frequencies
- Peak frequency vs window index
- RMS energy distribution
- Peak frequency vs RMS (color-coded by mislabeled status)

### `captured_audio_*.wav`
Audio files captured during testing/debugging sessions.

## Key Findings

From the frequency analysis:
- **74.3% of files labeled as "alarm" are mislabeled**
- Only 182 out of 708 files have peak > 2000 Hz (actual alarms)
- 526 files have peak < 2000 Hz (mostly 0-96 Hz silence/noise)
- This explains why the model fails to detect real alarms

## Recommended Workflow

### Option 1: Manual Labeling (BEST - Most Accurate) ⭐

1. **Label the training data manually:**
   ```bash
   cd tests
   source ../venv/bin/activate
   python3 label_training_data.py
   ```
   - Listen to each file and label it correctly
   - Can do in batches (progress is saved)
   - Takes time but ensures 100% accuracy

2. **Retrain the model:**
   - Open `train_cnn_window_split.ipynb`
   - Set `LOAD_MODEL_PATH = None`
   - Run all cells

3. **Test with live audio:**
   ```bash
   cd tests
   python3 test_alarm_audio.py
   ```

### Option 2: Automated Analysis (FASTER - Less Accurate)

1. **Analyze the training data:**
   ```bash
   cd tests
   source ../venv/bin/activate
   python3 analyze_all_training_files.py
   ```

2. **Review the results:**
   - Open `complete_training_analysis.png`
   - Check `training_frequency_analysis_results.csv`

3. **Fix mislabeled files:**
   ```bash
   python3 fix_mislabeled_files.py
   ```
   - Uses 2000 Hz threshold (may have false positives/negatives)

4. **Retrain the model:**
   - Open `train_cnn_window_split.ipynb`
   - Set `LOAD_MODEL_PATH = None`
   - Run all cells

5. **Test with live audio:**
   ```bash
   python3 test_alarm_audio.py
   ```

## Notes

- All scripts assume they're run from the `tests` directory or with proper paths
- The analysis scripts require the virtual environment to be activated
- Backup files are created automatically before making changes

