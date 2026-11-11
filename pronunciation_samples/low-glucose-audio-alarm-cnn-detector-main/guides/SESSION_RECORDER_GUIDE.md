# Audio Session Recorder - Setup Guide

## Purpose

This notebook is designed for collecting raw audio training data for a glucose alarm classification ML project.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (if not already created)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install sounddevice scipy numpy jupyter
```

### 2. Grant Microphone Permission (macOS)

**Important**: macOS requires explicit permission for microphone access.

1. Go to **System Preferences → Security & Privacy → Privacy → Microphone**
2. Enable microphone access for **Terminal** (or your IDE: VS Code, PyCharm, etc.)
3. Restart Jupyter if it's already running

### 3. Start Jupyter

```bash
source venv/bin/activate
jupyter notebook session_recorder.ipynb
```

### 4. Configure and Record

1. In the notebook, modify the configuration cell:
   ```python
   SESSION_TYPE = "glucose_alarm"          # or "no_glucose_alarm"
   BACKGROUND_NOISE = "background_noise"   # or "no_background_noise"
   DURATION_SECONDS = 60                   # 60-300 seconds recommended
   ```

2. Run all cells (Cell → Run All)

3. Recording will start after a 3-second countdown

4. Files are saved to `sessions/` directory

## File Naming Convention

Recordings are automatically named with this format:

```
session_<timestamp>__<session_type>__<background_noise>.wav
```

Examples:
- `session_20260111_213045__glucose_alarm__background_noise.wav`
- `session_20260111_214530__no_glucose_alarm__no_background_noise.wav`
- `session_20260111_220000__glucose_alarm__no_background_noise.wav`

## Audio Specifications

- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (mono)
- **Duration**: User-configurable (60-300 seconds recommended)

## Session Types

Two valid session types:

1. **`glucose_alarm`** - Recording contains glucose alarm sounds
2. **`no_glucose_alarm`** - Recording without glucose alarm sounds

## Background Noise Levels

Two valid background noise levels:

1. **`background_noise`** - Recording has ambient/background noise present
2. **`no_background_noise`** - Recording in quiet environment with minimal background noise

## Workflow for Data Collection

### Recording Glucose Alarm with Background Noise

```python
SESSION_TYPE = "glucose_alarm"
BACKGROUND_NOISE = "background_noise"
DURATION_SECONDS = 120  # 2 minutes
```

Run the recording cell, then trigger your glucose alarm during the recording in a normal environment.

### Recording Glucose Alarm in Quiet Environment

```python
SESSION_TYPE = "glucose_alarm"
BACKGROUND_NOISE = "no_background_noise"
DURATION_SECONDS = 120  # 2 minutes
```

Record in a quiet room with minimal ambient noise.

### Recording Background/Negative Samples with Noise

```python
SESSION_TYPE = "no_glucose_alarm"
BACKGROUND_NOISE = "background_noise"
DURATION_SECONDS = 180  # 3 minutes
```

Run the recording cell in normal environment without alarms.

### Recording Background/Negative Samples in Quiet

```python
SESSION_TYPE = "no_glucose_alarm"
BACKGROUND_NOISE = "no_background_noise"
DURATION_SECONDS = 180  # 3 minutes
```

Record in a quiet environment without alarms.

## Output Directory Structure

```
Alarm ML classification/
├── session_recorder.ipynb
├── sessions/
│   ├── session_20260111_213045__glucose_alarm.wav
│   ├── session_20260111_214530__no_glucose_alarm.wav
│   └── ...
└── SESSION_RECORDER_GUIDE.md
```

## Troubleshooting

### No Audio Recorded / Silent Files

**Most common issue**: Microphone permission not granted

**Solution**:
1. Check System Preferences → Security & Privacy → Privacy → Microphone
2. Ensure Terminal/IDE has microphone access enabled
3. Restart Jupyter
4. Try recording again

### Check Available Audio Devices

Run this in a notebook cell:

```python
import sounddevice as sd
print(sd.query_devices())
```

### Test Your Microphone

Quick test (run in notebook):

```python
import sounddevice as sd
import numpy as np

# Record 3 seconds
test = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()

# Check if audio was captured
max_amplitude = np.max(np.abs(test))
print(f"Max amplitude: {max_amplitude}")

if max_amplitude < 100:
    print("❌ No audio detected - check permissions!")
else:
    print("✅ Microphone working!")
```

## Modifying the Notebook

### Change Sample Rate

```python
SAMPLE_RATE = 44100  # CD quality
```

### Change to Stereo

```python
CHANNELS = 2  # Stereo
```

### Change Output Directory

```python
OUTPUT_DIR = 'my_recordings'
```

## Best Practices

1. **Consistent Environment**: Record in the same location/setup
2. **Label Immediately**: Set `SESSION_TYPE` before each recording
3. **Verify Files**: Check that WAV files are created and have reasonable size
4. **Backup Data**: Regularly backup your `sessions/` directory

## File Size Estimates

Approximate file sizes for mono 16kHz 16-bit recordings:

- 60 seconds: ~1.9 MB
- 120 seconds: ~3.8 MB
- 180 seconds: ~5.6 MB
- 300 seconds: ~9.4 MB

