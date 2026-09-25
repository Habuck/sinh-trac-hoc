#!/usr/bin/env python3
"""
Interactive tool for manually labeling training data.

Features:
- Plays each audio file
- Shows waveform and spectrogram
- Press 'y' for alarm, 'n' for no alarm, 's' to skip, 'r' to replay
- Updates dataset_metadata.csv automatically
- Saves progress so you can resume later
- Shows statistics and progress
- Works on both train and val datasets
- Can start from a specific file
"""
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import sounddevice as sd
import json
from datetime import datetime
import sys

print("="*70)
print("INTERACTIVE DATA LABELING TOOL")
print("="*70)
print()

# ============================================================================
# CONFIGURATION - Edit these to customize
# ============================================================================
DATASET_DIR = Path("../datasets/dataset_w1.5s_h0.25s_20260314")
METADATA_FILE = DATASET_DIR / "dataset_metadata.csv"

# Which dataset to label: 'train' or 'val'
DATASET_SPLIT = 'val'  # Change to 'val' to label validation data

# Start from a specific file (leave as None to start from beginning)
# Example: START_FROM_FILE = "20260111_204751_window_0050.wav"
START_FROM_FILE = '20260111_205217_window_0618.wav'

# Progress file (separate for train and val)
PROGRESS_FILE = Path(f"labeling_progress_{DATASET_SPLIT}.json")
# ============================================================================

# Set data directory based on split
DATA_DIR = DATASET_DIR / DATASET_SPLIT

print(f"Dataset split: {DATASET_SPLIT.upper()}")
print(f"Data directory: {DATA_DIR}")
print()

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(METADATA_FILE)
dataset_files = metadata[metadata['split'] == DATASET_SPLIT].copy()

print(f"Total {DATASET_SPLIT} files: {len(dataset_files)}")
print()

# Load progress if exists
labeled_files = set()
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
        labeled_files = set(progress.get('labeled_files', []))
    print(f"Resuming from previous session...")
    print(f"Already labeled: {len(labeled_files)} files")
    print()

# Filter to unlabeled files
unlabeled = dataset_files[~dataset_files['filename'].isin(labeled_files)]

# If START_FROM_FILE is specified, skip to that file
if START_FROM_FILE:
    start_idx = unlabeled[unlabeled['filename'] == START_FROM_FILE].index
    if len(start_idx) > 0:
        unlabeled = unlabeled.loc[start_idx[0]:]
        print(f"Starting from file: {START_FROM_FILE}")
    else:
        print(f"⚠️  START_FROM_FILE '{START_FROM_FILE}' not found in unlabeled files")
        print(f"Starting from beginning instead...")

print(f"Remaining to label: {len(unlabeled)} files")
print()

# Backup metadata before starting
backup_file = METADATA_FILE.with_suffix('.csv.backup_before_labeling')
if not backup_file.exists():
    metadata.to_csv(backup_file, index=False)
    print(f"✓ Metadata backed up to: {backup_file.name}")
    print()

print("="*70)
print("INSTRUCTIONS:")
print("="*70)
print("For each audio file:")
print("  - Listen to the audio (plays automatically)")
print("  - Look at the waveform and spectrogram")
print("  - Press:")
print("    'y' or '1' = ALARM present")
print("    'n' or '0' = NO alarm (silence/noise)")
print("    'r'       = REPLAY audio")
print("    's'       = SKIP this file")
print("    'q'       = QUIT and save progress")
print()
print("The alarm sound is a high-pitched beep around 3000 Hz")
print("="*70)
print()

input("Press Enter to start labeling...")
print()

# Statistics
stats = {
    'labeled_as_alarm': 0,
    'labeled_as_no_alarm': 0,
    'skipped': 0,
    'total_labeled': len(labeled_files)
}

def play_audio(audio, sr=16000):
    """Play audio through speakers"""
    sd.play(audio, sr)
    sd.wait()

def show_audio_visualization(audio, sr, filename):
    """Show waveform and spectrogram"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0])
    axes[0].set_title(f'Waveform: {filename}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', 
                                    y_axis='mel', ax=axes[1], cmap='viridis')
    axes[1].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# Main labeling loop
try:
    for idx, (row_idx, row) in enumerate(unlabeled.iterrows()):
        filename = row['filename']
        filepath = DATA_DIR / filename

        if not filepath.exists():
            print(f"⚠️  File not found: {filename}, skipping...")
            continue

        # Clear previous plot
        plt.close('all')

        # Load audio
        audio, sr = librosa.load(filepath, sr=16000, mono=True)

        # Show progress
        print(f"\n{'='*70}")
        print(f"File {idx + 1}/{len(unlabeled)}: {filename}")
        print(f"Current label: {'ALARM' if row['label'] == 1 else 'NO ALARM'}")
        print(f"Session: {row['session_id']}, Window: {row['window_index']}")
        print(f"{'='*70}")

        # Show visualization
        show_audio_visualization(audio, sr, filename)

        # Play audio
        print("\n🔊 Playing audio...")
        play_audio(audio, sr)

        # Get user input
        while True:
            response = input("\nIs there an ALARM? (y/n/r/s/q): ").strip().lower()

            if response in ['y', '1', 'yes']:
                metadata.loc[row_idx, 'label'] = 1
                stats['labeled_as_alarm'] += 1
                stats['total_labeled'] += 1
                labeled_files.add(filename)
                print("✓ Labeled as: ALARM")
                break
            elif response in ['n', '0', 'no']:
                metadata.loc[row_idx, 'label'] = 0
                stats['labeled_as_no_alarm'] += 1
                stats['total_labeled'] += 1
                labeled_files.add(filename)
                print("✓ Labeled as: NO ALARM")
                break
            elif response == 'r':
                print("\n🔊 Replaying audio...")
                play_audio(audio, sr)
                # Don't break - continue the loop to ask again
            elif response == 's':
                stats['skipped'] += 1
                print("⊘ Skipped")
                break
            elif response == 'q':
                print("\n🛑 Quitting...")
                raise KeyboardInterrupt
            else:
                print("Invalid input. Please enter y/n/r/s/q")
        
        # Save progress every 10 files
        if (idx + 1) % 10 == 0:
            metadata.to_csv(METADATA_FILE, index=False)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    'labeled_files': list(labeled_files),
                    'last_updated': datetime.now().isoformat(),
                    'stats': stats
                }, f, indent=2)
            print(f"\n💾 Progress saved ({stats['total_labeled']} files labeled)")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    # Close plots
    plt.close('all')
    
    # Save final results
    print("\n" + "="*70)
    print("SAVING RESULTS...")
    print("="*70)
    
    metadata.to_csv(METADATA_FILE, index=False)
    print(f"✓ Metadata saved to: {METADATA_FILE}")
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            'labeled_files': list(labeled_files),
            'last_updated': datetime.now().isoformat(),
            'stats': stats
        }, f, indent=2)
    print(f"✓ Progress saved to: {PROGRESS_FILE}")
    
    # Show final statistics
    print("\n" + "="*70)
    print("SESSION STATISTICS:")
    print("="*70)
    print(f"Total labeled this session: {stats['labeled_as_alarm'] + stats['labeled_as_no_alarm']}")
    print(f"  Labeled as ALARM:         {stats['labeled_as_alarm']}")
    print(f"  Labeled as NO ALARM:      {stats['labeled_as_no_alarm']}")
    print(f"  Skipped:                  {stats['skipped']}")
    print(f"\nTotal labeled overall:      {stats['total_labeled']}")
    print(f"Remaining:                  {len(dataset_files) - stats['total_labeled']}")
    print()

    # Show new label distribution
    final_stats = metadata[metadata['split'] == DATASET_SPLIT]['label'].value_counts()
    print(f"Current label distribution ({DATASET_SPLIT} set):")
    print(f"  ALARM (1):     {final_stats.get(1, 0)}")
    print(f"  NO ALARM (0):  {final_stats.get(0, 0)}")
    print()

