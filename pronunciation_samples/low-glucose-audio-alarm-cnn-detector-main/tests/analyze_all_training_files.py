#!/usr/bin/env python3
"""
Analyze ALL training files to identify which ones are actually mislabeled.
This does a proper frequency analysis on every file.
"""
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

print("="*70)
print("COMPLETE TRAINING DATA FREQUENCY ANALYSIS")
print("="*70)
print()

# Configuration
DATASET_DIR = Path("../datasets/dataset_w1.5s_h0.25s_20260314")
METADATA_FILE = DATASET_DIR / "dataset_metadata.csv"
TRAIN_DIR = DATASET_DIR / "train"
ALARM_FREQ_THRESHOLD = 2000  # Hz - alarms should be above this

print(f"Dataset directory: {DATASET_DIR}")
print(f"Alarm frequency threshold: {ALARM_FREQ_THRESHOLD} Hz")
print()

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv(METADATA_FILE)
train_alarms = metadata[(metadata['split'] == 'train') & (metadata['label'] == 1)]

print(f"Total training files labeled as 'alarm': {len(train_alarms)}")
print()
print("Analyzing frequency content of all alarm files...")
print("(This will take a few minutes - analyzing ~700 files)")
print()

results = []
mislabeled_count = 0
correct_count = 0

for i, (idx, row) in enumerate(train_alarms.iterrows()):
    filename = row['filename']
    filepath = TRAIN_DIR / filename
    
    # Show progress
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{len(train_alarms)} files analyzed...")
    
    if not filepath.exists():
        continue
    
    # Load and analyze
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    rms = np.sqrt(np.mean(audio**2))
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Get peak frequency
    freq_profile = mel_db.mean(axis=1)
    peak_band = np.argmax(freq_profile)
    peak_hz = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=8000)[peak_band]
    
    # Determine if mislabeled
    is_mislabeled = peak_hz < ALARM_FREQ_THRESHOLD
    
    if is_mislabeled:
        mislabeled_count += 1
    else:
        correct_count += 1
    
    results.append({
        'index': idx,
        'filename': filename,
        'session_id': row['session_id'],
        'window_index': row['window_index'],
        'peak_hz': peak_hz,
        'rms': rms,
        'is_mislabeled': is_mislabeled
    })

print(f"Progress: {len(train_alarms)}/{len(train_alarms)} files analyzed... DONE!")
print()

# Create DataFrame for analysis
results_df = pd.DataFrame(results)

print("="*70)
print("ANALYSIS RESULTS:")
print("="*70)
print(f"Correctly labeled (peak > {ALARM_FREQ_THRESHOLD} Hz): {correct_count}")
print(f"Mislabeled (peak < {ALARM_FREQ_THRESHOLD} Hz):        {mislabeled_count}")
print(f"Percentage mislabeled:                                {mislabeled_count / len(train_alarms) * 100:.1f}%")
print()

# Show mislabeled files
mislabeled_df = results_df[results_df['is_mislabeled']]
print(f"Mislabeled files (first 30):")
print("-"*70)
for _, row in mislabeled_df.head(30).iterrows():
    print(f"  {row['filename']:50s} Peak: {row['peak_hz']:5.0f} Hz, RMS: {row['rms']:.4f}")
if len(mislabeled_df) > 30:
    print(f"  ... and {len(mislabeled_df) - 30} more")
print()

# Save results to CSV
results_csv = Path("training_frequency_analysis_results.csv")
results_df.to_csv(results_csv, index=False)
print(f"✓ Full results saved to: {results_csv}")
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of peak frequencies
axes[0, 0].hist(results_df['peak_hz'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(x=ALARM_FREQ_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {ALARM_FREQ_THRESHOLD} Hz')
axes[0, 0].set_xlabel('Peak Frequency (Hz)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of Peak Frequencies in Training Data')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Peak frequency by window index
axes[0, 1].scatter(results_df['window_index'], results_df['peak_hz'], alpha=0.5, s=10)
axes[0, 1].axhline(y=ALARM_FREQ_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold: {ALARM_FREQ_THRESHOLD} Hz')
axes[0, 1].set_xlabel('Window Index')
axes[0, 1].set_ylabel('Peak Frequency (Hz)')
axes[0, 1].set_title('Peak Frequency vs Window Index')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# RMS distribution
axes[1, 0].hist(results_df['rms'], bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('RMS Energy')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Distribution of RMS Energy')
axes[1, 0].grid(True, alpha=0.3)

# Peak frequency vs RMS
colors = ['red' if m else 'green' for m in results_df['is_mislabeled']]
axes[1, 1].scatter(results_df['rms'], results_df['peak_hz'], alpha=0.5, s=10, c=colors)
axes[1, 1].axhline(y=ALARM_FREQ_THRESHOLD, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('RMS Energy')
axes[1, 1].set_ylabel('Peak Frequency (Hz)')
axes[1, 1].set_title('Peak Frequency vs RMS (Red=Mislabeled, Green=Correct)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complete_training_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: complete_training_analysis.png")
print()

print("="*70)
print("NEXT STEP:")
print("="*70)
print("Review the results and visualization, then run:")
print("  python3 fix_mislabeled_files.py")
print()
print("This will relabel the mislabeled files based on the frequency analysis.")
print()

