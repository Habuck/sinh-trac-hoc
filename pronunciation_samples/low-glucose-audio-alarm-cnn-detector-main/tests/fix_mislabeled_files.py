#!/usr/bin/env python3
"""
Fix mislabeled files based on the frequency analysis results.
"""
import pandas as pd
import shutil
from pathlib import Path

print("="*70)
print("FIX MISLABELED FILES")
print("="*70)
print()

# Load the analysis results
results_file = Path("training_frequency_analysis_results.csv")
if not results_file.exists():
    print(f"❌ Error: {results_file} not found!")
    print("Please run analyze_all_training_files.py first.")
    exit(1)

results_df = pd.read_csv(results_file)
mislabeled_df = results_df[results_df['is_mislabeled'] == True]

print(f"Analysis results loaded from: {results_file}")
print()
print(f"Total files analyzed: {len(results_df)}")
print(f"Mislabeled files found: {len(mislabeled_df)}")
print(f"Percentage mislabeled: {len(mislabeled_df) / len(results_df) * 100:.1f}%")
print()

# Load metadata
DATASET_DIR = Path("../datasets/dataset_w1.5s_h0.25s_20260314")
METADATA_FILE = DATASET_DIR / "dataset_metadata.csv"

metadata = pd.read_csv(METADATA_FILE)

print("Current label distribution:")
print(f"  Alarm (label=1):     {len(metadata[metadata['label'] == 1])}")
print(f"  Non-alarm (label=0): {len(metadata[metadata['label'] == 0])}")
print()

# Show sample of files to be relabeled
print("Sample of files to be relabeled (first 20):")
print("-"*70)
for _, row in mislabeled_df.head(20).iterrows():
    print(f"  {row['filename']:50s} Peak: {row['peak_hz']:5.0f} Hz")
if len(mislabeled_df) > 20:
    print(f"  ... and {len(mislabeled_df) - 20} more")
print()

# Ask for confirmation
print("="*70)
print("CONFIRMATION:")
print("="*70)
print(f"This will change {len(mislabeled_df)} files from label=1 (alarm) to label=0 (non-alarm)")
print()
print("These files have peak frequency < 2000 Hz (mostly 0-96 Hz silence/noise)")
print("and should NOT be labeled as alarms.")
print()
print("1. Yes, fix the labels")
print("2. No, cancel")
print()

choice = input("Enter your choice (1/2): ").strip()

if choice == "1":
    print()
    print("Fixing labels...")
    
    # Create a copy of metadata
    metadata_cleaned = metadata.copy()
    
    # Get indices of mislabeled files
    mislabeled_indices = mislabeled_df['index'].tolist()
    
    # Change labels
    metadata_cleaned.loc[mislabeled_indices, 'label'] = 0
    
    # Backup original metadata (if not already backed up)
    backup_metadata = METADATA_FILE.with_suffix('.csv.backup_original')
    if not backup_metadata.exists():
        shutil.copy(METADATA_FILE, backup_metadata)
        print(f"✓ Original metadata backed up to: {backup_metadata}")
    else:
        print(f"✓ Backup already exists: {backup_metadata}")
    
    # Save cleaned metadata
    metadata_cleaned.to_csv(METADATA_FILE, index=False)
    print(f"✓ Cleaned metadata saved to: {METADATA_FILE}")
    print()
    
    # Show new statistics
    print("="*70)
    print("CLEANING SUMMARY:")
    print("="*70)
    print(f"Files relabeled: {len(mislabeled_df)}")
    print()
    print("New label distribution:")
    print(f"  Alarm (label=1):     {len(metadata_cleaned[metadata_cleaned['label'] == 1])}")
    print(f"  Non-alarm (label=0): {len(metadata_cleaned[metadata_cleaned['label'] == 0])}")
    print()
    
    # Show what the training set will look like
    train_data = metadata_cleaned[metadata_cleaned['split'] == 'train']
    print("Training set:")
    print(f"  Alarm (label=1):     {len(train_data[train_data['label'] == 1])}")
    print(f"  Non-alarm (label=0): {len(train_data[train_data['label'] == 0])}")
    print()
    
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Open train_cnn_window_split.ipynb")
    print("2. Set LOAD_MODEL_PATH = None (to train from scratch)")
    print("3. Run all cells to retrain with cleaned data")
    print()
    print("Expected improvement:")
    print("  - Model will learn ONLY high-frequency alarms (~3000 Hz)")
    print("  - Model will NOT be confused by silence labeled as alarms")
    print("  - Live alarm detection should work correctly")
    print("  - Much better performance overall!")
    print()

else:
    print()
    print("No changes made.")
    print()

