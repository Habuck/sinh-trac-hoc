#!/usr/bin/env python3
"""
Quick setup verification script for the audio session recorder.
Run this to ensure all dependencies are installed and microphone is working.
"""

import sys

print("=" * 70)
print("AUDIO SESSION RECORDER - SETUP CHECK")
print("=" * 70)

# Check Python version
print("\n1. Python Version:")
print("-" * 70)
print(f"   {sys.version}")
if sys.version_info < (3, 7):
    print("   ⚠️  WARNING: Python 3.7+ recommended")
else:
    print("   ✓ Python version OK")

# Check dependencies
print("\n2. Checking Dependencies:")
print("-" * 70)

dependencies = {
    'sounddevice': 'sounddevice',
    'numpy': 'numpy',
    'scipy': 'scipy.io',
}

all_installed = True
for name, import_path in dependencies.items():
    try:
        __import__(import_path)
        print(f"   ✓ {name} installed")
    except ImportError:
        print(f"   ❌ {name} NOT installed")
        all_installed = False

if not all_installed:
    print("\n   To install missing packages:")
    print("   pip install sounddevice scipy numpy")
else:
    print("\n   ✓ All dependencies installed")

# Check microphone access
print("\n3. Microphone Test:")
print("-" * 70)

try:
    import sounddevice as sd
    import numpy as np
    
    # List devices
    print("   Available audio devices:")
    devices = sd.query_devices()
    print(f"\n{devices}\n")
    
    # Show default input
    default_input = sd.query_devices(kind='input')
    print(f"   Default input device: {default_input['name']}")
    
    # Quick recording test
    print("\n   Testing microphone (2 seconds)...")
    print("   🎤 Make some noise...")
    
    test_duration = 2
    test_recording = sd.rec(
        int(test_duration * 16000),
        samplerate=16000,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    
    max_amplitude = np.max(np.abs(test_recording))
    
    print(f"\n   Max amplitude: {max_amplitude}")
    
    if max_amplitude < 100:
        print("   ❌ FAILED: No audio detected")
        print("\n   Possible issues:")
        print("   • Microphone permission not granted")
        print("   • Wrong input device selected")
        print("   • Microphone is muted")
        print("\n   To fix:")
        print("   1. System Preferences → Security & Privacy → Privacy → Microphone")
        print("   2. Enable microphone access for Terminal/Python")
        print("   3. Restart this script")
    else:
        print("   ✓ Microphone is working!")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("\n   This might indicate a permission or device issue.")

# Check output directory
print("\n4. Output Directory:")
print("-" * 70)

import os
output_dir = 'sessions'
if os.path.exists(output_dir):
    print(f"   ✓ '{output_dir}/' directory exists")
    files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    print(f"   Found {len(files)} existing WAV file(s)")
else:
    print(f"   ℹ️  '{output_dir}/' directory will be created on first recording")

# Final summary
print("\n" + "=" * 70)
print("SETUP CHECK COMPLETE")
print("=" * 70)

if all_installed and max_amplitude >= 100:
    print("\n✅ Everything looks good! You're ready to record sessions.")
    print("\nNext steps:")
    print("   1. Open session_recorder.ipynb in Jupyter")
    print("   2. Set SESSION_TYPE and DURATION_SECONDS")
    print("   3. Run all cells to start recording")
else:
    print("\n⚠️  Please fix the issues above before recording.")
    print("\nSee SESSION_RECORDER_GUIDE.md for detailed troubleshooting.")

print("\n" + "=" * 70)

