#!/usr/bin/env python3
"""
Quick microphone test script to diagnose audio recording issues.
Run this to check if your microphone is working properly.
"""

import sounddevice as sd
import numpy as np

print("=" * 60)
print("MICROPHONE DIAGNOSTIC TEST")
print("=" * 60)

# List all audio devices
print("\n1. Available Audio Devices:")
print("-" * 60)
devices = sd.query_devices()
print(devices)

# Show default input device
print("\n2. Default Input Device:")
print("-" * 60)
default_input = sd.query_devices(kind='input')
print(default_input)

# Test recording
print("\n3. Recording Test (3 seconds):")
print("-" * 60)
print("🎤 Speak now or make some noise...")

duration = 3  # seconds
sample_rate = 16000

try:
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    # Convert to int16
    recording_int16 = np.int16(recording * 32767)
    
    # Analyze the recording
    max_amplitude = np.max(np.abs(recording_int16))
    rms = np.sqrt(np.mean(recording_int16.astype(float)**2))
    
    print("\n4. Recording Analysis:")
    print("-" * 60)
    print(f"Duration: {duration} seconds")
    print(f"Samples recorded: {len(recording_int16)}")
    print(f"Max amplitude: {max_amplitude} (out of 32767)")
    print(f"RMS level: {rms:.2f}")
    print(f"Peak percentage: {(max_amplitude / 32767) * 100:.2f}%")
    
    # Verdict
    print("\n5. Verdict:")
    print("-" * 60)
    if max_amplitude < 100:
        print("❌ FAILED: No audio detected!")
        print("\nPossible issues:")
        print("  • Microphone permission not granted")
        print("  • Wrong input device selected")
        print("  • Microphone is muted")
        print("  • Hardware issue")
        print("\nTo fix:")
        print("  1. Check System Preferences → Security & Privacy → Privacy → Microphone")
        print("  2. Ensure Terminal/Python has microphone access")
        print("  3. Check System Preferences → Sound → Input")
        print("  4. Verify input volume is turned up")
    elif max_amplitude < 1000:
        print("⚠️  WARNING: Audio detected but very quiet")
        print("\nSuggestions:")
        print("  • Increase microphone input volume")
        print("  • Speak louder or move closer to microphone")
    else:
        print("✅ SUCCESS: Microphone is working properly!")
        print(f"\nYour microphone captured audio with peak level of {(max_amplitude / 32767) * 100:.1f}%")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nThis might indicate a permission or device issue.")

print("\n" + "=" * 60)

