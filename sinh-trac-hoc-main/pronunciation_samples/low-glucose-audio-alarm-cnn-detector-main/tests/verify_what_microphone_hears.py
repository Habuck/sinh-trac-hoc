#!/usr/bin/env python3
"""
Verify what the microphone is actually capturing.
Records, plays back, and saves the audio so you can verify it.
"""
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from datetime import datetime

print("="*70)
print("MICROPHONE VERIFICATION TEST")
print("="*70)
print()
print("This will:")
print("  1. Record 3 seconds of audio")
print("  2. Save it to a file")
print("  3. Play it back to you")
print("  4. Show you what it captured")
print()
input("Press ENTER when ready to record...")
print()

# Record 3 seconds
print("🎤 RECORDING 3 SECONDS...")
print("   PLAY YOUR ALARM NOW!")
print()

audio = sd.rec(int(3.0 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

print("✓ Recording complete!")
print()

# Save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"captured_audio_{timestamp}.wav"
sf.write(filename, audio, 16000)
print(f"✓ Saved to: {filename}")
print()

# Analyze
rms = np.sqrt(np.mean(audio**2))
peak = np.max(np.abs(audio))

print("Audio Stats:")
print(f"  RMS:  {rms:.6f}")
print(f"  Peak: {peak:.6f}")
print()

if peak < 0.001:
    print("⚠️  VERY QUIET! Peak is < 0.001")
    print("    This might be silence or very faint audio.")
elif peak < 0.01:
    print("⚠️  QUIET. Peak is < 0.01")
    print("    This is quieter than typical alarm sounds.")
elif peak < 0.1:
    print("✓ Normal volume. Peak is in reasonable range.")
else:
    print("✓ LOUD! Peak is > 0.1")

print()
print("="*70)
print("PLAYING BACK WHAT WAS CAPTURED...")
print("="*70)
print()
print("Listen carefully - is this your alarm sound?")
print()
input("Press ENTER to play back...")

sd.play(audio, 16000)
sd.wait()

print()
print("="*70)
print("DID YOU HEAR YOUR ALARM?")
print("="*70)
print()
print("If YES:")
print("  → The microphone is working and capturing the alarm")
print("  → The problem is with the model/normalization")
print()
print("If NO (heard silence, noise, or something else):")
print("  → The microphone is NOT capturing the alarm properly")
print("  → Possible issues:")
print("     - Wrong microphone selected")
print("     - Alarm volume too low")
print("     - Alarm playing from wrong device")
print("     - Microphone permissions issue")
print()
print(f"The captured audio is saved as: {filename}")
print("You can open it in any audio player to verify.")
print()

