#!/usr/bin/env python3
"""
Debug script to see exactly what the model is seeing when you play the alarm.
"""
import numpy as np
import librosa
import torch
import sounddevice as sd
from pathlib import Path

print("="*70)
print("LIVE CAPTURE DEBUG")
print("="*70)

# Record 1.5 seconds of audio
print("\n🎤 Recording 1.5 seconds... PLAY THE ALARM NOW!")
duration = 1.5
sample_rate = 16000
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

print("✓ Recording complete!")

# Compute stats
rms = np.sqrt(np.mean(audio**2))
peak = np.max(np.abs(audio))

print(f"\n📊 Audio Stats:")
print(f"  RMS:  {rms:.6f}")
print(f"  Peak: {peak:.6f}")

# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_mels=64,
    n_fft=1024,
    hop_length=256
)

# Method 1: ref=np.max (pattern-focused)
mel_db_max = librosa.power_to_db(mel_spec, ref=np.max)

# Method 2: ref=1.0 (volume-aware)
mel_db_1 = librosa.power_to_db(mel_spec, ref=1.0)

# Method 3: Hybrid (what we're using now)
mel_db_hybrid = librosa.power_to_db(mel_spec, ref=np.max)
if rms < 0.001:
    mel_db_hybrid = mel_db_hybrid - 40
    penalty = "-40 dB"
elif rms < 0.01:
    mel_db_hybrid = mel_db_hybrid - 20
    penalty = "-20 dB"
else:
    penalty = "None"

print(f"\n📊 Mel Spectrogram Stats:")
print(f"\n  ref=np.max:")
print(f"    Range: [{mel_db_max.min():.1f}, {mel_db_max.max():.1f}] dB")
print(f"    Mean:  {mel_db_max.mean():.1f} dB")

print(f"\n  ref=1.0:")
print(f"    Range: [{mel_db_1.min():.1f}, {mel_db_1.max():.1f}] dB")
print(f"    Mean:  {mel_db_1.mean():.1f} dB")

print(f"\n  Hybrid (ref=np.max + RMS penalty):")
print(f"    RMS penalty: {penalty}")
print(f"    Range: [{mel_db_hybrid.min():.1f}, {mel_db_hybrid.max():.1f}] dB")
print(f"    Mean:  {mel_db_hybrid.mean():.1f} dB")

# Load a training sample for comparison
train_file = "datasets/dataset_w1.5s_h0.25s_20260314/train/20260111_204751_window_0000.wav"
if Path(train_file).exists():
    train_audio, _ = librosa.load(train_file, sr=16000, mono=True)
    train_rms = np.sqrt(np.mean(train_audio**2))
    train_mel = librosa.feature.melspectrogram(y=train_audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256)
    train_mel_db = librosa.power_to_db(train_mel, ref=np.max)
    
    if train_rms < 0.001:
        train_mel_db = train_mel_db - 40
    elif train_rms < 0.01:
        train_mel_db = train_mel_db - 20
    
    print(f"\n📊 Training Sample (for comparison):")
    print(f"  RMS:  {train_rms:.6f}")
    print(f"  Hybrid Range: [{train_mel_db.min():.1f}, {train_mel_db.max():.1f}] dB")
    print(f"  Hybrid Mean:  {train_mel_db.mean():.1f} dB")

# Try to load and test with the model
model_path = "models/glucose_alarm_cnn_w1.5s_20260315_175325.pth"
if Path(model_path).exists():
    print(f"\n🤖 Testing with model: {model_path}")
    
    # Load model architecture (you'll need to copy this from your notebook)
    import torch.nn as nn
    
    class GlucoseAlarmCNN(nn.Module):
        def __init__(self, input_height=64):
            super(GlucoseAlarmCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * (input_height // 8) * 5, 128)
            self.fc2 = nn.Linear(128, 1)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = GlucoseAlarmCNN(input_height=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Test with different normalizations
    for name, mel_db in [("ref=np.max", mel_db_max), ("ref=1.0", mel_db_1), ("Hybrid", mel_db_hybrid)]:
        mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(mel_tensor)
            prob = torch.sigmoid(output).item()
        print(f"\n  {name}: Probability = {prob:.4f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if rms < 0.001:
    print("⚠️  Your audio is VERY QUIET (RMS < 0.001)")
    print("   This is being treated as silence/background noise.")
    print("   Solutions:")
    print("   1. Turn up speaker volume")
    print("   2. Move microphone closer to speaker")
    print("   3. Check System Preferences → Sound → Input volume")
elif rms < 0.01:
    print("⚠️  Your audio is QUIET (RMS < 0.01)")
    print("   This is getting a -20 dB penalty.")
    print("   Try increasing volume slightly.")
else:
    print("✓ Audio volume looks good (RMS >= 0.01)")
    print("  If still getting 0.00 probability, the issue is elsewhere.")

print("\n💡 Next steps:")
print("  1. Check if the model was actually retrained with hybrid normalization")
print("  2. Verify the training data has similar RMS values")
print("  3. Consider if the alarm frequency pattern is being captured")

