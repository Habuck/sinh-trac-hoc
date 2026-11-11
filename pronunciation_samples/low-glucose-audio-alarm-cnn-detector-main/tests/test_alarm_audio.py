#!/usr/bin/env python3
"""
Test script to diagnose why alarm audio gets 0.00 probability.
"""
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
from pathlib import Path

# Model definition (must match training)
class GlucoseAlarmCNN(nn.Module):
    def __init__(self, n_mels=64):
        super(GlucoseAlarmCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load model
MODEL_PATH = "../models/glucose_alarm_cnn_w1.5s_20260315_184214.pth"
model = GlucoseAlarmCNN(n_mels=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

print("="*70)
print("ALARM AUDIO DIAGNOSTIC TEST")
print("="*70)
print()

# Record 1.5 seconds of audio
print("🎤 Recording 1.5 seconds of audio...")
print("   PLAY YOUR ALARM NOW!")
print()

audio = sd.rec(int(1.5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

# Analyze audio
rms = np.sqrt(np.mean(audio**2))
peak = np.max(np.abs(audio))

print("Audio Stats:")
print(f"  RMS:  {rms:.6f}")
print(f"  Peak: {peak:.6f}")
print()

# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256
)

# Hybrid normalization (UPDATED - lower threshold)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Apply RMS-based penalty (only for extreme silence)
if rms < 0.0001:
    mel_spec_db = mel_spec_db - 60
    penalty = "-60 dB (extreme silence)"
else:
    penalty = "None (has audio content)"

print("Mel Spectrogram (after hybrid normalization):")
print(f"  Range: [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}] dB")
print(f"  Mean:  {mel_spec_db.mean():.1f} dB")
print(f"  RMS Penalty: {penalty}")
print()

# Convert to tensor
mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)

# Predict
with torch.no_grad():
    logit = model(mel_tensor)
    probability = torch.sigmoid(logit).item()

print("="*70)
print("PREDICTION RESULT:")
print("="*70)
print(f"  Probability: {probability:.4f}")
print(f"  Classification: {'🚨 ALARM' if probability > 0.5 else '❌ NO ALARM'}")
print("="*70)
print()

# Compare with training data
print("Comparing with training data...")
train_file = "datasets/dataset_w1.5s_h0.25s_20260314/train/20260111_204751_window_0000.wav"
if Path(train_file).exists():
    train_audio, sr = librosa.load(train_file, sr=16000, mono=True)
    train_rms = np.sqrt(np.mean(train_audio**2))
    train_peak = np.max(np.abs(train_audio))
    
    train_mel = librosa.feature.melspectrogram(
        y=train_audio, sr=16000, n_mels=64, n_fft=1024, hop_length=256
    )
    train_mel_db = librosa.power_to_db(train_mel, ref=np.max)

    if train_rms < 0.0001:
        train_mel_db = train_mel_db - 60
    
    print()
    print("Training Sample:")
    print(f"  RMS:  {train_rms:.6f}")
    print(f"  Peak: {train_peak:.6f}")
    print(f"  Mel Range: [{train_mel_db.min():.1f}, {train_mel_db.max():.1f}] dB")
    print(f"  Mel Mean:  {train_mel_db.mean():.1f} dB")
    print()
    print("Your Capture:")
    print(f"  RMS:  {rms:.6f}")
    print(f"  Peak: {peak:.6f}")
    print(f"  Mel Range: [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}] dB")
    print(f"  Mel Mean:  {mel_spec_db.mean():.1f} dB")
    print()
    print("Difference:")
    print(f"  RMS Ratio: {train_rms/rms:.2f}x")
    print(f"  Peak Ratio: {train_peak/peak:.2f}x")

