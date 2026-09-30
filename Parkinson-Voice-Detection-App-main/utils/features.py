import numpy as np
import librosa

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    feats = {}
    feats["duration"] = len(y) / sr
    feats["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))
    feats["rms_mean"] = np.mean(librosa.feature.rms(y=y))
    feats["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    feats["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    feats["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}"] = np.mean(mfcc[i])

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        feats["pitch_mean"] = np.mean(f0[np.isfinite(f0)])
        feats["pitch_std"] = np.std(f0[np.isfinite(f0)])
    except Exception:
        feats["pitch_mean"] = 0.0
        feats["pitch_std"] = 0.0

    return feats
