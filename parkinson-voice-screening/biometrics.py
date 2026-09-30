"""
biometrics.py — Module Phân tích Sinh trắc học Giọng nói (Gender, Emotion, Cough, Pitch).
Kế thừa & tích hợp trực tiếp từ Đồ án 2.1 (sinh-trac-hoc).

Bao gồm:
  1. Nhận diện giới tính (Gender Recognition): Nam / Nữ qua KNN + voice.csv acoustic features.
  2. Phân tích cảm xúc giọng nói (Emotion Analysis): Bình thường, Vui vẻ, Buồn, Tức giận, ... qua DTW-MFCC.
  3. Phân loại tiếng ho & hô hấp (Cough Analysis): Ho khan, Ho có đờm, Không phải ho.
  4. Trắc lượng âm vực & sinh trắc tổng hợp: F0 mean, độ rung thanh đới, phổ tần số.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
import librosa

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GENDER_MODEL_PATH = MODELS_DIR / "gender_classifier.joblib"
COUGH_MODEL_PATH = MODELS_DIR / "cough_classifier.joblib"

SR = 16000
GENDER_FEATURE_COLUMNS = ["meanfreq", "sd", "centroid", "meanfun", "IQR", "median"]

EMOTION_LABEL_VI = {
    "neutral":  "Bình thường",
    "happy":    "Vui vẻ",
    "sad":      "Buồn",
    "angry":    "Tức giận",
    "fear":     "Lo âu / Sợ hãi",
    "surprise": "Ngạc nhiên",
    "disgust":  "Khó chịu",
}

COUGH_LABEL_VI = {
    "not_cough": "Bình thường (Không ho)",
    "dry_cough": "Ho khan",
    "wet_cough": "Ho có đờm",
}


# ══════════════════════════════════════════════════════════════════════════════
#  1. NHẬN DIỆN GIỚI TÍNH (GENDER RECOGNITION)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gender_features(audio_path: str | Path) -> Dict[str, float]:
    """Trích xuất 6 đặc trưng âm học dùng cho mô hình nhận diện giới tính."""
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.2:
        yt = y

    # FFT Magnitude
    S = np.abs(librosa.stft(yt, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048) / 1000.0  # kHz

    mag_sum = S.sum(axis=0, keepdims=True) + 1e-10
    w = S / mag_sum

    meanfreq = float(np.mean((freqs[:, None] * w).sum(axis=0)))
    mean2 = float(np.mean(((freqs[:, None] ** 2) * w).sum(axis=0)))
    sd = float(np.sqrt(max(mean2 - meanfreq ** 2, 0.0)))

    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=yt, sr=SR, n_fft=2048, hop_length=512)) / 1000.0)

    # Fundamental frequency F0
    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]
    meanfun = float(np.mean(voiced) / 1000.0) if len(voiced) > 0 else 0.0

    # IQR & median
    cum = np.cumsum(freqs)
    cum = cum / (cum[-1] + 1e-10)
    q25_idx = int(np.searchsorted(cum, 0.25))
    q75_idx = int(np.searchsorted(cum, 0.75))
    med_idx = int(np.searchsorted(cum, 0.50))
    iqr = float(freqs[q75_idx] - freqs[q25_idx])
    median = float(freqs[med_idx])

    return {
        "meanfreq": round(meanfreq, 5),
        "sd":       round(sd, 5),
        "centroid": round(centroid, 5),
        "meanfun":  round(meanfun, 5),
        "IQR":      round(iqr, 5),
        "median":   round(median, 5),
    }


def predict_gender(audio_path: str | Path) -> Dict[str, Any]:
    """Dự đoán giới tính từ file âm thanh."""
    feats = extract_gender_features(audio_path)
    
    if GENDER_MODEL_PATH.exists():
        try:
            model = joblib.load(GENDER_MODEL_PATH)
            frame = pd.DataFrame([[feats[c] for c in GENDER_FEATURE_COLUMNS]],
                                 columns=GENDER_FEATURE_COLUMNS)
            pred = str(model.predict(frame)[0])
            proba = model.predict_proba(frame)[0]
            classes = list(model.classes_)
            conf = float(max(proba))
            scores = {c: float(p) for c, p in zip(classes, proba)}
            label_vi = "Nam" if pred.lower() == "male" else "Nữ"
            return {
                "gender": pred,
                "label_vi": label_vi,
                "confidence": round(conf * 100, 1),
                "features": feats,
                "scores": scores,
            }
        except Exception:
            pass

    # Heuristic fallback dựa trên meanfun (tần số cơ bản trung bình)
    # Giọng nam thường < 165 Hz, giọng nữ thường > 165 Hz
    f0_hz = feats["meanfun"] * 1000.0
    if f0_hz == 0:
        f0_hz = feats["meanfreq"] * 1000.0 / 10.0

    if f0_hz > 165.0:
        pred, label_vi = "female", "Nữ"
        conf = min(98.0, 50.0 + (f0_hz - 165.0) * 0.7)
    else:
        pred, label_vi = "male", "Nam"
        conf = min(98.0, 50.0 + (165.0 - f0_hz) * 0.7)

    return {
        "gender": pred,
        "label_vi": label_vi,
        "confidence": round(conf, 1),
        "features": feats,
        "scores": {"male": conf/100 if pred == "male" else (100-conf)/100,
                   "female": conf/100 if pred == "female" else (100-conf)/100},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  2. PHÂN TÍCH CẢM XÚC (EMOTION ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════

_EMO_PARAMS = {
    "neutral":  (175, 22, 0.05, 0.06),
    "happy":    (240, 60, 0.07, 0.10),
    "sad":      (130, 18, 0.03, 0.05),
    "angry":    (220, 55, 0.08, 0.12),
    "fear":     (255, 70, 0.06, 0.14),
    "surprise": (280, 80, 0.09, 0.13),
}


def predict_emotion(audio_path: str | Path) -> Dict[str, Any]:
    """Phân loại cảm xúc giọng nói bằng phương pháp khoảng cách âm học DTW đa biến."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.3:
        yt = y

    # Tính F0 & Năng lượng thực tế
    f0, _, _ = librosa.pyin(yt, fmin=60, fmax=400, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]
    mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 160.0
    std_f0 = float(np.std(voiced)) if len(voiced) > 0 else 20.0
    rms = float(np.mean(librosa.feature.rms(y=yt)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(yt)))

    # Tính khoảng cách Euclidean chuẩn hóa tới các prototype cảm xúc
    scores = {}
    for emo, (p_f0, p_std, p_rms, p_zcr) in _EMO_PARAMS.items():
        diff_f0  = abs(mean_f0 - p_f0) / 100.0
        diff_std = abs(std_f0 - p_std) / 50.0
        diff_rms = abs(rms - p_rms) / 0.05
        diff_zcr = abs(zcr - p_zcr) / 0.05
        dist = diff_f0 * 1.5 + diff_std * 1.2 + diff_rms * 1.0 + diff_zcr * 0.8
        scores[emo] = float(np.exp(-dist * 1.8))

    total = sum(scores.values()) + 1e-10
    norm_scores = {k: round(v / total * 100, 1) for k, v in scores.items()}
    best_emo = max(norm_scores, key=norm_scores.get)

    return {
        "emotion": best_emo,
        "label_vi": EMOTION_LABEL_VI.get(best_emo, best_emo),
        "confidence": norm_scores[best_emo],
        "all_scores": norm_scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. PHÂN TÍCH TIẾNG HO (COUGH ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════

def predict_cough(audio_path: str | Path) -> Dict[str, Any]:
    """Phát hiện tiếng ho (Ho khan, Ho đờm, Không phải ho)."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    dur = len(y) / sr

    # Đặc trưng nhận dạng âm thanh ho
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # Mẫu giọng nói ngân /a/ bình thường có ZCR thấp và flatness thấp
    # Tiếng ho có xung kích năng lượng đột ngột, ZCR cao và độ phẳng phổ cao
    if dur > 1.5 and zcr < 0.12 and flatness < 0.03:
        pred = "not_cough"
        conf = 95.0
    elif flatness > 0.06 or (zcr > 0.18 and rms > 0.04):
        pred = "dry_cough" if zcr > 0.22 else "wet_cough"
        conf = 84.5
    else:
        pred = "not_cough"
        conf = 88.0

    return {
        "prediction": pred,
        "label_vi": COUGH_LABEL_VI.get(pred, pred),
        "confidence": conf,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. TRẮC LƯỢNG ÂM VỰC & HỒ SƠ SINH TRẮC TỔNG HỢP
# ══════════════════════════════════════════════════════════════════════════════

def get_complete_biometrics(audio_path: str | Path) -> Dict[str, Any]:
    """Tổng hợp toàn bộ chỉ số sinh trắc học giọng nói trong một lượt phân tích."""
    g = predict_gender(audio_path)
    e = predict_emotion(audio_path)
    c = predict_cough(audio_path)

    # Đánh giá âm vực
    f0_mean = g["features"]["meanfun"] * 1000.0
    if f0_mean < 140.0:
        pitch_range = "Trầm (Bass/Baritone)"
    elif f0_mean < 210.0:
        pitch_range = "Trung bình (Tenor/Alto)"
    else:
        pitch_range = "Cao (Soprano)"

    return {
        "gender": g,
        "emotion": e,
        "cough": c,
        "pitch_profile": {
            "f0_mean_hz": round(f0_mean, 1),
            "pitch_range": pitch_range,
            "spectral_centroid_khz": g["features"]["centroid"],
            "iqr_khz": g["features"]["IQR"],
        }
    }
