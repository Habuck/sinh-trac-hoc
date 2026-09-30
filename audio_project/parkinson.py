"""Sàng lọc Parkinson qua giọng nói — Voice-based Parkinson's Disease Screening.

Trích xuất đặc trưng:
  - Jitter variants (local, RAP, PPQ5, DDP)
  - Shimmer variants (local, dB, APQ3, APQ5)
  - HNR / NHR
  - Fundamental frequency (F0) statistics
  - MFCC (13 coefficients, mean & std)
  - PPE (Pitch Period Entropy)
  - Speech rate & pause ratio

Tham khảo:
  - Tsanas et al. (2012) — Novel speech signal processing algorithms
  - Little et al. (2009) — Dysphonia measurements for PD telemonitoring
  - UCI Parkinson's Dataset (195 samples, 23 features)

Phần cứng mục tiêu: Microphone INMP441 (I2S) + Raspberry Pi 4
"""
from pathlib import Path
from typing import Dict
import librosa
import numpy as np

SR = 16000
HOP_LENGTH = 512


# ══════════════════════════════════════════════════════════════════════
#  JITTER — Biến thiên chu kỳ dao động dây thanh
# ══════════════════════════════════════════════════════════════════════

def _jitter_local(f0: np.ndarray) -> float:
    """Jitter (local) — relative period-to-period perturbation."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _jitter_rap(f0: np.ndarray) -> float:
    """Jitter (RAP) — Relative Average Perturbation (3-point smoothing)."""
    voiced = f0[f0 > 0]
    if len(voiced) < 5:
        return 0.0
    periods = 1.0 / voiced
    smoothed = np.convolve(periods, [1 / 3, 1 / 3, 1 / 3], mode="valid")
    trimmed = periods[1:-1]
    diffs = np.abs(trimmed - smoothed)
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _jitter_ppq5(f0: np.ndarray) -> float:
    """Jitter (PPQ5) — five-point Period Perturbation Quotient."""
    voiced = f0[f0 > 0]
    if len(voiced) < 7:
        return 0.0
    periods = 1.0 / voiced
    smoothed = np.convolve(periods, np.ones(5) / 5, mode="valid")
    trimmed = periods[2:-2]
    diffs = np.abs(trimmed - smoothed)
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _jitter_ddp(f0: np.ndarray) -> float:
    """Jitter (DDP) — average |diff of diff| of periods."""
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return 0.0
    periods = 1.0 / voiced
    ddp = np.abs(np.diff(np.diff(periods)))
    return float(np.mean(ddp) / (np.mean(periods) + 1e-10))


# ══════════════════════════════════════════════════════════════════════
#  SHIMMER — Biến thiên biên độ
# ══════════════════════════════════════════════════════════════════════

def _shimmer_local(y: np.ndarray, sr: int = SR) -> float:
    """Shimmer (local) — relative amplitude perturbation."""
    fl = int(sr * 0.03)
    hl = int(sr * 0.01)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    amps = np.max(np.abs(frames), axis=0)
    if len(amps) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-10))


def _shimmer_db(y: np.ndarray, sr: int = SR) -> float:
    """Shimmer (dB) — amplitude variation in decibels."""
    fl = int(sr * 0.03)
    hl = int(sr * 0.01)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    amps = np.max(np.abs(frames), axis=0)
    amps = amps[amps > 0]
    if len(amps) < 3:
        return 0.0
    db_amps = 20 * np.log10(amps + 1e-10)
    return float(np.mean(np.abs(np.diff(db_amps))))


def _shimmer_apq(y: np.ndarray, sr: int = SR, n: int = 3) -> float:
    """Shimmer APQn — n-point Amplitude Perturbation Quotient."""
    fl = int(sr * 0.03)
    hl = int(sr * 0.01)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    amps = np.max(np.abs(frames), axis=0)
    if len(amps) < n + 2:
        return 0.0
    smoothed = np.convolve(amps, np.ones(n) / n, mode="valid")
    offset = n // 2
    trimmed = amps[offset : offset + len(smoothed)]
    diffs = np.abs(trimmed - smoothed)
    return float(np.mean(diffs) / (np.mean(amps) + 1e-10))


# ══════════════════════════════════════════════════════════════════════
#  HNR / NHR — Harmonics & Noise
# ══════════════════════════════════════════════════════════════════════

def _hnr(y: np.ndarray, sr: int = SR) -> float:
    """Harmonics-to-Noise Ratio (dB)."""
    ac = librosa.autocorrelate(y, max_size=sr // 60)
    if len(ac) < 2 or ac[0] == 0:
        return 0.0
    peak = np.max(ac[1:])
    ratio = np.clip(peak / ac[0], 1e-10, 1 - 1e-10)
    return float(10 * np.log10(ratio / (1 - ratio)))


def _nhr(y: np.ndarray, sr: int = SR) -> float:
    """Noise-to-Harmonics Ratio."""
    hnr_val = _hnr(y, sr)
    if hnr_val == 0:
        return 0.0
    return float(10 ** (-hnr_val / 10))


# ══════════════════════════════════════════════════════════════════════
#  PPE — Pitch Period Entropy
# ══════════════════════════════════════════════════════════════════════

def _ppe(f0: np.ndarray) -> float:
    """Pitch Period Entropy — entropy of semitone distribution."""
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        return 0.0
    median_f0 = np.median(voiced)
    semitones = 12 * np.log2(voiced / (median_f0 + 1e-10))
    n_bins = max(3, min(20, len(semitones) // 3))
    hist, _ = np.histogram(semitones, bins=n_bins, density=True)
    bin_width = (semitones.max() - semitones.min() + 1e-10) / n_bins
    probs = hist * bin_width
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ══════════════════════════════════════════════════════════════════════
#  Temporal — Speech Rate & Pause
# ══════════════════════════════════════════════════════════════════════

def _speech_rate(y: np.ndarray, sr: int = SR) -> float:
    """Ước tính tốc độ nói (onset/giây)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    dur = len(y) / sr
    return len(onsets) / dur if dur > 0 else 0.0


def _pause_ratio(y: np.ndarray, sr: int = SR) -> float:
    """Tỷ lệ khoảng lặng trong audio."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    threshold = np.mean(rms) * 0.3
    return float(np.sum(rms < threshold) / (len(rms) + 1e-10))


# ══════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_parkinson_features(audio_path: Path) -> Dict[str, float]:
    """Trích xuất toàn bộ đặc trưng liên quan Parkinson từ file audio."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.5:
        yt = y

    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    mfcc = librosa.feature.mfcc(y=yt, sr=SR, n_mfcc=13)

    features = {
        # F0 statistics
        "f0_mean": float(np.mean(voiced)) if len(voiced) else 0.0,
        "f0_std": float(np.std(voiced)) if len(voiced) else 0.0,
        "f0_min": float(np.min(voiced)) if len(voiced) else 0.0,
        "f0_max": float(np.max(voiced)) if len(voiced) else 0.0,
        "f0_range": float(np.ptp(voiced)) if len(voiced) else 0.0,
        # Jitter
        "jitter_local": _jitter_local(f0),
        "jitter_rap": _jitter_rap(f0),
        "jitter_ppq5": _jitter_ppq5(f0),
        "jitter_ddp": _jitter_ddp(f0),
        # Shimmer
        "shimmer_local": _shimmer_local(yt),
        "shimmer_db": _shimmer_db(yt),
        "shimmer_apq3": _shimmer_apq(yt, SR, 3),
        "shimmer_apq5": _shimmer_apq(yt, SR, 5),
        # Noise
        "hnr": _hnr(yt),
        "nhr": _nhr(yt),
        # Entropy
        "ppe": _ppe(f0),
        # Temporal
        "speech_rate": _speech_rate(yt),
        "pause_ratio": _pause_ratio(yt),
        "duration_sec": round(len(yt) / SR, 2),
    }

    # MFCC (13 hệ số × 2 thống kê = 26 features)
    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    return features


# ══════════════════════════════════════════════════════════════════════
#  SCREENING — Rule-based risk assessment
# ══════════════════════════════════════════════════════════════════════

def screen_parkinson(audio_path: Path) -> Dict:
    """Sàng lọc dấu hiệu Parkinson qua phân tích giọng nói.

    Returns dict gồm: risk_score, risk_level, indicators, features,
    advice, disclaimer.
    """
    feats = extract_parkinson_features(audio_path)
    indicators = []
    risk_score = 0.0

    # ── 1. Jitter local (BT < 1%, PD > 1.5%) ──
    if feats["jitter_local"] > 0.020:
        indicators.append({
            "feature": "Jitter cao bất thường",
            "detail": f"Jitter = {feats['jitter_local']:.4f} (ngưỡng BT < 0.010)",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["jitter_local"] > 0.012:
        indicators.append({
            "feature": "Jitter hơi cao",
            "detail": f"Jitter = {feats['jitter_local']:.4f}",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # ── 2. Shimmer local (BT < 3%, PD > 3.8%) ──
    if feats["shimmer_local"] > 0.15:
        indicators.append({
            "feature": "Shimmer cao",
            "detail": f"Shimmer = {feats['shimmer_local']:.4f} (ngưỡng BT < 0.10)",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["shimmer_local"] > 0.10:
        indicators.append({
            "feature": "Shimmer hơi cao",
            "detail": f"Shimmer = {feats['shimmer_local']:.4f}",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # ── 3. HNR (BT > 20 dB, PD < 20 dB) ──
    if feats["hnr"] < 10:
        indicators.append({
            "feature": "HNR rất thấp — giọng khàn/thở",
            "detail": f"HNR = {feats['hnr']:.1f} dB (bình thường > 20 dB)",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["hnr"] < 20:
        indicators.append({
            "feature": "HNR thấp",
            "detail": f"HNR = {feats['hnr']:.1f} dB",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # ── 4. F0 range (giảm trong PD → giọng monotone) ──
    if feats["f0_mean"] > 0:
        if feats["f0_range"] < 30:
            indicators.append({
                "feature": "Dải tần giọng rất hẹp (monotone)",
                "detail": f"F0 range = {feats['f0_range']:.1f} Hz",
                "severity": "cao", "weight": 0.12,
            })
            risk_score += 0.12
        elif feats["f0_range"] < 60:
            indicators.append({
                "feature": "Dải tần giọng hơi hẹp",
                "detail": f"F0 range = {feats['f0_range']:.1f} Hz",
                "severity": "trung bình", "weight": 0.06,
            })
            risk_score += 0.06

    # ── 5. PPE (tăng trong PD) ──
    if feats["ppe"] > 0.5:
        indicators.append({
            "feature": "Entropy pitch cao (PPE)",
            "detail": f"PPE = {feats['ppe']:.3f}",
            "severity": "cao", "weight": 0.12,
        })
        risk_score += 0.12
    elif feats["ppe"] > 0.3:
        indicators.append({
            "feature": "PPE hơi cao",
            "detail": f"PPE = {feats['ppe']:.3f}",
            "severity": "trung bình", "weight": 0.06,
        })
        risk_score += 0.06

    # ── 6. Nhịp nói chậm ──
    if feats["speech_rate"] < 1.5:
        indicators.append({
            "feature": "Nhịp nói rất chậm",
            "detail": f"{feats['speech_rate']:.1f} onset/s",
            "severity": "cao", "weight": 0.10,
        })
        risk_score += 0.10
    elif feats["speech_rate"] < 2.5:
        indicators.append({
            "feature": "Nhịp nói chậm",
            "detail": f"{feats['speech_rate']:.1f} onset/s",
            "severity": "trung bình", "weight": 0.05,
        })
        risk_score += 0.05

    # ── 7. Khoảng ngừng nhiều ──
    if feats["pause_ratio"] > 0.50:
        indicators.append({
            "feature": "Nhiều khoảng ngừng",
            "detail": f"{feats['pause_ratio']:.0%} thời gian im lặng",
            "severity": "cao", "weight": 0.10,
        })
        risk_score += 0.10
    elif feats["pause_ratio"] > 0.35:
        indicators.append({
            "feature": "Khoảng ngừng khá nhiều",
            "detail": f"{feats['pause_ratio']:.0%}",
            "severity": "trung bình", "weight": 0.05,
        })
        risk_score += 0.05

    # ── 8. NHR (tăng trong PD) ──
    if feats["nhr"] > 0.10:
        indicators.append({
            "feature": "Tỷ lệ nhiễu cao (NHR)",
            "detail": f"NHR = {feats['nhr']:.4f}",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # ── 9. Shimmer dB ──
    if feats["shimmer_db"] > 0.50:
        indicators.append({
            "feature": "Shimmer(dB) cao",
            "detail": f"Shimmer(dB) = {feats['shimmer_db']:.3f}",
            "severity": "trung bình", "weight": 0.07,
        })
        risk_score += 0.07

    # ── Tổng hợp ──
    risk_score = min(risk_score, 1.0)
    risk_pct = round(risk_score * 100, 1)

    if risk_pct >= 60:
        level, level_vi, advice = (
            "cao",
            "⚠️ Nguy cơ cao",
            "Phát hiện nhiều dấu hiệu bất thường trong giọng nói có liên quan "
            "đến bệnh Parkinson. Khuyến nghị tham khảo ý kiến bác sĩ chuyên "
            "khoa thần kinh để được đánh giá chuyên sâu.",
        )
    elif risk_pct >= 30:
        level, level_vi, advice = (
            "trung bình",
            "🔶 Nguy cơ trung bình",
            "Một số đặc trưng giọng nói cho thấy dấu hiệu bất thường nhẹ. "
            "Nên theo dõi định kỳ và tham vấn bác sĩ nếu xuất hiện thêm "
            "triệu chứng vận động (run tay, cứng cơ, chậm vận động).",
        )
    else:
        level, level_vi, advice = (
            "thấp",
            "✅ Nguy cơ thấp",
            "Giọng nói trong phạm vi bình thường. Không phát hiện dấu hiệu "
            "rối loạn giọng nói liên quan đến bệnh Parkinson.",
        )

    return {
        "risk_score": risk_pct,
        "risk_level": level,
        "risk_level_vi": level_vi,
        "advice": advice,
        "indicators": indicators,
        "indicator_count": len(indicators),
        "features": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in feats.items()
        },
        "hardware_info": {
            "microphone": "INMP441 (I2S MEMS) — mô phỏng qua mic máy tính",
            "processor": "Raspberry Pi 4 Model B — mô phỏng trên máy chủ",
            "sample_rate": f"{SR} Hz",
            "bit_depth": "24-bit (INMP441 native) → 16-bit (processed)",
        },
        "disclaimer": (
            "Đây chỉ là công cụ sàng lọc sơ bộ dựa trên phân tích giọng nói, "
            "KHÔNG thay thế chẩn đoán y khoa. Nếu có lo ngại, hãy liên hệ "
            "bác sĩ chuyên khoa thần kinh."
        ),
    }
