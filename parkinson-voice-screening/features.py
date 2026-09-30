"""
features.py — Trích xuất đặc trưng giọng nói cho phân tích Parkinson.

Các đặc trưng được trích xuất (tổng cộng 62+ đặc trưng):
  ── Nhóm 1: Tần số cơ bản F0 ──
  - F0 mean, std, min, max, range         → thống kê tần số cơ bản

  ── Nhóm 2: Bất ổn định chu kỳ Jitter ──
  - Jitter (local, RAP, PPQ5, DDP)        → bất ổn định chu kỳ

  ── Nhóm 3: Bất ổn định biên độ Shimmer ──
  - Shimmer (local, dB, APQ3, APQ5)       → bất ổn định biên độ

  ── Nhóm 4: Tỷ lệ hài âm / nhiễu ──
  - HNR / NHR                             → tỷ lệ hài âm / nhiễu

  ── Nhóm 5: Phi tuyến & Entropy (MỚI — Phase 7) ──
  - PPE (Pitch Period Entropy)            → entropy cao độ
  - DFA (Detrended Fluctuation Analysis)  → phân tích dao động phi xu hướng
  - RPDE (Recurrence Period Density Entropy) → entropy tái hiện chu kỳ
  - Spread1, Spread2                      → phân tán tần số cơ bản
  - D2 (Correlation Dimension)            → chiều tương quan phi tuyến

  ── Nhóm 6: Đặc trưng phổ (MỚI — Phase 7) ──
  - Spectral Centroid, Bandwidth, Rolloff → đặc trưng miền tần số
  - Zero Crossing Rate, RMS Energy        → đặc trưng miền thời gian

  ── Nhóm 7: Thời gian ──
  - Speech rate & pause ratio             → tốc độ nói

  ── Nhóm 8: MFCC ──
  - MFCC (13 hệ số × mean & std = 26)    → hệ số cepstral mel

Phần cứng mục tiêu: Microphone INMP441 (I2S MEMS) + Raspberry Pi 4
Tập dữ liệu: UCI Parkinson's Voice Dataset (195 mẫu, 22 đặc trưng)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import librosa

# ─── Hằng số ─────────────────────────────────────────────────────────────────
SR          = 16_000   # Tần số lấy mẫu (Hz) — INMP441 hỗ trợ 8–48 kHz
HOP_LENGTH  = 512
N_MFCC      = 13

# Danh sách tên đặc trưng (theo thứ tự feature vector)
FEATURE_NAMES: List[str] = [
    # F0
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
    # Jitter
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    # Shimmer
    "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5",
    # Noise
    "hnr", "nhr",
    # Entropy & Nonlinear (Phase 7)
    "ppe", "rpde", "dfa", "spread1", "spread2", "d2",
    # Spectral (Phase 7)
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "zero_crossing_rate", "rms_energy",
    # Temporal
    "speech_rate", "pause_ratio",
] + [f"mfcc_{i+1}_mean" for i in range(N_MFCC)] \
  + [f"mfcc_{i+1}_std"  for i in range(N_MFCC)]


# ══════════════════════════════════════════════════════════════════════════════
#  JITTER — Bất ổn định chu kỳ dao động dây thanh
# ══════════════════════════════════════════════════════════════════════════════

def _jitter_local(f0: np.ndarray) -> float:
    """Jitter (local) — biến thiên tương đối từng chu kỳ liền kề."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    return float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10))


def _jitter_rap(f0: np.ndarray) -> float:
    """Jitter (RAP) — Relative Average Perturbation (làm mượt 3 điểm)."""
    voiced = f0[f0 > 0]
    if len(voiced) < 5:
        return 0.0
    periods  = 1.0 / voiced
    smoothed = np.convolve(periods, [1/3, 1/3, 1/3], mode="valid")
    trimmed  = periods[1:-1]
    return float(np.mean(np.abs(trimmed - smoothed)) / (np.mean(periods) + 1e-10))


def _jitter_ppq5(f0: np.ndarray) -> float:
    """Jitter (PPQ5) — Period Perturbation Quotient 5 điểm."""
    voiced = f0[f0 > 0]
    if len(voiced) < 7:
        return 0.0
    periods  = 1.0 / voiced
    smoothed = np.convolve(periods, np.ones(5) / 5, mode="valid")
    trimmed  = periods[2:-2]
    return float(np.mean(np.abs(trimmed - smoothed)) / (np.mean(periods) + 1e-10))


def _jitter_ddp(f0: np.ndarray) -> float:
    """Jitter (DDP) — trung bình |diff(diff(periods))|."""
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return 0.0
    periods = 1.0 / voiced
    return float(np.mean(np.abs(np.diff(np.diff(periods)))) / (np.mean(periods) + 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
#  SHIMMER — Bất ổn định biên độ
# ══════════════════════════════════════════════════════════════════════════════

def _get_frame_amps(y: np.ndarray, sr: int = SR) -> np.ndarray:
    fl = max(256, int(sr * 0.03))
    hl = max(64,  int(sr * 0.01))
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    return np.max(np.abs(frames), axis=0)


def _shimmer_local(y: np.ndarray, sr: int = SR) -> float:
    amps = _get_frame_amps(y, sr)
    if len(amps) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-10))


def _shimmer_db(y: np.ndarray, sr: int = SR) -> float:
    amps = _get_frame_amps(y, sr)
    amps = amps[amps > 0]
    if len(amps) < 3:
        return 0.0
    db = 20 * np.log10(amps + 1e-10)
    return float(np.mean(np.abs(np.diff(db))))


def _shimmer_apq(y: np.ndarray, sr: int = SR, n: int = 3) -> float:
    amps = _get_frame_amps(y, sr)
    if len(amps) < n + 2:
        return 0.0
    smoothed = np.convolve(amps, np.ones(n) / n, mode="valid")
    offset   = n // 2
    trimmed  = amps[offset: offset + len(smoothed)]
    return float(np.mean(np.abs(trimmed - smoothed)) / (np.mean(amps) + 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
#  HNR / NHR — Harmonics & Noise Ratio
# ══════════════════════════════════════════════════════════════════════════════

def _hnr(y: np.ndarray, sr: int = SR) -> float:
    """Harmonics-to-Noise Ratio (dB) — càng cao giọng càng sạch."""
    ac = librosa.autocorrelate(y, max_size=sr // 60)
    if len(ac) < 2 or ac[0] == 0:
        return 0.0
    peak  = np.max(ac[1:])
    ratio = np.clip(peak / ac[0], 1e-10, 1 - 1e-10)
    return float(10 * np.log10(ratio / (1 - ratio)))


def _nhr(y: np.ndarray, sr: int = SR) -> float:
    """Noise-to-Harmonics Ratio — nghịch đảo tuyến tính của HNR."""
    h = _hnr(y, sr)
    return float(10 ** (-h / 10)) if h != 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  PPE — Pitch Period Entropy
# ══════════════════════════════════════════════════════════════════════════════

def _ppe(f0: np.ndarray) -> float:
    """PPE — entropy phân bố semitone của F0."""
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        return 0.0
    med  = np.median(voiced)
    semi = 12 * np.log2(voiced / (med + 1e-10))
    n_bins = max(3, min(20, len(semi) // 3))
    hist, _ = np.histogram(semi, bins=n_bins, density=True)
    w   = (semi.max() - semi.min() + 1e-10) / n_bins
    probs = hist * w
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ══════════════════════════════════════════════════════════════════════════════
#  RPDE — Recurrence Period Density Entropy (Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

def _rpde(y: np.ndarray, sr: int = SR, dim: int = 6, tau: int = 2) -> float:
    """
    Recurrence Period Density Entropy (RPDE) — đo mức độ bất ổn định
    của dao động dây thanh trong không gian phase tái cấu trúc.

    Giá trị RPDE cao → giọng nói không ổn định → nguy cơ Parkinson.
    Tham khảo: Little et al. (2007), Max A. Little (2019).
    """
    # Downsample để giảm tính toán
    if len(y) > sr * 3:
        y = y[:sr * 3]

    # Embedding delay
    n = len(y)
    m = n - (dim - 1) * tau
    if m < 100:
        return 0.0

    # Tái cấu trúc không gian phase (time-delay embedding)
    embedded = np.array([y[i:i + dim * tau:tau] for i in range(m)])

    # Tìm nearest neighbor cho mỗi điểm
    n_pts = min(m, 800)
    indices = np.linspace(0, m - 1, n_pts, dtype=int)
    recurrence_times = []

    for idx in indices:
        ref = embedded[idx]
        dists = np.sqrt(np.sum((embedded - ref) ** 2, axis=1))
        dists[max(0, idx-5):min(m, idx+5)] = np.inf  # Loại Theiler window

        threshold = np.percentile(dists[dists < np.inf], 10) if np.any(dists < np.inf) else 1.0
        close = np.where(dists < threshold)[0]
        if len(close) > 1:
            rt = np.diff(close)
            recurrence_times.extend(rt.tolist())

    if len(recurrence_times) < 10:
        return 0.0

    rt = np.array(recurrence_times)
    n_bins = max(5, min(50, len(rt) // 10))
    hist, _ = np.histogram(rt, bins=n_bins, density=True)
    w = (rt.max() - rt.min() + 1e-10) / n_bins
    probs = hist * w
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Chuẩn hóa về [0, 1]
    max_entropy = np.log2(n_bins)
    return float(np.clip(entropy / (max_entropy + 1e-10), 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
#  DFA — Detrended Fluctuation Analysis (Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

def _dfa(y: np.ndarray, sr: int = SR) -> float:
    """
    Detrended Fluctuation Analysis (DFA) — đo tương quan dài hạn
    trong tín hiệu giọng nói. Giá trị DFA cao → mất tính tự tương tự.

    Tham khảo: Peng et al. (1994), Little et al. (2007).
    """
    # Dùng F0 contour cho DFA (ý nghĩa hơn raw waveform)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    if len(voiced) < 20:
        return 0.75  # Giá trị mặc định hợp lý

    # Profile: tích lũy sai lệch so với trung bình
    mean_val = np.mean(voiced)
    profile = np.cumsum(voiced - mean_val)
    N = len(profile)

    # Các kích thước cửa sổ (box sizes)
    min_box = 4
    max_box = N // 4
    if max_box < min_box + 2:
        return 0.75

    n_boxes = min(15, max_box - min_box)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), n_boxes
    ).astype(int))

    fluctuations = []
    for box in box_sizes:
        n_boxes_fit = N // box
        if n_boxes_fit < 1:
            continue
        rms_list = []
        for i in range(n_boxes_fit):
            segment = profile[i * box:(i + 1) * box]
            x = np.arange(box)
            # Khử xu hướng bậc 1 (linear detrend)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend
            rms_list.append(np.sqrt(np.mean(residual ** 2)))
        if rms_list:
            fluctuations.append(np.mean(rms_list))
        else:
            fluctuations.append(1e-10)

    if len(box_sizes) < 3 or len(fluctuations) < 3:
        return 0.75

    # Fit log-log → hệ số góc = DFA exponent
    valid = np.array(fluctuations) > 0
    if np.sum(valid) < 3:
        return 0.75
    log_n = np.log(box_sizes[valid].astype(float))
    log_f = np.log(np.array(fluctuations)[valid])
    coeffs = np.polyfit(log_n, log_f, 1)

    return float(np.clip(coeffs[0], 0.3, 1.5))


# ══════════════════════════════════════════════════════════════════════════════
#  SPREAD & D2 — Nonlinear Dynamics Features (Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

def _spread_features(f0: np.ndarray) -> tuple:
    """
    Spread1, Spread2 — đo mức phân tán của phân bố F0.
    Spread1: phương sai logarit chính (principal component variance)
    Spread2: phương sai logarit phụ
    """
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        return -5.0, 0.2

    # Log F0
    log_f0 = np.log(voiced + 1e-10)
    mean_lf0 = np.mean(log_f0)

    # Spread1: biến thiên chủ yếu (âm, càng âm → phân bố tập trung)
    deviation = log_f0 - mean_lf0
    spread1 = float(-np.mean(deviation ** 2) * 10)  # Scale tương đương UCI
    spread1 = np.clip(spread1, -10, 0)

    # Spread2: biến thiên thứ cấp (dương, nhỏ)
    if len(voiced) > 3:
        autocorr = np.correlate(deviation, deviation, mode="full")
        autocorr = autocorr[len(autocorr)//2:]
        if len(autocorr) > 1 and autocorr[0] > 0:
            spread2 = float(autocorr[1] / autocorr[0])
        else:
            spread2 = 0.2
    else:
        spread2 = 0.2

    return spread1, np.clip(spread2, 0, 1)


def _d2(y: np.ndarray, sr: int = SR) -> float:
    """
    D2 (Correlation Dimension) — ước lượng chiều fractal
    của attractor trong không gian phase.

    Tham khảo: Grassberger-Procaccia algorithm.
    """
    # Đơn giản hóa: dùng F0 contour
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    if len(voiced) < 30:
        return 2.3  # Giá trị hợp lý mặc định

    # Time-delay embedding
    dim, tau = 3, 2
    m = len(voiced) - (dim - 1) * tau
    if m < 20:
        return 2.3

    embedded = np.array([voiced[i:i + dim * tau:tau] for i in range(m)])

    # Correlation integral approximation
    n_sample = min(m, 300)
    idx = np.random.RandomState(42).choice(m, n_sample, replace=False)
    sub = embedded[idx]

    # Tính khoảng cách giữa tất cả các cặp
    dists = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            dists.append(np.sqrt(np.sum((sub[i] - sub[j]) ** 2)))

    if len(dists) < 10:
        return 2.3

    dists = np.array(dists)
    dists = dists[dists > 0]

    # Ước lượng D2 qua log-log slope
    epsilons = np.percentile(dists, [10, 20, 30, 40, 50, 60, 70, 80])
    counts = []
    for eps in epsilons:
        c = np.sum(dists < eps) / len(dists)
        counts.append(max(c, 1e-10))

    log_eps = np.log(epsilons)
    log_c = np.log(np.array(counts))

    valid = np.isfinite(log_eps) & np.isfinite(log_c)
    if np.sum(valid) < 3:
        return 2.3

    coeffs = np.polyfit(log_eps[valid], log_c[valid], 1)
    return float(np.clip(coeffs[0], 1.0, 4.0))


# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES (Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

def _spectral_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """Trích xuất 5 đặc trưng phổ tần số."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    return {
        "spectral_centroid":   float(np.mean(centroid)),
        "spectral_bandwidth":  float(np.mean(bandwidth)),
        "spectral_rolloff":    float(np.mean(rolloff)),
        "zero_crossing_rate":  float(np.mean(zcr)),
        "rms_energy":          float(np.mean(rms)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL — Speech rate & Pause ratio
# ══════════════════════════════════════════════════════════════════════════════

def _speech_rate(y: np.ndarray, sr: int = SR) -> float:
    env    = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    dur    = len(y) / sr
    return len(onsets) / dur if dur > 0 else 0.0


def _pause_ratio(y: np.ndarray, sr: int = SR) -> float:
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    threshold = np.mean(rms) * 0.3
    return float(np.sum(rms < threshold) / (len(rms) + 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(audio_path: str | Path) -> Dict[str, float]:
    """
    Trích xuất toàn bộ đặc trưng giọng nói từ file audio.

    Parameters
    ----------
    audio_path : đường dẫn file .wav / .mp3 / .ogg

    Returns
    -------
    dict ánh xạ tên đặc trưng → giá trị float (62+ đặc trưng)
    """
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.3:
        yt = y  # quá ngắn sau trim → dùng nguyên

    # F0 via pYIN (chính xác hơn YIN cho giọng người)
    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    # MFCC
    mfcc = librosa.feature.mfcc(y=yt, sr=SR, n_mfcc=N_MFCC)

    # Spread features
    spread1, spread2 = _spread_features(f0)

    feats: Dict[str, float] = {
        # ── F0 statistics ──
        "f0_mean"  : float(np.mean(voiced))  if len(voiced) else 0.0,
        "f0_std"   : float(np.std(voiced))   if len(voiced) else 0.0,
        "f0_min"   : float(np.min(voiced))   if len(voiced) else 0.0,
        "f0_max"   : float(np.max(voiced))   if len(voiced) else 0.0,
        "f0_range" : float(np.ptp(voiced))   if len(voiced) else 0.0,

        # ── Jitter ──
        "jitter_local" : _jitter_local(f0),
        "jitter_rap"   : _jitter_rap(f0),
        "jitter_ppq5"  : _jitter_ppq5(f0),
        "jitter_ddp"   : _jitter_ddp(f0),

        # ── Shimmer ──
        "shimmer_local" : _shimmer_local(yt),
        "shimmer_db"    : _shimmer_db(yt),
        "shimmer_apq3"  : _shimmer_apq(yt, SR, 3),
        "shimmer_apq5"  : _shimmer_apq(yt, SR, 5),

        # ── Noise ──
        "hnr" : _hnr(yt),
        "nhr" : _nhr(yt),

        # ── Entropy & Nonlinear (Phase 7) ──
        "ppe"     : _ppe(f0),
        "rpde"    : _rpde(yt, SR),
        "dfa"     : _dfa(yt, SR),
        "spread1" : spread1,
        "spread2" : spread2,
        "d2"      : _d2(yt, SR),

        # ── Temporal ──
        "speech_rate" : _speech_rate(yt),
        "pause_ratio" : _pause_ratio(yt),
    }

    # ── Spectral features (Phase 7) ──
    feats.update(_spectral_features(yt, SR))

    # ── MFCC (13 × 2 = 26 đặc trưng) ──
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    return feats


def feature_vector(feats: Dict[str, float]) -> np.ndarray:
    """Chuyển dict đặc trưng → numpy array theo thứ tự FEATURE_NAMES."""
    return np.array([feats.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
