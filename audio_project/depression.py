"""Sàng lọc dấu hiệu trầm cảm qua giọng nói — tích hợp DTW.

Rule-based: pitch, energy, speech rate, pause, jitter, shimmer, HNR.
DTW-based : so sánh pitch-contour và energy-envelope với baseline tổng hợp.

Baseline synthetic:
  - Pitch bình thường: ~185 Hz, biến thiên ~30 Hz (sin wave, deterministic)
  - Energy bình thường: ~0.05, dao động nhẹ (deterministic)

Dựa trên: Cummins et al. (2015), Low et al. (2011).
"""
from pathlib import Path
from typing import Dict
import librosa, numpy as np

SR         = 16000
HOP_LENGTH = 512
_N_PITCH   = 80    # số frames pitch prototype
_N_ENERGY  = 60    # số frames energy prototype


# ── DTW (1-D, Sakoe-Chiba) ─────────────────────────────────────────────────
def _dtw(a: np.ndarray, b: np.ndarray, radius: int = 8) -> float:
    """Khoảng cách DTW chuẩn hóa giữa hai chuỗi 1-D (đã z-score hoặc [0,1])."""
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / n
    for i in range(1, n + 1):
        jc = int(round(i * scale))
        j0 = max(1, jc - radius)
        j1 = min(m, jc + radius) + 1
        for j in range(j0, j1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


# ── Deterministic baselines (sin-wave, no random) ─────────────────────────
def _baseline_pitch(n: int = _N_PITCH) -> np.ndarray:
    """
    Pitch baseline bình thường (deterministic):
    trung bình 185 Hz, biến thiên ±30 Hz theo sin — z-score normalized.
    """
    t = np.linspace(0, 2 * np.pi * 3, n)
    raw = 185.0 + 30 * np.sin(t)
    return (raw - raw.mean()) / (raw.std() + 1e-10)


def _baseline_energy(n: int = _N_ENERGY) -> np.ndarray:
    """
    Energy baseline bình thường (deterministic):
    khoảng 0.05 ± 0.02 dao động nhẹ — normalized về [0,1].
    """
    t = np.linspace(0, 2 * np.pi * 2, n)
    raw = 0.05 + 0.02 * np.sin(t)
    return raw / (raw.max() + 1e-10)


# Cache baselines
_BL_PITCH  = _baseline_pitch(_N_PITCH)
_BL_ENERGY = _baseline_energy(_N_ENERGY)


def _resample_1d(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Resample mảng 1-D về n_out điểm bằng nội suy tuyến tính."""
    idx = np.linspace(0, len(arr) - 1, n_out)
    return np.interp(idx, np.arange(len(arr)), arr)


# ── DTW Depression Scoring ─────────────────────────────────────────────────
def _dtw_depression_score(f0: np.ndarray, rms: np.ndarray) -> Dict[str, float]:
    """
    So sánh pitch-contour và energy-envelope của giọng với baseline bình thường.
    Cả hai đều được z-score / normalize trước khi tính DTW.
    Trả về dist và risk [0,1].
    """
    voiced = f0[f0 > 0]

    # ── Pitch DTW ──
    if len(voiced) >= 10:
        # Z-score normalize voiced pitch
        p_z = (voiced - voiced.mean()) / (voiced.std() + 1e-10)
        p_rs = _resample_1d(p_z, _N_PITCH)
        pitch_dist = _dtw(p_rs, _BL_PITCH, radius=10)
    else:
        pitch_dist = 3.0   # penalty nếu quá ít voiced frame

    # ── Energy DTW ──
    rms_clipped = np.clip(rms, 0, None)
    rms_norm    = rms_clipped / (rms_clipped.max() + 1e-10)
    rms_rs      = _resample_1d(rms_norm, _N_ENERGY)
    energy_dist = _dtw(rms_rs, _BL_ENERGY, radius=8)

    # Risk mapping: sigmoid, pitch thường dist ~0.3-0.8, energy ~0.1-0.4
    def _sigmoid_risk(d: float, midpoint: float, steepness: float = 6.0) -> float:
        return float(1.0 / (1.0 + np.exp(-steepness * (d - midpoint))))

    pitch_risk  = _sigmoid_risk(pitch_dist,  midpoint=0.60)
    energy_risk = _sigmoid_risk(energy_dist, midpoint=0.25)

    return {
        "dtw_pitch_dist":  round(pitch_dist, 4),
        "dtw_energy_dist": round(energy_dist, 4),
        "dtw_pitch_risk":  round(pitch_risk, 3),
        "dtw_energy_risk": round(energy_risk, 3),
    }


# ── Acoustic Feature Extractors ────────────────────────────────────────────
def _jitter(f0: np.ndarray) -> float:
    """Relative jitter — biến thiên chu kỳ giọng nói."""
    voiced = f0[f0 > 0]
    if len(voiced) < 3:
        return 0.0
    periods = 1.0 / voiced
    diffs   = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-10))


def _shimmer(y: np.ndarray, sr: int = SR) -> float:
    """Shimmer — biến thiên biên độ giữa các chu kỳ."""
    fl  = int(sr * 0.03)
    hl  = int(sr * 0.01)
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    amps   = np.max(np.abs(frames), axis=0)
    if len(amps) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-10))


def _hnr(y: np.ndarray, sr: int = SR) -> float:
    """Harmonics-to-Noise Ratio (dB)."""
    ac  = librosa.autocorrelate(y, max_size=sr // 60)
    if len(ac) < 2 or ac[0] == 0:
        return 0.0
    peak  = np.max(ac[1:])
    ratio = peak / ac[0]
    return float(10 * np.log10(ratio / (1 - ratio + 1e-10) + 1e-10))


def _speech_rate(y: np.ndarray, sr: int = SR) -> float:
    """Ước tính tốc độ nói (onset/giây)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets    = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    dur       = len(y) / sr
    return len(onsets) / dur if dur > 0 else 0.0


def _pause_ratio(y: np.ndarray, sr: int = SR) -> float:
    """Tỷ lệ khoảng lặng trong audio."""
    rms       = librosa.feature.rms(y=y, hop_length=512)[0]
    threshold = np.mean(rms) * 0.3
    silent    = np.sum(rms < threshold)
    return float(silent / (len(rms) + 1e-10))


# ── Main extraction ────────────────────────────────────────────────────────
def extract_depression_features(audio_path: Path) -> Dict[str, float]:
    """Trích xuất toàn bộ đặc trưng liên quan trầm cảm (bao gồm DTW)."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.5:
        yt = y

    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0   = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]
    rms  = librosa.feature.rms(y=yt, hop_length=HOP_LENGTH)[0]

    feats = {
        "pitch_mean":  float(np.mean(voiced)) if len(voiced) > 0 else 0.0,
        "pitch_std":   float(np.std(voiced))  if len(voiced) > 0 else 0.0,
        "pitch_range": float(np.ptp(voiced))  if len(voiced) > 0 else 0.0,
        "energy_mean": float(np.mean(rms)),
        "energy_std":  float(np.std(rms)),
        "jitter":      _jitter(f0),
        "shimmer":     _shimmer(yt),
        "hnr":         _hnr(yt),
        "speech_rate": _speech_rate(yt),
        "pause_ratio": _pause_ratio(yt),
        "duration_sec": round(len(yt) / SR, 2),
    }

    dtw_info = _dtw_depression_score(f0, rms)
    feats.update(dtw_info)
    return feats


# ── Screening ──────────────────────────────────────────────────────────────
def screen_depression(audio_path: Path) -> Dict:
    """Sàng lọc dấu hiệu trầm cảm — rule-based + DTW."""
    feats      = extract_depression_features(audio_path)
    indicators = []
    risk_score = 0.0

    # 1. Pitch
    if feats["pitch_mean"] > 0:
        if feats["pitch_std"] < 15:
            indicators.append({
                "feature": "Giọng đơn điệu",
                "detail":  f"Biến thiên pitch thấp ({feats['pitch_std']:.1f} Hz)",
                "severity": "cao", "weight": 0.20,
            })
            risk_score += 0.20
        elif feats["pitch_std"] < 25:
            indicators.append({
                "feature": "Giọng ít biến thiên",
                "detail":  f"Pitch std = {feats['pitch_std']:.1f} Hz",
                "severity": "trung bình", "weight": 0.10,
            })
            risk_score += 0.10

    # 2. Energy
    if feats["energy_mean"] < 0.02:
        indicators.append({
            "feature": "Giọng yếu",
            "detail":  f"Energy = {feats['energy_mean']:.4f}",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["energy_mean"] < 0.04:
        indicators.append({
            "feature": "Giọng hơi yếu",
            "detail":  f"Energy = {feats['energy_mean']:.4f}",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # 3. Speech rate
    if feats["speech_rate"] < 2.0:
        indicators.append({
            "feature": "Nhịp nói chậm",
            "detail":  f"{feats['speech_rate']:.1f} onset/giây",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["speech_rate"] < 3.0:
        indicators.append({
            "feature": "Nhịp nói hơi chậm",
            "detail":  f"{feats['speech_rate']:.1f} onset/giây",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # 4. Pause ratio
    if feats["pause_ratio"] > 0.5:
        indicators.append({
            "feature": "Nhiều khoảng lặng",
            "detail":  f"{feats['pause_ratio']:.0%} thời gian im lặng",
            "severity": "cao", "weight": 0.15,
        })
        risk_score += 0.15
    elif feats["pause_ratio"] > 0.35:
        indicators.append({
            "feature": "Khoảng lặng khá nhiều",
            "detail":  f"{feats['pause_ratio']:.0%}",
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # 5. Jitter
    if feats["jitter"] > 0.03:
        indicators.append({
            "feature": "Giọng run (jitter cao)",
            "detail":  f"Jitter = {feats['jitter']:.4f}",
            "severity": "cao", "weight": 0.10,
        })
        risk_score += 0.10
    elif feats["jitter"] > 0.015:
        indicators.append({
            "feature": "Jitter hơi cao",
            "detail":  f"Jitter = {feats['jitter']:.4f}",
            "severity": "trung bình", "weight": 0.05,
        })
        risk_score += 0.05

    # 6. Shimmer
    if feats["shimmer"] > 0.15:
        indicators.append({
            "feature": "Biên độ không ổn định (shimmer)",
            "detail":  f"Shimmer = {feats['shimmer']:.4f}",
            "severity": "cao", "weight": 0.10,
        })
        risk_score += 0.10

    # 7. HNR
    if feats["hnr"] < 5:
        indicators.append({
            "feature": "Giọng thở/khàn (HNR thấp)",
            "detail":  f"HNR = {feats['hnr']:.1f} dB",
            "severity": "trung bình", "weight": 0.10,
        })
        risk_score += 0.10

    # 8. DTW pitch
    if feats.get("dtw_pitch_risk", 0) > 0.60:
        indicators.append({
            "feature": "Đường pitch bất thường (DTW)",
            "detail":  (f"DTW dist = {feats['dtw_pitch_dist']:.3f}, "
                        f"risk = {feats['dtw_pitch_risk']:.2f}"),
            "severity": "trung bình", "weight": 0.08,
        })
        risk_score += 0.08

    # 9. DTW energy
    if feats.get("dtw_energy_risk", 0) > 0.60:
        indicators.append({
            "feature": "Năng lượng giọng bất thường (DTW)",
            "detail":  (f"DTW dist = {feats['dtw_energy_dist']:.3f}, "
                        f"risk = {feats['dtw_energy_risk']:.2f}"),
            "severity": "trung bình", "weight": 0.07,
        })
        risk_score += 0.07

    risk_score = min(risk_score, 1.0)
    risk_pct   = round(risk_score * 100, 1)

    if risk_pct >= 60:
        level, level_vi, advice = (
            "cao", "⚠️ Nguy cơ cao",
            "Có nhiều dấu hiệu bất thường. Khuyến nghị tham vấn chuyên gia tâm lý.",
        )
    elif risk_pct >= 30:
        level, level_vi, advice = (
            "trung bình", "🔶 Nguy cơ trung bình",
            "Một số đặc trưng giọng bất thường. Nên nghỉ ngơi và theo dõi.",
        )
    else:
        level, level_vi, advice = (
            "thấp", "✅ Nguy cơ thấp",
            "Giọng nói trong phạm vi bình thường. Không phát hiện dấu hiệu bất thường.",
        )

    return {
        "risk_score":      risk_pct,
        "risk_level":      level,
        "risk_level_vi":   level_vi,
        "advice":          advice,
        "indicators":      indicators,
        "indicator_count": len(indicators),
        "features":        {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in feats.items()},
        "dtw_analysis": {
            "pitch_distance":  feats.get("dtw_pitch_dist"),
            "energy_distance": feats.get("dtw_energy_dist"),
            "pitch_risk":      feats.get("dtw_pitch_risk"),
            "energy_risk":     feats.get("dtw_energy_risk"),
            "baseline_note":   "Pitch ~185±30 Hz sin-wave, Energy ~0.05±0.02 (deterministic)",
        },
        "disclaimer": (
            "Đây chỉ là công cụ sàng lọc sơ bộ, KHÔNG thay thế chẩn đoán y khoa. "
            "Nếu có lo ngại, hãy liên hệ chuyên gia tâm lý/bác sĩ."
        ),
    }
