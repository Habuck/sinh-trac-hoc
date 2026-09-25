"""Phân loại tiếng ho: ho khan (dry), ho có đờm (wet), không phải ho (none).

Features : MFCC, spectral centroid/bandwidth/rolloff, ZCR, RMS, duration.
Model    : RandomForest trained on extracted features (nếu có data).
Fallback : DTW prototype matching (frame-level RMS envelope).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import joblib, librosa, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

SR              = 16000
HOP_LENGTH      = 512           # frames: sr/hop ≈ 31 frames/sec
MODELS_DIR      = Path(__file__).resolve().parents[1] / "models"
COUGH_MODEL_PATH = MODELS_DIR / "cough_classifier.joblib"

LABEL_VI = {
    "dry_cough": "Ho khan",
    "wet_cough": "Ho có đờm",
    "not_cough": "Không phải ho",
}


# ── DTW (frame-level, Sakoe-Chiba band) ─────────────────────────────────
def _dtw(a: np.ndarray, b: np.ndarray, radius: int = 5) -> float:
    """
    Khoảng cách DTW chuẩn hóa giữa hai chuỗi 1-D (frame-level).
    Đầu vào ĐÃ normalize về [0,1] hoặc z-score.
    """
    a = a.reshape(-1, 1).astype(float)
    b = b.reshape(-1, 1).astype(float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / max(n, 1)
    for i in range(1, n + 1):
        jc = int(round(i * scale))
        j0 = max(1, jc - radius)
        j1 = min(m, jc + radius) + 1
        for j in range(j0, j1):
            cost = abs(a[i - 1, 0] - b[j - 1, 0])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


# ── Frame-level prototypes (N_FRAMES frames, normalized [0,1]) ───────────
_N = 50   # số frames prototype (đủ ngắn để DTW nhanh)


def _proto_dry() -> np.ndarray:
    """Ho khan: attack sắc ở frame đầu, giảm nhanh."""
    t = np.linspace(0, 1, _N)
    env = np.exp(-6 * t) * (1 - np.exp(-40 * t))
    return env / (env.max() + 1e-10)


def _proto_wet() -> np.ndarray:
    """Ho đờm: decay chậm hơn, secondary bump nhỏ ở giữa."""
    t = np.linspace(0, 1, _N)
    env = np.exp(-3 * t) * (1 - np.exp(-25 * t))
    env += 0.25 * np.exp(-30 * (t - 0.35) ** 2)
    return env / (env.max() + 1e-10)


def _proto_speech() -> np.ndarray:
    """Tiếng nói: energy dao động đều, không có peak sắc."""
    t = np.linspace(0, 2 * np.pi * 2, _N)
    env = 0.5 + 0.25 * np.sin(t)
    return env / (env.max() + 1e-10)


# Cache prototypes (tính 1 lần)
_PROTO = {
    "dry_cough": _proto_dry(),
    "wet_cough": _proto_wet(),
    "not_cough": _proto_speech(),
}


def _rms_envelope(y: np.ndarray, n_out: int = _N) -> np.ndarray:
    """Tính RMS envelope, resample về n_out frames và normalize."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    # Resample tuyến tính về n_out frames để match prototype
    idx = np.linspace(0, len(rms) - 1, n_out)
    rms_rs = np.interp(idx, np.arange(len(rms)), rms)
    peak = rms_rs.max()
    return rms_rs / (peak + 1e-10)


# ── DTW-based classifier ─────────────────────────────────────────────────
def _classify_by_dtw(y: np.ndarray, sr: int = SR) -> Dict:
    """Phân loại tiếng ho bằng DTW envelope so với prototype."""
    env = _rms_envelope(y, _N)

    dists = {k: _dtw(env, p) for k, p in _PROTO.items()}

    # Heuristic bổ sung
    yt, _ = librosa.effects.trim(y, top_db=20)
    dur      = len(yt) / sr
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(yt)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=yt)))

    # Đảo distance → score (softmax trên -dist)
    d_arr  = np.array([dists["dry_cough"], dists["wet_cough"], dists["not_cough"]])
    e_arr  = np.exp(-d_arr * 10)          # hệ số 10 để khuếch đại sự khác biệt
    s_arr  = e_arr / e_arr.sum()

    raw = {
        "dry_cough": float(s_arr[0]),
        "wet_cough": float(s_arr[1]),
        "not_cough": float(s_arr[2]),
    }

    # Điều chỉnh heuristic
    if dur < 1.0 and zcr > 0.08:
        raw["dry_cough"] *= 1.4
    if flatness > 0.04:
        raw["wet_cough"] *= 1.25

    tot    = sum(raw.values())
    scores = {k: round(v / tot, 4) for k, v in raw.items()}
    pred   = max(scores, key=scores.get)

    return {
        "prediction":    pred,
        "prediction_vi": LABEL_VI.get(pred, pred),
        "confidence":    scores[pred],
        "scores":        scores,
        "scores_vi":     {LABEL_VI.get(k, k): v for k, v in scores.items()},
        "method":        "dtw",
        "dtw_distances": {k: round(v, 4) for k, v in dists.items()},
    }


# ── Feature extraction (for ML model) ───────────────────────────────────
def _extract_cough_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """Extract features optimized for cough detection (MFCC + spectral + DTW)."""
    _m = lambda v: float(np.mean(v))
    _s = lambda v: float(np.std(v))

    yt, _ = librosa.effects.trim(y, top_db=20)
    if len(yt) < sr * 0.05:
        yt = y

    mfcc     = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=20)
    cent     = librosa.feature.spectral_centroid(y=yt, sr=sr)
    bw       = librosa.feature.spectral_bandwidth(y=yt, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=yt, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(yt)
    rms      = librosa.feature.rms(y=yt, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=yt)

    feats = {}
    for i in range(13):
        feats[f"mfcc_{i}_mean"] = _m(mfcc[i])
        feats[f"mfcc_{i}_std"]  = _s(mfcc[i])
    feats["centroid_mean"]  = _m(cent)
    feats["centroid_std"]   = _s(cent)
    feats["bandwidth_mean"] = _m(bw)
    feats["rolloff_mean"]   = _m(rolloff)
    feats["zcr_mean"]       = _m(zcr)
    feats["zcr_std"]        = _s(zcr)
    feats["rms_mean"]       = _m(rms)
    feats["rms_std"]        = _s(rms)
    feats["rms_max"]        = float(np.max(rms))
    feats["flatness_mean"]  = _m(flatness)
    feats["duration"]       = len(yt) / sr

    rms_frames = rms[0]
    if len(rms_frames) > 4:
        q = len(rms_frames) // 4
        feats["energy_attack"]  = float(np.mean(rms_frames[:q]))
        feats["energy_sustain"] = float(np.mean(rms_frames[q:]))
        feats["attack_ratio"]   = feats["energy_attack"] / (feats["energy_sustain"] + 1e-10)
    else:
        feats["energy_attack"]  = feats["rms_mean"]
        feats["energy_sustain"] = feats["rms_mean"]
        feats["attack_ratio"]   = 1.0

    # DTW distances (frame-level, nhanh)
    env = _rms_envelope(yt, _N)
    for k, p in _PROTO.items():
        feats[f"dtw_{k}"] = round(_dtw(env, p), 5)

    return feats


# ── Training ─────────────────────────────────────────────────────────────
@dataclass
class CoughTrainResult:
    model_path: Path
    report: str
    train_size: int
    test_size: int


def train_cough_model(data_dir: Path = None) -> CoughTrainResult:
    """Train cough classifier từ labeled audio folders."""
    data_dir = data_dir or (
        Path(__file__).resolve().parents[1] / "data" / "raw_medical" / "cough"
    )
    rows = []
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            try:
                y, _ = librosa.load(wav, sr=SR, mono=True)
                row = _extract_cough_features(y)
                row["label"] = label_dir.name
                rows.append(row)
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No cough data in {data_dir}. "
            "Need subfolders: dry_cough/, wet_cough/, not_cough/"
        )
    x = df.drop(columns=["label"])
    y_col = df["label"]
    xt, xe, yt_, ye = train_test_split(
        x, y_col, test_size=0.2, random_state=42,
        stratify=y_col if len(y_col.unique()) > 1 else None,
    )
    m = RandomForestClassifier(n_estimators=150, random_state=42)
    m.fit(xt, yt_)
    rpt = classification_report(ye, m.predict(xe), zero_division=0)
    COUGH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, COUGH_MODEL_PATH)
    return CoughTrainResult(
        model_path=COUGH_MODEL_PATH, report=rpt,
        train_size=len(xt), test_size=len(xe),
    )


def load_cough_model():
    return joblib.load(COUGH_MODEL_PATH)


# ── Inference ─────────────────────────────────────────────────────────────
def predict_cough(audio_path: Path) -> Dict:
    """Predict cough type — dùng ML model nếu có, fallback DTW."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    if not COUGH_MODEL_PATH.exists():
        return _classify_by_dtw(y)

    feats  = _extract_cough_features(y)
    model  = load_cough_model()
    df_f   = pd.DataFrame([feats])
    pred   = model.predict(df_f)[0]
    proba  = model.predict_proba(df_f)[0]
    classes = list(model.classes_)
    ml_scores = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    # Hybrid: 70% ML + 30% DTW
    dtw_res = _classify_by_dtw(y)
    hybrid  = {}
    for k in ml_scores:
        hybrid[k] = round(
            0.7 * ml_scores.get(k, 0.0) + 0.3 * dtw_res["scores"].get(k, 0.0), 4
        )
    tot        = sum(hybrid.values())
    hybrid     = {k: round(v / tot, 4) for k, v in hybrid.items()}
    pred_final = max(hybrid, key=hybrid.get)

    return {
        "prediction":    pred_final,
        "prediction_vi": LABEL_VI.get(pred_final, pred_final),
        "confidence":    hybrid[pred_final],
        "scores":        hybrid,
        "scores_vi":     {LABEL_VI.get(k, k): v for k, v in hybrid.items()},
        "method":        "hybrid_ml_dtw",
        "dtw_details":   dtw_res.get("dtw_distances", {}),
    }
