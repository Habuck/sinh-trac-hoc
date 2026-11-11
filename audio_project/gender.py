"""Train & predict giới tính bằng KNN từ đặc trưng âm thanh.

Hỗ trợ 2 chế độ:
  1. predict_gender(features_dict)  — từ dict đặc trưng số (legacy)
  2. predict_gender_from_file(path) — từ file audio (trích xuất tự động)
"""
from dataclasses import dataclass
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from audio_project.core import MODELS_DIR, RAND

GENDER_FEATURE_COLUMNS = ["meanfreq", "sd", "centroid", "meanfun", "IQR", "median"]
GENDER_LABEL_COLUMN    = "label"
SR = 16000


@dataclass
class GenderTrainResult:
    model_path: Path
    report: str
    train_size: int
    test_size: int


# ── Data loading ────────────────────────────────────────────────────────────
def find_voice_csv(base_dir: Path | None = None) -> Path:
    search_root = base_dir or Path(__file__).resolve().parents[1]
    candidates  = sorted(search_root.rglob("voice.csv"))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy file voice.csv trong project.")
    return candidates[0]


# ── Training ────────────────────────────────────────────────────────────────
def train_gender_model(csv_path: Path | None = None,
                       model_path: Path | None = None) -> GenderTrainResult:
    df = pd.read_csv(csv_path or find_voice_csv())
    missing = [c for c in GENDER_FEATURE_COLUMNS + [GENDER_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"voice.csv thiếu cột: {missing}")
    x, y = df[GENDER_FEATURE_COLUMNS], df[GENDER_LABEL_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RAND,
        stratify=y if len(y.unique()) > 1 else None)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    report     = classification_report(y_test, model.predict(x_test), zero_division=0)
    model_path = model_path or (MODELS_DIR / "gender_classifier.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return GenderTrainResult(model_path=model_path, report=report,
                             train_size=len(x_train), test_size=len(x_test))


# ── Model loading ───────────────────────────────────────────────────────────
def load_gender_model(model_path: Path | None = None):
    return joblib.load(model_path or (MODELS_DIR / "gender_classifier.joblib"))


# ── Feature extraction from audio ───────────────────────────────────────────
def extract_gender_features_from_audio(audio_path: Path) -> dict[str, float]:
    """
    Trích xuất 6 đặc trưng voice.csv-compatible từ file audio.

    Mapping:
      meanfreq — tần số trung bình (kHz)
      sd       — độ lệch chuẩn tần số (kHz)
      centroid — spectral centroid trung bình (kHz)
      meanfun  — tần số cơ bản (fundamental) trung bình (kHz)
      IQR      — IQR của phổ tần số (kHz)
      median   — tần số trung vị (kHz)
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < SR * 0.2:
        yt = y

    # Spectral frequencies (via STFT magnitude)
    S     = np.abs(librosa.stft(yt, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048) / 1000  # kHz

    # Weighted mean, std, median, IQR của tần số
    mag_sum = S.sum(axis=0, keepdims=True) + 1e-10
    w       = S / mag_sum                                        # (F, T) normalized

    meanfreq = float(np.mean((freqs[:, None] * w).sum(axis=0)))
    # std per frame
    mean2    = float(np.mean(((freqs[:, None] ** 2) * w).sum(axis=0)))
    sd       = float(np.sqrt(max(mean2 - meanfreq ** 2, 0)))

    # spectral centroid (kHz)
    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=yt, sr=SR, n_fft=2048, hop_length=512)) / 1000)

    # Fundamental frequency / meanfun (kHz)
    f0, _, _ = librosa.pyin(yt, fmin=50, fmax=500, sr=SR)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]
    meanfun = float(np.mean(voiced) / 1000) if len(voiced) > 0 else 0.0

    # IQR của phổ tần số (cumulative distribution approach)
    cum = np.cumsum(freqs)
    cum = cum / (cum[-1] + 1e-10)
    q25_idx = int(np.searchsorted(cum, 0.25))
    q75_idx = int(np.searchsorted(cum, 0.75))
    med_idx  = int(np.searchsorted(cum, 0.50))
    IQR    = float(freqs[q75_idx] - freqs[q25_idx])
    median = float(freqs[med_idx])

    return {
        "meanfreq": round(meanfreq, 6),
        "sd":       round(sd, 6),
        "centroid": round(centroid, 6),
        "meanfun":  round(meanfun, 6),
        "IQR":      round(IQR, 6),
        "median":   round(median, 6),
    }


# ── Prediction from dict ─────────────────────────────────────────────────────
def predict_gender(features: dict[str, float], model=None) -> str:
    model = model or load_gender_model()
    frame = pd.DataFrame(
        [[features[c] for c in GENDER_FEATURE_COLUMNS]],
        columns=GENDER_FEATURE_COLUMNS,
    )
    return str(model.predict(frame)[0])


# ── Prediction from audio file ───────────────────────────────────────────────
def predict_gender_from_file(audio_path: Path, model=None) -> dict:
    """
    Dự đoán giới tính trực tiếp từ file audio — tự động trích xuất đặc trưng.
    Trả về: prediction (str), confidence (float), features (dict).
    """
    model    = model or load_gender_model()
    features = extract_gender_features_from_audio(audio_path)
    frame    = pd.DataFrame(
        [[features[c] for c in GENDER_FEATURE_COLUMNS]],
        columns=GENDER_FEATURE_COLUMNS,
    )
    pred  = str(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    classes = list(model.classes_)

    label_vi = {"male": "Nam", "female": "Nữ"}
    return {
        "prediction":    pred,
        "prediction_vi": label_vi.get(pred, pred),
        "confidence":    round(float(max(proba)), 4),
        "scores":        {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "scores_vi":     {label_vi.get(c, c): round(float(p), 4) for c, p in zip(classes, proba)},
        "features":      features,
    }
