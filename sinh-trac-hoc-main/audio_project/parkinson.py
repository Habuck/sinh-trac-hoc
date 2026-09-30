"""
audio_project/parkinson.py
Parkinson's Disease Voice Screening Module — Đồ án 2.1

Pipeline:
    Audio Input
      → Preprocessing (mono, 16 kHz, normalize, trim silence)
      → Feature Extraction (MFCC, F0, Jitter, Shimmer, HNR, Spectral)
      → Machine Learning (LR / SVM / RF / GradientBoosting)
      → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity)
      → Inference (predict new audio)

Dataset hỗ trợ:
  1. UCI Parkinson's Voice Dataset (parkinsons.data — CSV features)
  2. Parkinson audio recordings (data/raw/parkinson/{parkinson,control}/)

Tham khảo:
  - Little MA et al. (2007) — "Exploiting Nonlinear Recurrence..."
  - Tsanas A et al. (2012) — "Novel speech signal processing algorithms..."
  - UCI ML Repository — Parkinson's Disease Data Set (195 samples)

Phần cứng mục tiêu: Microphone INMP441 (I2S MEMS) + Raspberry Pi 4
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Đường dẫn ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"
_RESULTS_DIR  = _PROJECT_ROOT / "results"
_FEATURES_DIR = _DATA_DIR / "features"

# UCI CSV (được copy vào data/ hoặc dùng trực tiếp từ các project khác)
UCI_CSV_PATH = _DATA_DIR / "parkinsons.data"

# Parkinson audio directories
RAW_PARKINSON_DIR  = _DATA_DIR / "raw"    / "parkinson"
PROC_PARKINSON_DIR = _DATA_DIR / "processed" / "parkinson"
FEATURES_CSV_PATH  = _FEATURES_DIR / "parkinson_features.csv"

# Model
PARKINSON_MODEL_PATH = _MODELS_DIR / "parkinson_classifier.joblib"

# ── Hằng số xử lý audio ──────────────────────────────────────────────────────
SR         = 16_000   # 16 kHz — INMP441 native
N_MFCC     = 13
HOP_LENGTH = 512
TOP_DB     = 25       # Trim silence threshold

# ── UCI feature columns ───────────────────────────────────────────────────────
UCI_FEATURE_COLS: List[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# ── Ánh xạ audio feature → UCI feature (cho inference từ audio thực) ──────────
_AUDIO_TO_UCI: Dict[str, str] = {
    "f0_mean"       : "MDVP:Fo(Hz)",
    "f0_max"        : "MDVP:Fhi(Hz)",
    "f0_min"        : "MDVP:Flo(Hz)",
    "jitter_local"  : "MDVP:Jitter(%)",
    "jitter_rap"    : "MDVP:RAP",
    "jitter_ppq5"   : "MDVP:PPQ",
    "jitter_ddp"    : "Jitter:DDP",
    "shimmer_local" : "MDVP:Shimmer",
    "shimmer_db"    : "MDVP:Shimmer(dB)",
    "shimmer_apq3"  : "Shimmer:APQ3",
    "shimmer_apq5"  : "Shimmer:APQ5",
    "nhr"           : "NHR",
    "hnr"           : "HNR",
    "ppe"           : "PPE",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — AUDIO PREPROCESSING
#  Tái sử dụng logic từ core.py, chuyên biệt cho Parkinson
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_audio_parkinson(
    path: str | Path,
    sr: int = SR,
    top_db: int = TOP_DB,
) -> Tuple[np.ndarray, int]:
    """
    Tiền xử lý audio cho phân tích Parkinson.

    Bước:
      1. Load → mono
      2. Resample → sr (mặc định 16 kHz)
      3. Normalize amplitude
      4. Trim silence

    Parameters
    ----------
    path    : đường dẫn file audio
    sr      : target sample rate
    top_db  : ngưỡng cắt silence (dB)

    Returns
    -------
    (y, sr) — waveform đã xử lý và sample rate
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)

    # Normalize amplitude
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Trim silence
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < sr * 0.3:   # nếu còn quá ngắn, dùng nguyên
        yt = y

    return yt, sr


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

# ── Jitter helpers ────────────────────────────────────────────────────────────

def _jitter_local(f0: np.ndarray) -> float:
    """Jitter (local) — biến thiên chu kỳ tương đối giữa các chu kỳ liền kề."""
    v = f0[f0 > 0]
    if len(v) < 3:
        return 0.0
    p = 1.0 / v
    return float(np.mean(np.abs(np.diff(p))) / (np.mean(p) + 1e-10))


def _jitter_rap(f0: np.ndarray) -> float:
    """Jitter (RAP) — Relative Average Perturbation (3-point smoothing)."""
    v = f0[f0 > 0]
    if len(v) < 5:
        return 0.0
    p = 1.0 / v
    s = np.convolve(p, [1/3, 1/3, 1/3], mode="valid")
    return float(np.mean(np.abs(p[1:-1] - s)) / (np.mean(p) + 1e-10))


def _jitter_ppq5(f0: np.ndarray) -> float:
    """Jitter (PPQ5) — 5-point Period Perturbation Quotient."""
    v = f0[f0 > 0]
    if len(v) < 7:
        return 0.0
    p = 1.0 / v
    s = np.convolve(p, np.ones(5) / 5, mode="valid")
    return float(np.mean(np.abs(p[2:-2] - s)) / (np.mean(p) + 1e-10))


def _jitter_ddp(f0: np.ndarray) -> float:
    """Jitter (DDP) — average absolute second difference of periods."""
    v = f0[f0 > 0]
    if len(v) < 4:
        return 0.0
    p = 1.0 / v
    return float(np.mean(np.abs(np.diff(np.diff(p)))) / (np.mean(p) + 1e-10))


# ── Shimmer helpers ───────────────────────────────────────────────────────────

def _frame_amps(y: np.ndarray, sr: int = SR) -> np.ndarray:
    fl = max(256, int(sr * 0.03))
    hl = max(64,  int(sr * 0.01))
    frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
    return np.max(np.abs(frames), axis=0)


def _shimmer_local(y: np.ndarray, sr: int = SR) -> float:
    a = _frame_amps(y, sr)
    if len(a) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(a))) / (np.mean(a) + 1e-10))


def _shimmer_db(y: np.ndarray, sr: int = SR) -> float:
    a = _frame_amps(y, sr)
    a = a[a > 0]
    if len(a) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(20 * np.log10(a + 1e-10)))))


def _shimmer_apq(y: np.ndarray, sr: int = SR, n: int = 3) -> float:
    a = _frame_amps(y, sr)
    if len(a) < n + 2:
        return 0.0
    s = np.convolve(a, np.ones(n) / n, mode="valid")
    offset = n // 2
    t = a[offset: offset + len(s)]
    return float(np.mean(np.abs(t - s)) / (np.mean(a) + 1e-10))


# ── HNR / NHR ─────────────────────────────────────────────────────────────────

def _hnr(y: np.ndarray, sr: int = SR) -> float:
    ac = librosa.autocorrelate(y, max_size=sr // 60)
    if len(ac) < 2 or ac[0] == 0:
        return 0.0
    peak  = np.max(ac[1:])
    ratio = np.clip(peak / ac[0], 1e-10, 1 - 1e-10)
    return float(10 * np.log10(ratio / (1 - ratio)))


def _nhr(y: np.ndarray, sr: int = SR) -> float:
    h = _hnr(y, sr)
    return float(10 ** (-h / 10)) if h != 0 else 0.0


# ── PPE ───────────────────────────────────────────────────────────────────────

def _ppe(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    if len(v) < 10:
        return 0.0
    semi = 12 * np.log2(v / (np.median(v) + 1e-10))
    n_bins = max(3, min(20, len(semi) // 3))
    hist, _ = np.histogram(semi, bins=n_bins, density=True)
    w = (semi.max() - semi.min() + 1e-10) / n_bins
    probs = (hist * w)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ── MAIN FEATURE EXTRACTION ───────────────────────────────────────────────────

def extract_parkinson_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """
    Trích xuất toàn bộ đặc trưng giọng nói cho phân tích Parkinson.

    Nhóm đặc trưng (theo spec mục 8):
      - MFCC (13 × mean + std = 26)
      - Fundamental Frequency F0 (mean, std, min, max, range)
      - Voice quality: Jitter, Shimmer, HNR
      - Spectral: Centroid, Bandwidth, Rolloff, ZCR, RMS Energy
      - Pitch-related: mean, variance, range
      - PPE (Pitch Period Entropy)

    Returns
    -------
    dict tên đặc trưng → giá trị float
    """
    feats: Dict[str, float] = {}

    # ── MFCC ─────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    # ── Fundamental Frequency (F0) via pYIN ──────────────────────────────
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced = f0[f0 > 0]

    feats["f0_mean"]  = float(np.mean(voiced))  if len(voiced) else 0.0
    feats["f0_std"]   = float(np.std(voiced))   if len(voiced) else 0.0
    feats["f0_min"]   = float(np.min(voiced))   if len(voiced) else 0.0
    feats["f0_max"]   = float(np.max(voiced))   if len(voiced) else 0.0
    feats["f0_range"] = float(np.ptp(voiced))   if len(voiced) else 0.0

    # ── Voice quality: Jitter ────────────────────────────────────────────
    feats["jitter_local"] = _jitter_local(f0)
    feats["jitter_rap"]   = _jitter_rap(f0)
    feats["jitter_ppq5"]  = _jitter_ppq5(f0)
    feats["jitter_ddp"]   = _jitter_ddp(f0)

    # ── Voice quality: Shimmer ────────────────────────────────────────────
    feats["shimmer_local"] = _shimmer_local(y, sr)
    feats["shimmer_db"]    = _shimmer_db(y, sr)
    feats["shimmer_apq3"]  = _shimmer_apq(y, sr, 3)
    feats["shimmer_apq5"]  = _shimmer_apq(y, sr, 5)

    # ── HNR / NHR ────────────────────────────────────────────────────────
    feats["hnr"] = _hnr(y, sr)
    feats["nhr"] = _nhr(y, sr)

    # ── Spectral features ─────────────────────────────────────────────────
    feats["spectral_centroid"]   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_bandwidth"]  = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats["spectral_rolloff"]    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats["zcr_mean"]            = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["rms_energy"]          = float(np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)))

    # ── Pitch-related (từ đặc trưng F0 đã tính) ──────────────────────────
    feats["pitch_mean"]     = feats["f0_mean"]
    feats["pitch_variance"] = float(np.var(voiced)) if len(voiced) else 0.0
    feats["pitch_range"]    = feats["f0_range"]

    # ── PPE ──────────────────────────────────────────────────────────────
    feats["ppe"] = _ppe(f0)

    # ── Metadata ─────────────────────────────────────────────────────────
    feats["duration_sec"] = round(len(y) / sr, 3)

    return feats


def extract_features_from_file(
    path: str | Path,
    sr: int = SR,
) -> Dict[str, float]:
    """Load, preprocess và trích xuất đặc trưng từ file audio."""
    y, sr_out = preprocess_audio_parkinson(path, sr=sr)
    return extract_parkinson_features(y, sr_out)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4-5 — TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    sensitivity: float
    specificity: float
    train_size: int
    test_size: int
    model_path: str
    report: str
    confusion_matrix: List[List[int]]
    best_params: Dict = field(default_factory=dict)


def _build_models() -> Dict:
    """Tạo dict các model ML cần huấn luyện và so sánh."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=42, C=1.0,
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=4, random_state=42,
        ),
    }


def _evaluate(y_true, y_pred, y_proba, model_name: str) -> Dict:
    """Tính toàn bộ metric theo spec mục 11."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-10)   # Recall for positive class
    specificity = tn / (tn + fp + 1e-10)   # True Negative Rate

    return {
        "model"          : model_name,
        "accuracy"       : float(accuracy_score(y_true, y_pred)),
        "precision"      : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"         : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"             : float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc"        : float(roc_auc_score(y_true, y_proba)),
        "sensitivity"    : float(sensitivity),
        "specificity"    : float(specificity),
        "confusion_matrix": cm.tolist(),
        "report"         : classification_report(
            y_true, y_pred,
            target_names=["Control", "Parkinson"],
            zero_division=0,
        ),
    }


def _save_plots(
    y_true, y_proba_best,
    cm_data,
    best_model_name: str,
    out_dir: Path,
) -> None:
    """Lưu Confusion Matrix và ROC Curve (spec mục 12)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Confusion Matrix ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_arr = np.array(cm_data)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks([0, 1]);  ax.set_xticklabels(["Control", "Parkinson"])
        ax.set_yticks([0, 1]);  ax.set_yticklabels(["Control", "Parkinson"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {best_model_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        # ── ROC Curve ─────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_true, y_proba_best)
        roc_auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#6366f1", lw=2,
                label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {best_model_name}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

    except Exception:
        pass   # matplotlib optional


def _load_uci_dataset() -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Tải UCI Parkinson's Dataset (CSV dạng features).

    Tìm kiếm theo thứ tự:
      1. data/parkinsons.data (UCI canonical)
      2. audio_project/voice.csv (dataset hiện có trong Đồ án 2.1)
    """
    candidates = [
        UCI_CSV_PATH,
        _DATA_DIR / "raw_medical" / "parkinsons.data",
        _PROJECT_ROOT / "audio_project" / "voice.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # UCI canonical format
            if "status" in df.columns and all(c in df.columns for c in UCI_FEATURE_COLS):
                X = df[UCI_FEATURE_COLS].copy()
                y = df["status"].astype(int)
                return X, y
    return None


def _load_audio_dataset() -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Tải features từ audio recordings trong data/raw/parkinson/.

    Returns (X, y, groups) hoặc None nếu không có data.
    groups: subject ID (để dùng GroupKFold — tránh data leakage)
    """
    parkinson_dir = RAW_PARKINSON_DIR
    if not parkinson_dir.exists():
        return None

    audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    rows = []

    for label_dir in ["parkinson", "control"]:
        class_dir = parkinson_dir / label_dir
        if not class_dir.exists():
            continue
        label = 1 if label_dir == "parkinson" else 0
        for audio_file in sorted(class_dir.rglob("*")):
            if audio_file.suffix.lower() not in audio_ext:
                continue
            try:
                feats = extract_features_from_file(audio_file)
                feats["label"]   = label
                feats["subject"] = audio_file.stem.split("_")[0]  # subject ID from filename
                rows.append(feats)
            except Exception:
                continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    X  = df.drop(columns=["label", "subject"])
    y  = df["label"]
    groups = df["subject"]
    return X, y, groups


def train_parkinson_model(verbose: bool = True) -> TrainResult:
    """
    Huấn luyện và so sánh nhiều model ML trên Parkinson dataset.

    Ưu tiên:
      1. UCI CSV dataset (features đã có)
      2. Audio recordings (trích xuất features)

    Anti-leakage: dùng GroupKFold nếu có subject ID.
    Lưu model tốt nhất (theo ROC-AUC) vào models/parkinson_classifier.joblib.
    Lưu bảng so sánh vào results/parkinson_model_comparison.csv.

    Returns TrainResult của model tốt nhất.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tải dataset ──────────────────────────────────────────────────────
    groups = None
    uci = _load_uci_dataset()

    if uci is not None:
        X, y = uci
        if verbose:
            print(f"📂 UCI Parkinson's Dataset: {len(y)} mẫu, "
                  f"{y.sum()} Parkinson, {(y == 0).sum()} Control")
        # Không có subject info trong UCI canonical CSV → random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
    else:
        audio_data = _load_audio_dataset()
        if audio_data is None:
            raise FileNotFoundError(
                "Không tìm thấy dataset Parkinson.\n"
                "Hãy đặt file 'parkinsons.data' vào data/ hoặc\n"
                "audio recordings vào data/raw/parkinson/{parkinson,control}/."
            )
        X, y, groups = audio_data
        if verbose:
            print(f"📂 Audio Dataset: {len(y)} mẫu")
        # Anti-leakage: GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if verbose:
        print(f"   Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Pipeline: Imputer → Scaler → Classifier ───────────────────────────
    all_metrics = []
    best_result = None
    best_auc    = -1.0
    best_pipeline = None

    for name, clf in _build_models().items():
        if verbose:
            print(f"\n🔄 Training {name}...")
        try:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("clf",     clf),
            ])
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            metrics = _evaluate(y_test, y_pred, y_proba, name)
            all_metrics.append(metrics)

            if verbose:
                print(f"   ✅ Accuracy={metrics['accuracy']:.4f} "
                      f"AUC={metrics['roc_auc']:.4f} "
                      f"Sensitivity={metrics['sensitivity']:.4f} "
                      f"Specificity={metrics['specificity']:.4f}")

            if metrics["roc_auc"] > best_auc:
                best_auc      = metrics["roc_auc"]
                best_result   = metrics
                best_pipeline = pipe

        except Exception as e:
            if verbose:
                print(f"   ❌ {name}: {e}")

    if best_pipeline is None:
        raise RuntimeError("Tất cả model đều thất bại khi huấn luyện.")

    # ── Lưu bảng so sánh (spec mục 12) ───────────────────────────────────
    comp_rows = []
    for m in all_metrics:
        comp_rows.append({
            "Model"      : m["model"],
            "Accuracy"   : round(m["accuracy"],    4),
            "Precision"  : round(m["precision"],   4),
            "Recall"     : round(m["recall"],      4),
            "F1"         : round(m["f1"],          4),
            "ROC-AUC"    : round(m["roc_auc"],     4),
            "Sensitivity": round(m["sensitivity"], 4),
            "Specificity": round(m["specificity"], 4),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = _RESULTS_DIR / "parkinson_model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    if verbose:
        print(f"\n📊 Bảng so sánh:\n{comp_df.to_string(index=False)}")
        print(f"\n💾 Lưu bảng so sánh: {comp_path}")

    # ── Lưu plots ─────────────────────────────────────────────────────────
    best_proba = best_pipeline.predict_proba(X_test)[:, 1]
    _save_plots(y_test, best_proba,
                best_result["confusion_matrix"],
                best_result["model"],
                _RESULTS_DIR)

    # ── Lưu model tốt nhất (joblib) ───────────────────────────────────────
    bundle = {
        "pipeline"     : best_pipeline,
        "model_name"   : best_result["model"],
        "feature_names": list(X.columns),
        "metrics"      : best_result,
        "comparison"   : comp_rows,
        "dataset_type" : "uci" if uci is not None else "audio",
    }
    joblib.dump(bundle, PARKINSON_MODEL_PATH)
    if verbose:
        print(f"\n✅ Best model: {best_result['model']} (AUC={best_auc:.4f})")
        print(f"💾 Lưu model: {PARKINSON_MODEL_PATH}")

    return TrainResult(
        model_name       = best_result["model"],
        accuracy         = best_result["accuracy"],
        precision        = best_result["precision"],
        recall           = best_result["recall"],
        f1               = best_result["f1"],
        roc_auc          = best_result["roc_auc"],
        sensitivity      = best_result["sensitivity"],
        specificity      = best_result["specificity"],
        train_size       = len(y_train),
        test_size        = len(y_test),
        model_path       = str(PARKINSON_MODEL_PATH),
        report           = best_result["report"],
        confusion_matrix = best_result["confusion_matrix"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _load_bundle():
    """Tải model bundle từ disk."""
    if not PARKINSON_MODEL_PATH.exists():
        raise FileNotFoundError("Parkinson model is not trained yet.")
    return joblib.load(PARKINSON_MODEL_PATH)


def is_model_trained() -> bool:
    """Kiểm tra model đã được huấn luyện chưa."""
    return PARKINSON_MODEL_PATH.exists()


def predict_parkinson(audio_path: str | Path) -> Dict:
    """
    Suy luận Parkinson từ file audio thực.

    Trích xuất đặc trưng từ audio → ánh xạ sang feature space của model
    → predict → trả về kết quả.

    Returns
    -------
    dict: prediction, probability, model, features (tóm tắt)
    """
    bundle   = _load_bundle()
    pipeline = bundle["pipeline"]
    feat_names = bundle["feature_names"]
    dataset_type = bundle.get("dataset_type", "uci")
    model_name = bundle["model_name"]

    # Trích xuất đặc trưng audio
    audio_feats = extract_features_from_file(audio_path)

    # Xây dựng input vector phù hợp với feature_names của pipeline
    if dataset_type == "uci":
        # Ánh xạ audio features → UCI feature space
        row: Dict[str, float] = {}
        for audio_key, uci_key in _AUDIO_TO_UCI.items():
            row[uci_key] = audio_feats.get(audio_key, 0.0)
        for col in feat_names:
            if col not in row:
                row[col] = 0.0
        X_input = pd.DataFrame([{col: row[col] for col in feat_names}])
    else:
        # Audio dataset mode: dùng trực tiếp extracted features
        X_input = pd.DataFrame([
            {col: audio_feats.get(col, 0.0) for col in feat_names}
        ])

    pred  = int(pipeline.predict(X_input)[0])
    proba = float(pipeline.predict_proba(X_input)[0][1])

    label = "Parkinson" if pred == 1 else "Control"

    return {
        "prediction"  : label,
        "probability" : round(proba, 4),
        "model"       : model_name,
        "risk_pct"    : round(proba * 100, 2),
        "features_summary": {
            "f0_mean"      : round(audio_feats.get("f0_mean", 0.0), 2),
            "jitter_local" : round(audio_feats.get("jitter_local", 0.0), 6),
            "shimmer_local": round(audio_feats.get("shimmer_local", 0.0), 6),
            "hnr"          : round(audio_feats.get("hnr", 0.0), 2),
            "nhr"          : round(audio_feats.get("nhr", 0.0), 6),
            "ppe"          : round(audio_feats.get("ppe", 0.0), 4),
            "duration_sec" : round(audio_feats.get("duration_sec", 0.0), 2),
        },
        "hardware_info": {
            "microphone": "INMP441 (I2S MEMS) — simulated via computer mic",
            "processor" : "Raspberry Pi 4 — simulated on server",
            "sample_rate": f"{SR} Hz",
        },
        "disclaimer": (
            "This system is intended for research and preliminary screening purposes only. "
            "It is not a medical diagnostic tool and should not replace professional medical evaluation."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPAT — screen_parkinson() (rule-based, kept for compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def screen_parkinson(audio_path) -> Dict:
    """
    Điểm vào chính cho Flask API /api/parkinson/screen.

    Nếu model ML đã train → dùng ML (predict_parkinson).
    Nếu chưa train → fallback về rule-based screening cũ.
    """
    if is_model_trained():
        try:
            result = predict_parkinson(audio_path)
            # Gộp thêm format cũ để tương thích với frontend hiện tại
            risk_pct = result["risk_pct"]
            if risk_pct >= 60:
                level_vi = "⚠️ Nguy cơ cao"
                advice = ("Phát hiện nhiều dấu hiệu bất thường. "
                          "Khuyến nghị tham khảo bác sĩ chuyên khoa thần kinh.")
            elif risk_pct >= 30:
                level_vi = "🔶 Nguy cơ trung bình"
                advice = ("Một số đặc trưng giọng nói bất thường nhẹ. "
                          "Nên theo dõi định kỳ và tham vấn bác sĩ.")
            else:
                level_vi = "✅ Nguy cơ thấp"
                advice = ("Giọng nói trong phạm vi bình thường. "
                          "Không phát hiện dấu hiệu Parkinson rõ rệt.")
            return {
                **result,
                "risk_score"   : risk_pct,
                "risk_level_vi": level_vi,
                "advice"       : advice,
                "features"     : result["features_summary"],
            }
        except Exception:
            pass   # fallback về rule-based

    # ── Rule-based fallback ───────────────────────────────────────────────
    return _rule_based_screen(audio_path)


def _rule_based_screen(audio_path) -> Dict:
    """Rule-based Parkinson screening (fallback khi chưa có ML model)."""
    from pathlib import Path as _Path
    y, sr = preprocess_audio_parkinson(audio_path)
    feats = extract_parkinson_features(y, sr)

    indicators = []
    risk_score = 0.0

    def _add(feature, detail, severity, weight):
        indicators.append({"feature": feature, "detail": detail,
                            "severity": severity, "weight": weight})
        return weight

    # Jitter
    if feats["jitter_local"] > 0.020:
        risk_score += _add("Jitter cao bất thường",
                           f"Jitter = {feats['jitter_local']:.4f} (ngưỡng < 0.010)",
                           "cao", 0.15)
    elif feats["jitter_local"] > 0.012:
        risk_score += _add("Jitter hơi cao",
                           f"Jitter = {feats['jitter_local']:.4f}",
                           "trung bình", 0.08)

    # Shimmer
    if feats["shimmer_local"] > 0.15:
        risk_score += _add("Shimmer cao",
                           f"Shimmer = {feats['shimmer_local']:.4f} (ngưỡng < 0.10)",
                           "cao", 0.15)
    elif feats["shimmer_local"] > 0.10:
        risk_score += _add("Shimmer hơi cao",
                           f"Shimmer = {feats['shimmer_local']:.4f}",
                           "trung bình", 0.08)

    # HNR
    if feats["hnr"] < 10:
        risk_score += _add("HNR rất thấp — giọng khàn/thở",
                           f"HNR = {feats['hnr']:.1f} dB (bình thường > 20 dB)",
                           "cao", 0.15)
    elif feats["hnr"] < 20:
        risk_score += _add("HNR thấp",
                           f"HNR = {feats['hnr']:.1f} dB",
                           "trung bình", 0.08)

    # F0 range (giảm trong PD → monotone)
    if feats["f0_mean"] > 0:
        if feats["f0_range"] < 30:
            risk_score += _add("Dải tần giọng rất hẹp (monotone)",
                               f"F0 range = {feats['f0_range']:.1f} Hz",
                               "cao", 0.12)
        elif feats["f0_range"] < 60:
            risk_score += _add("Dải tần giọng hơi hẹp",
                               f"F0 range = {feats['f0_range']:.1f} Hz",
                               "trung bình", 0.06)

    # PPE
    if feats["ppe"] > 0.5:
        risk_score += _add("Entropy pitch cao (PPE)",
                           f"PPE = {feats['ppe']:.3f}",
                           "cao", 0.12)
    elif feats["ppe"] > 0.3:
        risk_score += _add("PPE hơi cao",
                           f"PPE = {feats['ppe']:.3f}",
                           "trung bình", 0.06)

    # NHR
    if feats["nhr"] > 0.10:
        risk_score += _add("Tỷ lệ nhiễu cao (NHR)",
                           f"NHR = {feats['nhr']:.4f}",
                           "trung bình", 0.08)

    risk_score = min(risk_score, 1.0)
    risk_pct   = round(risk_score * 100, 1)

    if risk_pct >= 60:
        level_vi = "⚠️ Nguy cơ cao"
        advice   = ("Phát hiện nhiều dấu hiệu bất thường trong giọng nói. "
                    "Khuyến nghị tham khảo bác sĩ chuyên khoa thần kinh.")
        pred     = "Parkinson"
    elif risk_pct >= 30:
        level_vi = "🔶 Nguy cơ trung bình"
        advice   = ("Một số đặc trưng giọng nói bất thường nhẹ. "
                    "Nên theo dõi định kỳ và tham vấn bác sĩ.")
        pred     = "Control"
    else:
        level_vi = "✅ Nguy cơ thấp"
        advice   = ("Giọng nói trong phạm vi bình thường. "
                    "Không phát hiện dấu hiệu Parkinson rõ rệt.")
        pred     = "Control"

    return {
        "prediction"    : pred,
        "probability"   : round(risk_score, 4),
        "risk_score"    : risk_pct,
        "risk_level_vi" : level_vi,
        "advice"        : advice,
        "indicators"    : indicators,
        "model"         : "Rule-based (chưa train ML)",
        "features"      : {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in feats.items()},
        "features_summary": {
            "f0_mean"      : round(feats.get("f0_mean", 0.0), 2),
            "jitter_local" : round(feats.get("jitter_local", 0.0), 6),
            "shimmer_local": round(feats.get("shimmer_local", 0.0), 6),
            "hnr"          : round(feats.get("hnr", 0.0), 2),
            "nhr"          : round(feats.get("nhr", 0.0), 6),
            "ppe"          : round(feats.get("ppe", 0.0), 4),
            "duration_sec" : round(feats.get("duration_sec", 0.0), 2),
        },
        "hardware_info": {
            "microphone" : "INMP441 (I2S MEMS) — simulated via computer mic",
            "processor"  : "Raspberry Pi 4 — simulated on server",
            "sample_rate": f"{SR} Hz",
        },
        "disclaimer": (
            "This system is intended for research and preliminary screening purposes only. "
            "It is not a medical diagnostic tool and should not replace professional medical evaluation. "
            "Hệ thống chỉ phục vụ mục đích nghiên cứu và sàng lọc ban đầu."
        ),
    }
