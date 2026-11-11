"""
audio_project/parkinson.py
Parkinson's Disease Voice Screening Module — Đồ án 2.1

Pipeline:
    Audio Input
      → Preprocessing (mono, 16 kHz, normalize, trim silence)
      → Feature Extraction
          · Praat (parselmouth): F0, Jitter, Shimmer, HNR, NHR  — cùng định nghĩa MDVP của UCI
          · Librosa: MFCC, Spectral, ZCR, RMS
      → Machine Learning (LR / SVM / RF / GradientBoosting, GridSearchCV theo nhóm bệnh nhân)
      → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity)
      → Inference (predict new audio)

Dataset hỗ trợ:
  1. UCI Parkinson's Voice Dataset (data/parkinsons.data — CSV features, 32 subjects)
  2. Parkinson audio recordings (data/raw/parkinson/{parkinson,control}/<subject>_<n>.wav)

Chống data leakage: mọi bản ghi của cùng một bệnh nhân nằm trọn trong train hoặc test
(StratifiedGroupKFold), và model được chọn bằng CV theo nhóm trên tập train — không nhìn tập test.

Tham khảo:
  - Little MA et al. (2009) — "Suitability of dysphonia measurements for telemonitoring
    of Parkinson's disease", IEEE TBME (UCI dataset)
  - Boersma P. & Weenink D. — Praat: doing phonetics by computer

Phần cứng mục tiêu: Microphone INMP441 (I2S MEMS) + Raspberry Pi 4
"""

from __future__ import annotations

import logging
import math
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

log = logging.getLogger(__name__)

# ── Đường dẫn ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"
_RESULTS_DIR  = _PROJECT_ROOT / "results"
_FEATURES_DIR = _DATA_DIR / "features"

UCI_CSV_PATH = _DATA_DIR / "parkinsons.data"

RAW_PARKINSON_DIR  = _DATA_DIR / "raw"    / "parkinson"
PROC_PARKINSON_DIR = _DATA_DIR / "processed" / "parkinson"
FEATURES_CSV_PATH  = _FEATURES_DIR / "parkinson_features.csv"

PARKINSON_MODEL_PATH = _MODELS_DIR / "parkinson_classifier.joblib"
COMPARISON_CSV_PATH  = _RESULTS_DIR / "parkinson_model_comparison.csv"

# ── Hằng số xử lý audio ──────────────────────────────────────────────────────
SR         = 16_000   # 16 kHz — INMP441 native
N_MFCC     = 13
HOP_LENGTH = 512
TOP_DB     = 25       # Trim silence threshold
F0_MIN     = 75.0     # Praat pitch floor (Hz)
F0_MAX     = 500.0    # Praat pitch ceiling (Hz)
RANDOM_STATE = 42

# Praat jitter/shimmer arguments (giá trị mặc định của Praat Voice Report)
_PERIOD_FLOOR, _PERIOD_CEIL, _MAX_PERIOD_FACTOR, _MAX_AMP_FACTOR = 0.0001, 0.02, 1.3, 1.6

# ── UCI feature columns (toàn bộ 22 thuộc tính của dataset) ──────────────────
UCI_FEATURE_COLS: List[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# Audio feature → UCI column. Model UCI chỉ được train trên các cột tính được từ audio
# (RPDE, DFA, spread1/2, D2, PPE là độ đo phi tuyến không tái tạo được nên bị loại),
# để lúc suy luận không phải điền giá trị giả cho cột thiếu.
_AUDIO_TO_UCI: Dict[str, str] = {
    "f0_mean"       : "MDVP:Fo(Hz)",
    "f0_max"        : "MDVP:Fhi(Hz)",
    "f0_min"        : "MDVP:Flo(Hz)",
    "jitter_local"  : "MDVP:Jitter(%)",
    "jitter_abs"    : "MDVP:Jitter(Abs)",
    "jitter_rap"    : "MDVP:RAP",
    "jitter_ppq5"   : "MDVP:PPQ",
    "jitter_ddp"    : "Jitter:DDP",
    "shimmer_local" : "MDVP:Shimmer",
    "shimmer_db"    : "MDVP:Shimmer(dB)",
    "shimmer_apq3"  : "Shimmer:APQ3",
    "shimmer_apq5"  : "Shimmer:APQ5",
    "shimmer_apq11" : "MDVP:APQ",
    "shimmer_dda"   : "Shimmer:DDA",
    "nhr"           : "NHR",
    "hnr"           : "HNR",
}
UCI_MODEL_COLS: List[str] = list(_AUDIO_TO_UCI.values())

# Khóa metadata — không dùng làm đầu vào model
_META_KEYS = {"duration_sec", "voiced_ratio"}

DISCLAIMER_EN = (
    "This system is intended for research and preliminary screening purposes only. "
    "It is not a medical diagnostic tool and should not replace professional medical evaluation."
)
DISCLAIMER_VI = (
    "Hệ thống chỉ phục vụ mục đích nghiên cứu và sàng lọc ban đầu, không phải công cụ chẩn đoán "
    "y khoa và không thay thế đánh giá của chuyên gia y tế."
)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — AUDIO PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_audio_parkinson(
    path: str | Path,
    sr: int = SR,
    top_db: int = TOP_DB,
) -> Tuple[np.ndarray, int]:
    """
    Tiền xử lý audio cho phân tích Parkinson.

    Bước: load → mono → resample `sr` → peak-normalize → trim silence.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)

    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < sr * 0.3:   # nếu còn quá ngắn, dùng nguyên
        yt = y

    return yt, sr


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _nan_if_undefined(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _praat_voice_measures(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Đo F0 / Jitter / Shimmer / HNR / NHR bằng Praat (parselmouth).

    Đây là cùng họ thuật toán (chu kỳ-đến-chu kỳ) mà dataset UCI dùng, nên giá trị
    tương thích trực tiếp với các cột MDVP. Giá trị không xác định trả về NaN
    (SimpleImputer trong pipeline sẽ xử lý).
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)

    pitch = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    m: Dict[str, float] = {
        "f0_mean" : float(np.mean(voiced)) if len(voiced) else float("nan"),
        "f0_std"  : float(np.std(voiced))  if len(voiced) else float("nan"),
        "f0_min"  : float(np.min(voiced))  if len(voiced) else float("nan"),
        "f0_max"  : float(np.max(voiced))  if len(voiced) else float("nan"),
        "voiced_ratio": float(len(voiced) / len(f0)) if len(f0) else 0.0,
    }
    m["f0_range"] = m["f0_max"] - m["f0_min"]

    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    jit_args = (0, 0, _PERIOD_FLOOR, _PERIOD_CEIL, _MAX_PERIOD_FACTOR)
    shim_args = jit_args + (_MAX_AMP_FACTOR,)
    for key, cmd in [
        ("jitter_local", "Get jitter (local)"),
        ("jitter_abs",   "Get jitter (local, absolute)"),
        ("jitter_rap",   "Get jitter (rap)"),
        ("jitter_ppq5",  "Get jitter (ppq5)"),
        ("jitter_ddp",   "Get jitter (ddp)"),
    ]:
        m[key] = _nan_if_undefined(call(pp, cmd, *jit_args))
    for key, cmd in [
        ("shimmer_local", "Get shimmer (local)"),
        ("shimmer_db",    "Get shimmer (local_dB)"),
        ("shimmer_apq3",  "Get shimmer (apq3)"),
        ("shimmer_apq5",  "Get shimmer (apq5)"),
        ("shimmer_apq11", "Get shimmer (apq11)"),
        ("shimmer_dda",   "Get shimmer (dda)"),
    ]:
        m[key] = _nan_if_undefined(call([snd, pp], cmd, *shim_args))

    # Praat đánh dấu frame vô thanh bằng -200 dB
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=F0_MIN)
    hv = harm.values[0]
    hv = hv[hv > -200]
    m["hnr"] = float(np.mean(hv)) if len(hv) else float("nan")
    m["nhr"] = float(np.mean(10.0 ** (-hv / 10.0))) if len(hv) else float("nan")
    return m


def extract_parkinson_features(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """
    Trích xuất đặc trưng giọng nói cho phân tích Parkinson (spec mục 8).

      - F0 (mean, std, min, max, range) + pitch mean/variance/range   [Praat]
      - Jitter: local, absolute, RAP, PPQ5, DDP                        [Praat]
      - Shimmer: local, dB, APQ3, APQ5, APQ11, DDA                     [Praat]
      - HNR, NHR                                                       [Praat]
      - MFCC 1–13 (mean + std)                                         [Librosa]
      - Spectral Centroid / Bandwidth / Rolloff, ZCR, RMS Energy       [Librosa]

    Giá trị không đo được (vd. audio không có đoạn hữu thanh) là NaN.
    """
    feats: Dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    feats.update(_praat_voice_measures(y, sr))

    feats["spectral_centroid"]  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats["spectral_rolloff"]   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats["zcr_mean"]           = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["rms_energy"]         = float(np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)))

    feats["pitch_mean"]     = feats["f0_mean"]
    feats["pitch_variance"] = feats["f0_std"] ** 2
    feats["pitch_range"]    = feats["f0_range"]

    feats["duration_sec"] = round(len(y) / sr, 3)
    return feats


def extract_features_from_file(path: str | Path, sr: int = SR) -> Dict[str, float]:
    """Load, preprocess và trích xuất đặc trưng từ file audio."""
    y, sr_out = preprocess_audio_parkinson(path, sr=sr)
    return extract_parkinson_features(y, sr_out)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def uci_subject_ids(names: pd.Series) -> pd.Series:
    """'phon_R01_S01_1' → 'S01' (mỗi subject có ~6 bản ghi)."""
    subj = names.astype(str).str.extract(r"_(S\d+)_")[0]
    if subj.isna().any():
        bad = names[subj.isna()].tolist()[:3]
        raise ValueError(f"Không đọc được subject ID từ cột 'name': {bad}")
    return subj


def _load_uci_dataset() -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Tải UCI Parkinson's Dataset → (X, y, groups).

    X chỉ gồm UCI_MODEL_COLS (các cột tái tạo được từ audio); groups = subject ID.
    """
    for path in [UCI_CSV_PATH, _DATA_DIR / "raw_medical" / "parkinsons.data"]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"status", "name"} <= set(df.columns) and all(c in df.columns for c in UCI_MODEL_COLS):
            return (
                df[UCI_MODEL_COLS].astype(float),
                df["status"].astype(int),
                uci_subject_ids(df["name"]),
            )
    return None


def _load_audio_dataset() -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """
    Trích xuất features từ data/raw/parkinson/{parkinson,control}/.

    Tên file phải có dạng `<subjectID>_<anything>.wav` để gom nhóm theo bệnh nhân.
    Kết quả được lưu ra data/features/parkinson_features.csv.
    """
    if not RAW_PARKINSON_DIR.exists():
        return None

    audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    rows = []
    for label_dir, label in [("parkinson", 1), ("control", 0)]:
        class_dir = RAW_PARKINSON_DIR / label_dir
        if not class_dir.exists():
            continue
        for audio_file in sorted(class_dir.rglob("*")):
            if audio_file.suffix.lower() not in audio_ext:
                continue
            try:
                feats = extract_features_from_file(audio_file)
            except Exception as e:
                log.warning("Bỏ qua %s: %s", audio_file, e)
                continue
            feats["label"]   = label
            feats["subject"] = f"{label_dir}:{audio_file.stem.split('_')[0]}"
            feats["file"]    = str(audio_file.relative_to(RAW_PARKINSON_DIR))
            rows.append(feats)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV_PATH, index=False)

    feature_cols = [c for c in df.columns if c not in {"label", "subject", "file"} | _META_KEYS]
    return df[feature_cols].astype(float), df["label"].astype(int), df["subject"]


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
    cv_roc_auc: float
    train_size: int
    test_size: int
    train_subjects: int
    test_subjects: int
    model_path: str
    report: str
    confusion_matrix: List[List[int]]
    best_params: Dict = field(default_factory=dict)
    nested_cv: Dict = field(default_factory=dict)
    comparison: List[Dict] = field(default_factory=list)


def _build_models() -> Dict[str, Tuple[object, Dict[str, list]]]:
    """Model + lưới siêu tham số cho GridSearchCV."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__max_depth": [None, 4, 8], "clf__min_samples_leaf": [1, 3]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
             "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def _evaluate(y_true, y_pred, y_proba, model_name: str) -> Dict:
    """Tính toàn bộ metric theo spec mục 11 (Parkinson = lớp dương)."""
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "model"          : model_name,
        "accuracy"       : float(accuracy_score(y_true, y_pred)),
        "precision"      : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"         : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"             : float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc"        : float(roc_auc_score(y_true, y_proba)),
        "sensitivity"    : float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity"    : float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "confusion_matrix": cm.tolist(),
        "report"         : classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Control", "Parkinson"], zero_division=0,
        ),
    }


def _save_plots(y_true, probas: Dict[str, np.ndarray], cm_data, best_model_name: str,
                out_dir: Path) -> None:
    """Lưu Confusion Matrix (model tốt nhất) và ROC Curve (tất cả model)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        log.warning("matplotlib chưa cài — bỏ qua vẽ biểu đồ.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm_arr = np.array(cm_data)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Parkinson"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "Parkinson"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_model_name}\n(nested CV theo subject)", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        lw = 2.5 if name == best_model_name else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — nested CV theo subject")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _group_cv(y: pd.Series, groups: pd.Series, max_splits: int, random_state: int = RANDOM_STATE):
    """StratifiedGroupKFold với số fold không vượt quá số subject của lớp nhỏ nhất."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = int(min(max_splits, groups.groupby(y).nunique().min()))
    if n_splits < 2:
        raise ValueError(
            "Cần ít nhất 2 subject cho mỗi lớp (Parkinson/Control) để chia dữ liệu theo bệnh nhân."
        )
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _nested_cv_proba(pipe, grid, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                     outer_splits: int = 5) -> np.ndarray:
    """
    Nested CV theo subject: vòng ngoài ước lượng hiệu năng, vòng trong GridSearchCV.
    Trả về xác suất out-of-fold cho mọi bản ghi — mỗi bản ghi được dự đoán bởi model
    chưa từng thấy bệnh nhân đó.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    oof = np.full(len(y), np.nan)
    for tr, te in _group_cv(y, groups, outer_splits).split(X, y, groups):
        y_tr, g_tr = y.iloc[tr], groups.iloc[tr]
        gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc",
                          cv=_group_cv(y_tr, g_tr, 4), n_jobs=-1)
        gs.fit(X.iloc[tr], y_tr, groups=g_tr)
        oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
    return oof


def group_train_test_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                           n_splits: int = 4, random_state: int = RANDOM_STATE):
    """
    Chia train/test theo bệnh nhân, giữ tỉ lệ lớp (StratifiedGroupKFold, lấy fold đầu làm test).
    n_splits=4 → ~25% subject vào test (UCI chỉ có 8 người Control).

    Trả về (train_idx, test_idx). Đảm bảo không subject nào xuất hiện ở cả hai tập.
    """
    train_idx, test_idx = next(_group_cv(y, groups, n_splits, random_state).split(X, y, groups))

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    assert not overlap, f"Data leakage: subject xuất hiện ở cả train và test: {overlap}"
    return train_idx, test_idx


def train_parkinson_model(verbose: bool = True) -> TrainResult:
    """
    Huấn luyện và so sánh LR / SVM / RF / GradientBoosting.

    1. Tải dataset (ưu tiên UCI CSV, sau đó audio recordings) kèm subject ID.
    2. Chia train/test theo bệnh nhân (StratifiedGroupKFold, ~25% subject vào test).
    3. Với mỗi model: GridSearchCV (CV theo nhóm bệnh nhân trên tập train, scoring ROC-AUC)
       → model triển khai; đánh giá trên tập test giữ riêng.
    4. Nested CV theo subject trên toàn bộ dữ liệu → ước lượng ổn định hơn (mọi subject đều
       được dự đoán out-of-fold một lần); Confusion Matrix và ROC vẽ từ dự đoán này.
    5. Chọn model có nested-CV ROC-AUC cao nhất. Inner-CV trên tập train (chỉ ~6 người Control)
       quá nhiễu để làm tiêu chí chọn.
    6. Lưu pipeline (Imputer → Scaler → Classifier), bảng so sánh và biểu đồ.

    Nhãn dự đoán luôn = (xác suất Parkinson ≥ 0.5) để nhất quán với probability hiển thị.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_uci_dataset()
    dataset_type = "uci"
    if data is None:
        data = _load_audio_dataset()
        dataset_type = "audio"
    if data is None:
        raise FileNotFoundError(
            "Không tìm thấy dataset Parkinson. Hãy đặt file 'parkinsons.data' vào data/ hoặc "
            "audio recordings vào data/raw/parkinson/{parkinson,control}/."
        )
    X, y, groups = data
    X = X.reset_index(drop=True); y = y.reset_index(drop=True); groups = groups.reset_index(drop=True)

    train_idx, test_idx = group_train_test_split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    if verbose:
        print(f"📂 Dataset ({dataset_type}): {len(y)} bản ghi, {groups.nunique()} subject "
              f"({int(y.sum())} Parkinson / {int((y == 0).sum())} Control bản ghi)")
        print(f"   Train: {len(y_train)} bản ghi / {g_train.nunique()} subject | "
              f"Test: {len(y_test)} bản ghi / {groups.iloc[test_idx].nunique()} subject")

    inner_cv = _group_cv(y_train, g_train, 4)

    all_metrics: List[Dict] = []
    fitted: Dict[str, object] = {}
    nested_probas: Dict[str, np.ndarray] = {}

    for name, (clf, grid) in _build_models().items():
        if verbose:
            print(f"\n🔄 {name}: GridSearchCV ({inner_cv.n_splits}-fold theo subject) + nested CV...")
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     clf),
        ])
        try:
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
            gs.fit(X_train, y_train, groups=g_train)
            oof = _nested_cv_proba(pipe, grid, X, y, groups)
        except Exception as e:
            log.warning("%s thất bại: %s", name, e)
            if verbose:
                print(f"   ❌ {name}: {e}")
            continue

        best = gs.best_estimator_
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, (y_proba >= 0.5).astype(int), y_proba, name)
        metrics["cv_roc_auc"] = float(gs.best_score_)
        metrics["best_params"] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        nested = _evaluate(y, (oof >= 0.5).astype(int), oof, name)
        metrics["nested_cv"] = {k: v for k, v in nested.items() if k not in ("model", "report")}
        all_metrics.append(metrics)
        fitted[name] = best
        nested_probas[name] = oof

        if verbose:
            n = metrics["nested_cv"]
            print(f"   Inner CV-AUC={metrics['cv_roc_auc']:.4f} | Test: AUC={metrics['roc_auc']:.4f} "
                  f"Sen={metrics['sensitivity']:.4f} Spe={metrics['specificity']:.4f} | "
                  f"Nested CV: AUC={n['roc_auc']:.4f} Sen={n['sensitivity']:.4f} "
                  f"Spe={n['specificity']:.4f} | {metrics['best_params']}")

    if not all_metrics:
        raise RuntimeError("Tất cả model đều thất bại khi huấn luyện.")

    best_result = max(all_metrics, key=lambda m: m["nested_cv"]["roc_auc"])
    best_pipeline = fitted[best_result["model"]]

    comp_rows = [{
        "Model"      : m["model"],
        "Accuracy"   : round(m["accuracy"],    4),
        "Precision"  : round(m["precision"],   4),
        "Recall"     : round(m["recall"],      4),
        "F1"         : round(m["f1"],          4),
        "ROC-AUC"    : round(m["roc_auc"],     4),
        "Sensitivity": round(m["sensitivity"], 4),
        "Specificity": round(m["specificity"], 4),
        "Inner_CV_ROC-AUC"    : round(m["cv_roc_auc"], 4),
        "NestedCV_Accuracy"   : round(m["nested_cv"]["accuracy"],    4),
        "NestedCV_F1"         : round(m["nested_cv"]["f1"],          4),
        "NestedCV_ROC-AUC"    : round(m["nested_cv"]["roc_auc"],     4),
        "NestedCV_Sensitivity": round(m["nested_cv"]["sensitivity"], 4),
        "NestedCV_Specificity": round(m["nested_cv"]["specificity"], 4),
        "Best_Params": str(m["best_params"]),
    } for m in all_metrics]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
    if verbose:
        print(f"\n📊 Bảng so sánh:\n{comp_df.drop(columns='Best_Params').T.to_string(header=False)}")
        print(f"💾 {COMPARISON_CSV_PATH}")

    _save_plots(y, nested_probas, best_result["nested_cv"]["confusion_matrix"],
                best_result["model"], _RESULTS_DIR)

    split_info = {
        "train_size": int(len(y_train)), "test_size": int(len(y_test)),
        "train_subjects": int(g_train.nunique()),
        "test_subjects": int(groups.iloc[test_idx].nunique()),
        "strategy": "StratifiedGroupKFold (subject-wise)",
    }
    bundle = {
        "pipeline"     : best_pipeline,
        "model_name"   : best_result["model"],
        "feature_names": list(X.columns),
        "metrics"      : {k: v for k, v in best_result.items() if k != "report"},
        "comparison"   : comp_rows,
        "dataset_type" : dataset_type,
        "split"        : split_info,
    }
    joblib.dump(bundle, PARKINSON_MODEL_PATH)
    if verbose:
        print(f"\n✅ Best model (theo nested-CV AUC): {best_result['model']} "
              f"(nested-CV AUC={best_result['nested_cv']['roc_auc']:.4f}, "
              f"test AUC={best_result['roc_auc']:.4f})")
        print(f"💾 {PARKINSON_MODEL_PATH}")

    return TrainResult(
        model_name       = best_result["model"],
        accuracy         = best_result["accuracy"],
        precision        = best_result["precision"],
        recall           = best_result["recall"],
        f1               = best_result["f1"],
        roc_auc          = best_result["roc_auc"],
        sensitivity      = best_result["sensitivity"],
        specificity      = best_result["specificity"],
        cv_roc_auc       = best_result["cv_roc_auc"],
        train_size       = split_info["train_size"],
        test_size        = split_info["test_size"],
        train_subjects   = split_info["train_subjects"],
        test_subjects    = split_info["test_subjects"],
        model_path       = str(PARKINSON_MODEL_PATH),
        report           = best_result["report"],
        confusion_matrix = best_result["confusion_matrix"],
        best_params      = best_result["best_params"],
        nested_cv        = best_result["nested_cv"],
        comparison       = comp_rows,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_parkinson_bundle() -> Dict:
    """Tải model bundle từ disk."""
    if not PARKINSON_MODEL_PATH.exists():
        raise FileNotFoundError("Parkinson model is not trained yet.")
    return joblib.load(PARKINSON_MODEL_PATH)


def is_model_trained() -> bool:
    return PARKINSON_MODEL_PATH.exists()


def get_model_info() -> Dict:
    """Thông tin model đã train (tên, metrics, bảng so sánh, cách chia dữ liệu)."""
    b = load_parkinson_bundle()
    return {
        "model": b["model_name"],
        "dataset_type": b.get("dataset_type"),
        "n_features": len(b["feature_names"]),
        "metrics": json_safe(b.get("metrics", {})),
        "comparison": json_safe(b.get("comparison", [])),
        "split": b.get("split"),
    }


def build_model_input(audio_feats: Dict[str, float], bundle: Dict) -> pd.DataFrame:
    """Ánh xạ đặc trưng audio sang đúng thứ tự cột model đã train. Cột thiếu → NaN (imputer)."""
    feat_names = bundle["feature_names"]
    if bundle.get("dataset_type", "uci") == "uci":
        source = {uci: audio_feats.get(a, np.nan) for a, uci in _AUDIO_TO_UCI.items()}
    else:
        source = audio_feats
    return pd.DataFrame([{c: source.get(c, np.nan) for c in feat_names}], columns=feat_names)


def json_safe(v):
    """NaN/Inf → None, numpy → Python (JSON chuẩn không có NaN)."""
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def _round(v: Optional[float], nd: int) -> Optional[float]:
    return round(float(v), nd) if v is not None and math.isfinite(float(v)) else None


def _features_summary(f: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {
        "f0_mean"      : _round(f.get("f0_mean"), 2),
        "f0_std"       : _round(f.get("f0_std"), 2),
        "jitter_local" : _round(f.get("jitter_local"), 6),
        "shimmer_local": _round(f.get("shimmer_local"), 6),
        "hnr"          : _round(f.get("hnr"), 2),
        "nhr"          : _round(f.get("nhr"), 6),
        "voiced_ratio" : _round(f.get("voiced_ratio"), 3),
        "duration_sec" : _round(f.get("duration_sec"), 2),
    }


def _quality_warnings(f: Dict[str, float]) -> List[str]:
    warns = []
    if f.get("duration_sec", 0) < 1.0:
        warns.append("Đoạn ghi âm quá ngắn (< 1 giây) — nên nói 'Aaaa' kéo dài 3–5 giây.")
    if f.get("voiced_ratio", 0) < 0.3 or not math.isfinite(f.get("jitter_local", float("nan"))):
        warns.append("Không phát hiện đủ đoạn hữu thanh ổn định — kết quả kém tin cậy. "
                     "Hãy ngân nguyên âm /a/ đều giọng trong môi trường yên tĩnh.")
    return warns


def _audio_info(audio_path, feats: Dict[str, float]) -> Dict:
    try:
        original_sr = int(librosa.get_samplerate(str(audio_path)))
    except Exception:
        original_sr = None
    return {
        "duration_sec": _round(feats.get("duration_sec"), 2),
        "sample_rate": SR,
        "original_sample_rate": original_sr,
    }


def predict_parkinson(audio_path: str | Path, bundle: Optional[Dict] = None) -> Dict:
    """
    Suy luận Parkinson từ file audio (dùng cho Flask API và Raspberry Pi).

    Returns dict: prediction, probability, model, risk_pct, features_summary,
    audio_info, warnings, disclaimer.
    """
    if bundle is None:
        bundle = load_parkinson_bundle()
    pipeline = bundle["pipeline"]

    audio_feats = extract_features_from_file(audio_path)
    X_input = build_model_input(audio_feats, bundle)

    proba = float(pipeline.predict_proba(X_input)[0][1])
    label = "Parkinson" if proba >= 0.5 else "Control"

    return {
        "prediction"      : label,
        "probability"     : round(proba, 4),
        "model"           : bundle["model_name"],
        "risk_pct"        : round(proba * 100, 2),
        "features_summary": _features_summary(audio_feats),
        "audio_info"      : _audio_info(audio_path, audio_feats),
        "warnings"        : _quality_warnings(audio_feats),
        "model_metrics"   : json_safe({k: bundle.get("metrics", {}).get("nested_cv", {}).get(k)
                                        for k in ("roc_auc", "sensitivity", "specificity")}),
        "disclaimer"      : DISCLAIMER_EN,
        "disclaimer_vi"   : DISCLAIMER_VI,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPAT — screen_parkinson() cho /api/parkinson/screen
# ══════════════════════════════════════════════════════════════════════════════

def _risk_level(risk_pct: float) -> Tuple[str, str]:
    if risk_pct >= 60:
        return ("⚠️ Nguy cơ cao",
                "Phát hiện nhiều dấu hiệu bất thường. Khuyến nghị tham khảo bác sĩ chuyên khoa thần kinh.")
    if risk_pct >= 30:
        return ("🔶 Nguy cơ trung bình",
                "Một số đặc trưng giọng nói bất thường nhẹ. Nên theo dõi định kỳ và tham vấn bác sĩ.")
    return ("✅ Nguy cơ thấp",
            "Giọng nói trong phạm vi bình thường. Không phát hiện dấu hiệu Parkinson rõ rệt.")


def screen_parkinson(audio_path) -> Dict:
    """
    Điểm vào cho /api/parkinson/screen.

    Model ML đã train → predict_parkinson. Chưa train (hoặc lỗi model) → rule-based theo ngưỡng MDVP.
    """
    if is_model_trained():
        try:
            result = predict_parkinson(audio_path)
            level_vi, advice = _risk_level(result["risk_pct"])
            return {
                **result,
                "risk_score"   : result["risk_pct"],
                "risk_level_vi": level_vi,
                "advice"       : advice,
                "features"     : result["features_summary"],
            }
        except Exception as e:
            log.warning("ML Parkinson predict lỗi, dùng rule-based: %s", e)

    return _rule_based_screen(audio_path)


def _rule_based_screen(audio_path) -> Dict:
    """
    Rule-based screening dựa trên ngưỡng bệnh lý MDVP (Kay Elemetrics):
    Jitter local 1.04%, Shimmer local 3.81%, NHR 0.19; HNR < 20 dB được coi là giảm.
    """
    feats = extract_features_from_file(audio_path)
    indicators: List[Dict] = []
    risk_score = 0.0

    def _add(feature, detail, severity, weight):
        indicators.append({"feature": feature, "detail": detail,
                           "severity": severity, "weight": weight})
        return weight

    jit, shim = feats.get("jitter_local", np.nan), feats.get("shimmer_local", np.nan)
    hnr, nhr = feats.get("hnr", np.nan), feats.get("nhr", np.nan)

    if jit > 0.0208:
        risk_score += _add("Jitter cao", f"Jitter = {jit*100:.2f}% (ngưỡng MDVP 1.04%)", "cao", 0.25)
    elif jit > 0.0104:
        risk_score += _add("Jitter vượt ngưỡng", f"Jitter = {jit*100:.2f}%", "trung bình", 0.15)

    if shim > 0.0762:
        risk_score += _add("Shimmer cao", f"Shimmer = {shim*100:.2f}% (ngưỡng MDVP 3.81%)", "cao", 0.25)
    elif shim > 0.0381:
        risk_score += _add("Shimmer vượt ngưỡng", f"Shimmer = {shim*100:.2f}%", "trung bình", 0.15)

    if hnr < 12:
        risk_score += _add("HNR rất thấp — giọng khàn/thở", f"HNR = {hnr:.1f} dB (bình thường > 20 dB)",
                           "cao", 0.25)
    elif hnr < 20:
        risk_score += _add("HNR thấp", f"HNR = {hnr:.1f} dB", "trung bình", 0.15)

    if nhr > 0.19:
        risk_score += _add("Tỷ lệ nhiễu/hài cao (NHR)", f"NHR = {nhr:.3f} (ngưỡng MDVP 0.19)",
                           "trung bình", 0.15)

    risk_score = min(risk_score, 1.0)
    risk_pct = round(risk_score * 100, 1)
    level_vi, advice = _risk_level(risk_pct)

    return {
        "prediction"      : "Parkinson" if risk_pct >= 60 else "Control",
        "probability"     : round(risk_score, 4),
        "risk_score"      : risk_pct,
        "risk_pct"        : risk_pct,
        "risk_level_vi"   : level_vi,
        "advice"          : advice,
        "indicators"      : indicators,
        "model"           : "Rule-based (chưa train ML)",
        "features"        : json_safe({k: round(v, 6) for k, v in feats.items()}),
        "features_summary": _features_summary(feats),
        "audio_info"      : _audio_info(audio_path, feats),
        "warnings"        : _quality_warnings(feats),
        "disclaimer"      : DISCLAIMER_EN,
        "disclaimer_vi"   : DISCLAIMER_VI,
    }
