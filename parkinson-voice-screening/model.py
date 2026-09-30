"""
model.py — Huấn luyện, đánh giá và so sánh đa mô hình ML cho phát hiện Parkinson qua giọng nói.

Pipeline:
  1. Đọc UCI Parkinson's Dataset (parkinsons.data — 195 mẫu, 22 đặc trưng)
  2. Chuẩn hoá bằng StandardScaler
  3. Huấn luyện & so sánh 4 mô hình: Logistic Regression, SVM, Random Forest, Gradient Boosting
  4. Subject-aware GroupKFold CV (chống Data Leakage — mỗi bệnh nhân chỉ nằm 1 fold)
  5. Đánh giá: Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity
  6. Lưu model tốt nhất + scaler + kết quả vào models/

Tham khảo:
  - Little MA et al. (2007) — "Exploiting Nonlinear Recurrence..."
  - Tsanas A et al. (2012) — "Novel speech signal processing algorithms..."
  - UCI ML Repository: Parkinson's Disease Dataset (195 mẫu, 23 features)
"""

from __future__ import annotations
import pickle
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GridSearchCV, cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline

BASE_DIR    = Path(__file__).resolve().parent
DATA_PATH   = BASE_DIR / "data" / "parkinsons.data"
MODEL_PATH  = BASE_DIR / "models" / "svm_parkinson.pkl"
RESULTS_DIR = BASE_DIR / "results"

# Các cột đặc trưng từ UCI Parkinson's dataset cần sử dụng
UCI_FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# Ánh xạ từ đặc trưng audio của chúng ta → đặc trưng UCI gần nhất
# (dùng cho chế độ predict từ file audio thực)
AUDIO_TO_UCI_MAP: Dict[str, str] = {
    "f0_mean"      : "MDVP:Fo(Hz)",
    "f0_max"       : "MDVP:Fhi(Hz)",
    "f0_min"       : "MDVP:Flo(Hz)",
    "jitter_local" : "MDVP:Jitter(%)",
    "jitter_rap"   : "MDVP:RAP",
    "jitter_ppq5"  : "MDVP:PPQ",
    "jitter_ddp"   : "Jitter:DDP",
    "shimmer_local": "MDVP:Shimmer",
    "shimmer_db"   : "MDVP:Shimmer(dB)",
    "shimmer_apq3" : "Shimmer:APQ3",
    "shimmer_apq5" : "Shimmer:APQ5",
    "nhr"          : "NHR",
    "hnr"          : "HNR",
    "ppe"          : "PPE",
    # Spectral features mới (Phase 7)
    "rpde"         : "RPDE",
    "dfa"          : "DFA",
    "spread1"      : "spread1",
    "spread2"      : "spread2",
    "d2"           : "D2",
}

# Nhãn cho hiển thị tiếng Việt
MODEL_LABELS = {
    "logistic_regression": "Hồi quy Logistic (LR)",
    "svm":                 "Máy vector hỗ trợ (SVM-RBF)",
    "random_forest":       "Rừng ngẫu nhiên (RF)",
    "gradient_boosting":   "Tăng cường Gradient (GB)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SUBJECT ID EXTRACTION — Chống Data Leakage (bắt buộc)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_subject_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Trích xuất Subject ID từ cột 'name' trong UCI dataset.
    Mỗi bệnh nhân có nhiều recording, nhưng toàn bộ phải nằm 1 fold.
    Ví dụ: 'phon_R01_S01_1' → subject = 'R01_S01'
    """
    names = df["name"].values
    groups = []
    for n in names:
        parts = n.split("_")
        if len(parts) >= 3:
            groups.append(f"{parts[1]}_{parts[2]}")
        else:
            groups.append(n)
    return np.array(groups)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS — 4 mô hình ML (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def _get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Trả về cấu hình 4 mô hình ML với param_grid cho GridSearchCV."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000, random_state=42, solver="lbfgs"
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
            },
            "label_vi": "Hồi quy Logistic (LR)",
        },
        "svm": {
            "estimator": SVC(
                probability=True, random_state=42
            ),
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "kernel": ["rbf"],
            },
            "label_vi": "Máy Vector Hỗ trợ (SVM-RBF)",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=42, n_jobs=-1
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "label_vi": "Rừng Ngẫu nhiên (RF)",
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(
                random_state=42
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
            "label_vi": "Tăng cường Gradient (GB)",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS — Đầy đủ cho ĐATN (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Tính toàn bộ metrics đánh giá cho bài toán nhị phân (Parkinson screening).

    Metrics:
      - Accuracy, Precision, Recall, F1-score
      - ROC-AUC (nếu có y_proba)
      - Sensitivity (= Recall) — Khả năng phát hiện đúng ca bệnh
      - Specificity — Khả năng phát hiện đúng ca khỏe mạnh
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.0
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE MODEL TRAINING — Huấn luyện 1 model với GridSearchCV
# ══════════════════════════════════════════════════════════════════════════════

def _train_single_model(
    model_name: str,
    X_scaled: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Huấn luyện 1 model ML với GridSearchCV + GroupKFold CV."""
    config = _get_model_configs()[model_name]
    label_vi = config["label_vi"]

    if verbose:
        print(f"\n{'─'*60}")
        print(f"🔧 Đang huấn luyện: {label_vi}")
        print(f"{'─'*60}")

    t0 = time.time()

    # GridSearchCV trên StratifiedKFold (tìm hyperparameters tối ưu)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        config["estimator"],
        config["param_grid"],
        cv=cv_inner,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_scaled, y)
    best_params = grid.best_params_

    if verbose:
        print(f"   ✅ Best params: {best_params}")
        print(f"   📊 GridSearch best CV AUC: {grid.best_score_:.4f}")

    # Huấn luyện model cuối trên toàn bộ dữ liệu
    best_model = grid.best_estimator_

    # Subject-aware Cross-Validation (GroupKFold — chống Data Leakage)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_splits_group = min(10, n_groups)

    if n_splits_group >= 2:
        gkf = GroupKFold(n_splits=n_splits_group)
        y_pred_cv = cross_val_predict(best_model, X_scaled, y, cv=gkf, groups=groups)
        try:
            y_proba_cv = cross_val_predict(
                best_model, X_scaled, y, cv=gkf, groups=groups, method="predict_proba"
            )[:, 1]
        except Exception:
            y_proba_cv = None

        cv_metrics = _compute_metrics(y, y_pred_cv, y_proba_cv)
        if verbose:
            print(f"   📈 GroupKFold CV ({n_splits_group}-fold, {n_groups} subjects):")
            print(f"      Accuracy   : {cv_metrics['accuracy']*100:.2f}%")
            print(f"      Precision  : {cv_metrics['precision']*100:.2f}%")
            print(f"      Recall     : {cv_metrics['recall']*100:.2f}%")
            print(f"      F1-score   : {cv_metrics['f1']*100:.2f}%")
            print(f"      Sensitivity: {cv_metrics['sensitivity']*100:.2f}%")
            print(f"      Specificity: {cv_metrics['specificity']*100:.2f}%")
            if cv_metrics.get("roc_auc"):
                print(f"      ROC-AUC    : {cv_metrics['roc_auc']:.4f}")
    else:
        cv_metrics = None
        y_proba_cv = None

    # Đánh giá trên toàn bộ tập (resubstitution — để ROC Curve)
    y_pred_full = best_model.predict(X_scaled)
    try:
        y_proba_full = best_model.predict_proba(X_scaled)[:, 1]
    except Exception:
        y_proba_full = None

    full_metrics = _compute_metrics(y, y_pred_full, y_proba_full)

    # ROC Curve data
    roc_data = None
    if y_proba_full is not None:
        fpr, tpr, thresholds = roc_curve(y, y_proba_full)
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    # Feature Importance (RF & GB)
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        feature_importance = dict(zip(UCI_FEATURE_COLS, fi.tolist()))
    elif hasattr(best_model, "coef_"):
        fi = np.abs(best_model.coef_[0])
        feature_importance = dict(zip(UCI_FEATURE_COLS, fi.tolist()))

    elapsed = time.time() - t0

    if verbose:
        print(f"   ⏱  Thời gian huấn luyện: {elapsed:.2f}s")

    return {
        "model_name":         model_name,
        "label_vi":           label_vi,
        "model":              best_model,
        "best_params":        best_params,
        "grid_best_auc":      float(grid.best_score_),
        "cv_metrics":         cv_metrics,
        "full_metrics":       full_metrics,
        "roc_data":           roc_data,
        "feature_importance": feature_importance,
        "confusion_matrix":   confusion_matrix(y, y_pred_full).tolist(),
        "report":             classification_report(y, y_pred_full,
                                                     target_names=["Khỏe mạnh", "Parkinson"]),
        "train_time":         round(elapsed, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-MODEL TRAINING — So sánh 4 model (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    data_path: Path = DATA_PATH,
    save_path: Path = MODEL_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Huấn luyện & so sánh 4 mô hình ML trên UCI Parkinson's dataset.

    Returns:
        dict chứa:
        - models: dict model_name → model results
        - comparison_table: DataFrame so sánh
        - best_model_name: tên model tốt nhất (theo CV AUC)
        - scaler, feature_names
    """
    if verbose:
        print("=" * 60)
        print("🧠 ParkiScan — Huấn luyện & So sánh 4 Mô hình ML")
        print("=" * 60)
        print(f"\n📂 Đang tải UCI Parkinson's Dataset...")

    df = pd.read_csv(data_path)
    X  = df[UCI_FEATURE_COLS].values.astype(np.float64)
    y  = df["status"].values.astype(int)
    groups = _extract_subject_groups(df)

    n_pd      = int(y.sum())
    n_healthy = len(y) - n_pd
    n_subjects = len(np.unique(groups))

    if verbose:
        print(f"   ✅ {len(df)} mẫu | {n_pd} Parkinson | {n_healthy} Khỏe mạnh")
        print(f"   👥 {n_subjects} đối tượng riêng biệt (subject-aware CV)")
        print(f"   📊 {len(UCI_FEATURE_COLS)} đặc trưng UCI")

    # Chuẩn hoá
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Huấn luyện tất cả 4 model
    all_results = {}
    for model_name in ["logistic_regression", "svm", "random_forest", "gradient_boosting"]:
        result = _train_single_model(model_name, X_scaled, y, groups, verbose)
        all_results[model_name] = result

    # Bảng so sánh
    comparison_rows = []
    for name, res in all_results.items():
        cv = res["cv_metrics"] or {}
        row = {
            "Model":       res["label_vi"],
            "model_key":   name,
            "Accuracy":    cv.get("accuracy", 0.0),
            "Precision":   cv.get("precision", 0.0),
            "Recall":      cv.get("recall", 0.0),
            "F1":          cv.get("f1", 0.0),
            "ROC-AUC":     cv.get("roc_auc", 0.0),
            "Sensitivity": cv.get("sensitivity", 0.0),
            "Specificity": cv.get("specificity", 0.0),
            "Train Time":  res["train_time"],
        }
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)

    # Xác định model tốt nhất (theo CV ROC-AUC)
    best_idx = comparison_df["ROC-AUC"].idxmax()
    best_model_name = comparison_df.loc[best_idx, "model_key"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 BẢNG SO SÁNH 4 MÔ HÌNH (GroupKFold CV)")
        print(f"{'='*60}")
        for _, row in comparison_df.iterrows():
            marker = " ⭐ BEST" if row["model_key"] == best_model_name else ""
            print(f"\n  {row['Model']}{marker}")
            print(f"    Accuracy   : {row['Accuracy']*100:.2f}%")
            print(f"    Precision  : {row['Precision']*100:.2f}%")
            print(f"    Recall     : {row['Recall']*100:.2f}%")
            print(f"    F1-score   : {row['F1']*100:.2f}%")
            print(f"    ROC-AUC    : {row['ROC-AUC']:.4f}")
            print(f"    Sensitivity: {row['Sensitivity']*100:.2f}%")
            print(f"    Specificity: {row['Specificity']*100:.2f}%")
            print(f"    Train Time : {row['Train Time']:.3f}s")
        print(f"\n{'='*60}")
        print(f"🏆 Model tốt nhất: {all_results[best_model_name]['label_vi']}")
        print(f"{'='*60}")

    # Lưu model tốt nhất
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_result = all_results[best_model_name]
    with open(save_path, "wb") as f:
        pickle.dump({
            "model":          best_result["model"],
            "scaler":         scaler,
            "feature_names":  UCI_FEATURE_COLS,
            "best_params":    best_result["best_params"],
            "model_name":     best_model_name,
            "results":        best_result,
        }, f)

    if verbose:
        print(f"💾 Model tốt nhất đã lưu: {save_path}")

    # Lưu kết quả so sánh vào CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / "parkinson_model_comparison.csv", index=False)

    if verbose:
        print(f"📄 Bảng so sánh đã lưu: {RESULTS_DIR / 'parkinson_model_comparison.csv'}")

    return {
        "models":           all_results,
        "comparison_table": comparison_df,
        "best_model_name":  best_model_name,
        "scaler":           scaler,
        "feature_names":    UCI_FEATURE_COLS,
        "n_samples":        len(y),
        "n_features":       len(UCI_FEATURE_COLS),
        "n_subjects":       n_subjects,
        "n_parkinson":      n_pd,
        "n_healthy":        n_healthy,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBLE — train_model() (giữ tương thích app.py cũ)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    data_path: Path = DATA_PATH,
    save_path: Path = MODEL_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Huấn luyện SVM trên UCI Parkinson's dataset (backward compatible).
    Bây giờ nội bộ gọi train_all_models() để so sánh 4 model.
    """
    multi_results = train_all_models(data_path, save_path, verbose)
    best_name = multi_results["best_model_name"]
    best_res  = multi_results["models"][best_name]
    cv = best_res["cv_metrics"] or {}

    # Trả về format tương thích app.py cũ
    return {
        "accuracy":          cv.get("accuracy", 0.0),
        "auc":               cv.get("roc_auc", 0.0),
        "cv_accuracy":       f"{cv.get('accuracy', 0)*100:.2f}%",
        "cv_auc":            f"{cv.get('roc_auc', 0):.4f}",
        "report":            best_res["report"],
        "confusion_matrix":  best_res["confusion_matrix"],
        "best_params":       best_res["best_params"],
        "n_samples":         multi_results["n_samples"],
        "n_features":        multi_results["n_features"],
        "feature_names":     UCI_FEATURE_COLS,
        # Mới: dữ liệu so sánh đa model
        "multi_results":     multi_results,
        "best_model_name":   best_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_model(path: Path = MODEL_PATH) -> Tuple[Any, Any, list]:
    """Tải model, scaler và danh sách feature names từ file pkl."""
    if not path.exists():
        raise FileNotFoundError(f"Model chưa được huấn luyện. Chạy train_model() trước.\nPath: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["feature_names"]


def predict_from_audio_features(
    audio_feats: Dict[str, float],
    model=None,
    scaler=None,
    feature_names: list = None,
) -> Dict[str, Any]:
    """
    Suy luận từ dict đặc trưng audio (trích xuất từ file thực).

    Ánh xạ audio features → UCI feature space qua AUDIO_TO_UCI_MAP.
    Các đặc trưng UCI không có ánh xạ sẽ dùng giá trị 0 (median của dataset).
    """
    if model is None:
        model, scaler, feature_names = load_model()

    # Tạo UCI feature vector
    uci_row: Dict[str, float] = {}
    for audio_key, uci_key in AUDIO_TO_UCI_MAP.items():
        uci_row[uci_key] = audio_feats.get(audio_key, 0.0)

    # Điền 0 cho các đặc trưng không có ánh xạ
    for col in feature_names:
        if col not in uci_row:
            uci_row[col] = 0.0

    X = np.array([[uci_row[col] for col in feature_names]], dtype=np.float64)
    X_scaled = scaler.transform(X)

    pred  = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction"   : pred,
        "probability"  : round(proba * 100, 2),
        "label"        : "Parkinson" if pred == 1 else "Khỏe mạnh",
        "uci_features" : {k: round(v, 5) for k, v in uci_row.items()},
    }


def is_model_trained() -> bool:
    """Kiểm tra xem model đã được huấn luyện chưa."""
    return MODEL_PATH.exists()
