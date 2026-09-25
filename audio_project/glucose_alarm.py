"""Glucose Alarm Detection — CNN-based CGM alarm audio classifier.

Ported từ project: low-glucose-audio-alarm-cnn-detector
Kiến trúc: 3-layer CNN + Mel Spectrogram (64 bands, 1024 FFT)
Metrics: Accuracy 98.32%, Recall 96.43%, Precision 99.66%
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent
GLUCOSE_MODELS_DIR = _BASE / "models"

# Tên model mặc định (file .pth đầu tiên tìm thấy có chứa "glucose")
_DEFAULT_MODEL_NAME = "glucose_alarm_cnn_w1.5s_20260316_125405"
GLUCOSE_MODEL_PATH = GLUCOSE_MODELS_DIR / f"{_DEFAULT_MODEL_NAME}.pth"
GLUCOSE_METADATA_PATH = GLUCOSE_MODELS_DIR / f"{_DEFAULT_MODEL_NAME}_metadata.json"

# ── Audio / Model Parameters (default, có thể ghi đè từ metadata) ─────────────
SAMPLE_RATE = 16000
WINDOW_DURATION = 1.5   # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# ── CNN Architecture ──────────────────────────────────────────────────────────

def _build_model(n_mels: int = N_MELS):
    """Khởi tạo kiến trúc GlucoseAlarmCNN."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch chưa được cài đặt. Chạy: pip install torch")

    class GlucoseAlarmCNN(nn.Module):
        """
        3-layer CNN phát hiện âm thanh báo động máy đo đường huyết (CGM).

        Input : (batch, 1, n_mels, time_frames)
        Output: (batch, 1)  — single logit for binary classification
        """

        def __init__(self, n_mels: int = 64):
            super().__init__()
            # Block 1
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            # Block 2
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            # Block 3
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)
            # Global avg pool + FC
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc      = nn.Linear(128, 1)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            return self.fc(x)

    return GlucoseAlarmCNN(n_mels=n_mels)


# ── Model Loading ─────────────────────────────────────────────────────────────

_cached_model = None
_cached_metadata = None


def load_glucose_metadata() -> dict:
    """Đọc metadata JSON của model (audio params, metrics...)."""
    global _cached_metadata
    if _cached_metadata is not None:
        return _cached_metadata
    if not GLUCOSE_METADATA_PATH.exists():
        return {}
    with open(GLUCOSE_METADATA_PATH, "r", encoding="utf-8") as f:
        _cached_metadata = json.load(f)
    return _cached_metadata


def load_glucose_model():
    """
    Load model CNN từ file .pth. Kết quả được cache trong bộ nhớ.

    Returns
    -------
    model : torch.nn.Module  ở eval mode
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch chưa được cài đặt.")

    if not GLUCOSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {GLUCOSE_MODEL_PATH}\n"
            "Vui lòng đặt file .pth vào thư mục models/"
        )

    # Lấy n_mels từ metadata nếu có
    meta = load_glucose_metadata()
    n_mels = meta.get("audio_params", {}).get("n_mels", N_MELS)

    device = torch.device("cpu")
    model = _build_model(n_mels=n_mels)
    model.load_state_dict(torch.load(str(GLUCOSE_MODEL_PATH), map_location=device))
    model.to(device)
    model.eval()

    _cached_model = model
    logging.info("Glucose alarm model loaded: %s", GLUCOSE_MODEL_PATH.name)
    return model


# ── Feature Extraction ────────────────────────────────────────────────────────

def _audio_to_melspectrogram(audio: np.ndarray,
                              sr: int = SAMPLE_RATE,
                              n_mels: int = N_MELS,
                              n_fft: int = N_FFT,
                              hop_length: int = HOP_LENGTH):
    """
    Chuyển audio numpy → mel-spectrogram tensor PyTorch.

    Chiến lược chuẩn hoá hybrid:
    - ref=np.max  → nhận dạng pattern độc lập âm lượng
    - Penalise im lặng cực đoan (rms < 0.0001) → giảm false positive
    """
    import librosa
    import torch

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    if rms_energy < 0.0001:          # im lặng cực đoan
        mel_spec_db = mel_spec_db - 60

    # (1, 1, n_mels, time_frames)
    return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0), rms_energy


# ── Main Prediction API ───────────────────────────────────────────────────────

def predict_glucose_alarm(audio_path,
                          confidence_threshold: float = 0.9) -> dict:
    """
    Dự đoán xác suất âm thanh là CGM alarm từ file audio.

    Parameters
    ----------
    audio_path : str | Path
        Đường dẫn file WAV/MP3/...
    confidence_threshold : float
        Ngưỡng phân loại (mặc định 0.9 theo paper gốc)

    Returns
    -------
    dict
        {
            "probability": float,   # 0.0 – 1.0
            "is_alarm": bool,
            "confidence": float,    # % độ chắc chắn (max side)
            "label": str,           # "glucose_alarm" | "no_alarm"
            "rms_energy": float,
            "threshold_used": float
        }
    """
    import librosa
    import torch

    meta = load_glucose_metadata()
    audio_params = meta.get("audio_params", {})
    sr         = audio_params.get("sample_rate",   SAMPLE_RATE)
    n_mels     = audio_params.get("n_mels",        N_MELS)
    n_fft      = audio_params.get("n_fft",         N_FFT)
    hop_length = audio_params.get("hop_length",    HOP_LENGTH)

    # Load audio
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    # Feature extraction
    mel_tensor, rms_energy = _audio_to_melspectrogram(
        audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    # Inference
    model = load_glucose_model()
    device = next(model.parameters()).device
    mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        logit = model(mel_tensor)
        probability = float(torch.sigmoid(logit).item())

    is_alarm   = probability >= confidence_threshold
    confidence = max(probability, 1.0 - probability)
    label      = "glucose_alarm" if is_alarm else "no_alarm"

    return {
        "probability":      round(probability, 6),
        "is_alarm":         is_alarm,
        "confidence":       round(confidence, 6),
        "label":            label,
        "rms_energy":       round(rms_energy, 8),
        "threshold_used":   confidence_threshold,
    }


def get_glucose_model_info() -> dict:
    """
    Trả về thông tin model và metrics từ metadata JSON.
    Dùng cho API /api/glucose/model-info.
    """
    if not GLUCOSE_METADATA_PATH.exists():
        return {"available": False, "model_path": str(GLUCOSE_MODEL_PATH)}

    meta = load_glucose_metadata()
    return {
        "available":      GLUCOSE_MODEL_PATH.exists(),
        "model_name":     meta.get("model_name", _DEFAULT_MODEL_NAME),
        "model_file":     meta.get("model_file", GLUCOSE_MODEL_PATH.name),
        "audio_params":   meta.get("audio_params", {}),
        "training_params": meta.get("training_params", {}),
        "dataset_info":   meta.get("dataset_info", {}),
        "metrics":        meta.get("metrics", {}),
        "confusion_matrix": meta.get("confusion_matrix", {}),
        "system_info":    meta.get("system_info", {}),
    }
