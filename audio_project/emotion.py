"""Nhận diện cảm xúc giọng nói — Wav2Vec2 (primary) + DTW fallback.

Primary  : r-f/wav2vec-english-speech-emotion-recognition (HuggingFace)
Fallback : DTW so sánh MFCC (z-score, frame-level) với prototype tổng hợp
GPU      : DirectML (AMD) > CUDA > CPU
"""
from pathlib import Path
from typing import Dict
import logging, numpy as np

_model     = None
_extractor = None
_device    = None
_id2label  = None

MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
SR         = 16000
_N_MFCC    = 13
_N_FRAMES  = 60    # số frames prototype (đủ ngắn, DTW nhanh)
_HOP       = 256

LABEL_VI = {
    "angry":    "Tức giận",
    "disgust":  "Ghê tởm",
    "fear":     "Sợ hãi",
    "happy":    "Vui vẻ",
    "neutral":  "Bình thường",
    "sad":      "Buồn",
    "surprise": "Ngạc nhiên",
}

# ── Emotion prototype params ───────────────────────────────────────────────
# (pitch_hz, pitch_var, energy, zcr, tempo_hz)
# Dùng để sinh MFCC giả → mô tả "shape" đặc trưng của từng cảm xúc
_EMO_PARAMS = {
    "angry":    (220, 55, 0.08, 0.12, 3.0),
    "disgust":  (155, 28, 0.04, 0.07, 1.5),
    "fear":     (255, 70, 0.06, 0.14, 3.5),
    "happy":    (240, 60, 0.07, 0.10, 3.2),
    "neutral":  (175, 22, 0.05, 0.06, 2.0),
    "sad":      (130, 18, 0.03, 0.05, 1.2),
    "surprise": (280, 80, 0.09, 0.13, 4.0),
}


# ── DTW (1-D, Sakoe-Chiba) ─────────────────────────────────────────────────
def _dtw_1d(a: np.ndarray, b: np.ndarray, radius: int = 8) -> float:
    """Khoảng cách DTW chuẩn hóa giữa hai chuỗi 1-D."""
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / max(n, 1)
    for i in range(1, n + 1):
        jc = int(round(i * scale))
        j0, j1 = max(1, jc - radius), min(m, jc + radius) + 1
        for j in range(j0, j1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def _dtw_multivariate(A: np.ndarray, B: np.ndarray, radius: int = 8) -> float:
    """
    DTW đa chiều (frame × dim): tính tổng DTW theo từng chiều rồi lấy mean.
    A, B shape: (T, D).
    """
    assert A.ndim == 2 and B.ndim == 2, "Cần 2-D array (frames x dims)"
    n_dims = min(A.shape[1], B.shape[1])
    return float(np.mean([_dtw_1d(A[:, d], B[:, d], radius) for d in range(n_dims)]))


# ── Deterministic MFCC prototype ───────────────────────────────────────────
def _gen_proto(pitch: float, pitch_var: float,
               energy: float, zcr: float, tempo: float,
               n: int = _N_FRAMES) -> np.ndarray:
    """
    Sinh chuỗi MFCC prototype tổng hợp (n frames × _N_MFCC dims).
    Hoàn toàn deterministic (dùng sin/cos, không random).
    Sau đó z-score theo từng dim để khớp với MFCC thực.
    """
    t = np.linspace(0, 2 * np.pi, n)

    # Dim 0: log-energy ~ energy × scale + tempo modulation
    c0 = energy * 150 + 12 * np.sin(tempo * t)

    # Dim 1: spectral tilt ~ pitch
    c1 = (pitch / 200) * 8 + (pitch_var / 200) * 3 * np.cos(2 * t)

    # Dim 2: ZCR-related ~ high freq content
    c2 = zcr * 40 * np.sin(3 * t + 0.5)

    # Dims 3–12: harmonic texture từ pitch và energy
    rest = np.zeros((n, _N_MFCC - 3))
    for k in range(_N_MFCC - 3):
        freq = (k + 1) * tempo * 0.5
        amp  = energy * 30 / (k + 1)
        rest[:, k] = amp * np.sin(freq * t + k * 0.3)

    proto = np.column_stack([c0, c1, c2, rest])  # (n, 13)

    # Z-score normalize từng dim (giống với MFCC thực)
    mean = proto.mean(0)
    std  = proto.std(0) + 1e-10
    return (proto - mean) / std


# Cache prototypes khi module load
_PROTO_CACHE: Dict[str, np.ndarray] = {}


def _get_proto(emo: str) -> np.ndarray:
    if emo not in _PROTO_CACHE:
        params = _EMO_PARAMS[emo]
        _PROTO_CACHE[emo] = _gen_proto(*params, n=_N_FRAMES)
    return _PROTO_CACHE[emo]


def _resample_frames(A: np.ndarray, n_out: int) -> np.ndarray:
    """Resample mảng 2-D (T, D) về (n_out, D) bằng nội suy tuyến tính."""
    if len(A) == n_out:
        return A
    idx = np.linspace(0, len(A) - 1, n_out)
    return np.stack([np.interp(idx, np.arange(len(A)), A[:, d])
                     for d in range(A.shape[1])], axis=1)


# ── DTW-based emotion classifier ──────────────────────────────────────────
def _classify_by_dtw(y: np.ndarray, sr: int = SR) -> Dict:
    """Phân loại cảm xúc bằng DTW MFCC so với prototype — không cần model."""
    import librosa as _lib

    mfcc = _lib.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC, hop_length=_HOP).T  # (T,13)
    # Z-score normalize từng dim
    mfcc = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-10)
    # Resample về _N_FRAMES để tính DTW nhanh
    mfcc_rs = _resample_frames(mfcc, _N_FRAMES)

    distances: Dict[str, float] = {}
    for emo in _EMO_PARAMS:
        proto = _get_proto(emo)
        distances[emo] = _dtw_multivariate(mfcc_rs, proto, radius=8)

    # Softmax trên negative distance
    d_arr = np.array([distances[e] for e in _EMO_PARAMS])
    e_arr = np.exp(-d_arr * 5)             # hệ số khuếch đại sự khác biệt
    p_arr = e_arr / e_arr.sum()

    scores = []
    for emo, p in zip(_EMO_PARAMS, p_arr):
        scores.append({
            "label":    emo,
            "label_vi": LABEL_VI.get(emo, emo),
            "score":    round(float(p), 4),
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    top = scores[0]

    return {
        "emotion":    top["label"],
        "emotion_vi": top["label_vi"],
        "confidence": top["score"],
        "all_scores": scores,
        "device":     "dtw_fallback",
    }


# ── GPU Device ─────────────────────────────────────────────────────────────
def _get_device():
    import torch
    try:
        import torch_directml
        dev = torch_directml.device()
        logging.info("Using DirectML (AMD GPU): %s", dev)
        return dev
    except ImportError:
        pass
    if torch.cuda.is_available():
        logging.info("Using CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logging.info("Using CPU")
    return torch.device("cpu")


# ── Wav2Vec2 Model Loader ──────────────────────────────────────────────────
def _load_model():
    global _model, _extractor, _device, _id2label
    if _model is None:
        import torch, torch.nn as nn
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoConfig
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.info("Loading emotion model: %s ...", MODEL_NAME)

        config     = AutoConfig.from_pretrained(MODEL_NAME)
        _id2label  = config.id2label
        num_labels = len(_id2label)
        hidden     = config.hidden_size
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        class EmotionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.wav2vec2   = Wav2Vec2Model(config)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, num_labels),
                )

            def forward(self, input_values):
                out    = self.wav2vec2(input_values).last_hidden_state
                pooled = out.mean(dim=1)
                return self.classifier(pooled)

        _model   = EmotionModel()
        sf_path  = hf_hub_download(MODEL_NAME, "model.safetensors", revision="refs/pr/6")
        state    = load_file(sf_path)
        base_s, cls_s = {}, {}
        for k, v in state.items():
            if k.startswith("classifier.dense."):
                cls_s[k.replace("classifier.dense.", "classifier.0.")] = v
            elif k.startswith("classifier.out_proj."):
                cls_s[k.replace("classifier.out_proj.", "classifier.2.")] = v
            else:
                base_s[k] = v
        _model.load_state_dict({**base_s, **cls_s}, strict=False)
        _device = _get_device()
        _model  = _model.to(_device)
        _model.eval()
        logging.info("Emotion model ready on %s. Labels: %s",
                     _device, list(_id2label.values()))
    return _model, _extractor, _device, _id2label


# ── Inference ──────────────────────────────────────────────────────────────
def predict_emotion_from_array(y: np.ndarray, sr: int = SR) -> Dict:
    """Predict emotion — tries Wav2Vec2 first, falls back to DTW if failed."""
    try:
        import torch
        model, extractor, device, id2label = _load_model()
        inputs = extractor(y.astype(np.float32), sampling_rate=sr,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device))
            probs  = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        scores = [
            {"label": id2label[i], "label_vi": LABEL_VI.get(id2label[i], id2label[i]),
             "score": round(float(p), 4)}
            for i, p in enumerate(probs)
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[0]
        return {
            "emotion":    top["label"],
            "emotion_vi": top["label_vi"],
            "confidence": top["score"],
            "all_scores": scores,
            "device":     str(device),
        }
    except Exception as e:
        logging.warning("Wav2Vec2 unavailable (%s), using DTW fallback.", e)
        return _classify_by_dtw(y, sr)


def predict_emotion_from_file(audio_path: Path) -> Dict:
    """Predict emotion from audio file."""
    import librosa
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    return predict_emotion_from_array(y)
