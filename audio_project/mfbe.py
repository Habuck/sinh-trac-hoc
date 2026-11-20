"""Mel-scale Filter Bank Energies (MFBEs) — theo paper Vietnamese Speaker Verification
(Nguyen et al., IEEE Access 2024).

Paper: MFBEs hiệu quả hơn MFCCs cho deep learning speaker verification tiếng Việt
vì tránh nén log + DCT (năng lượng dồn vào hệ số bậc thấp).

Tham số thực nghiệm trong paper:
  - 80 filter banks
  - Hamming window 25 ms, shift 10 ms
  - Mean-normalize theo cửa sổ ~2 giây
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import librosa
import numpy as np

FeatureKind = Literal["mfbe", "mfcc"]


@dataclass(frozen=True)
class AcousticCfg:
    """Cấu hình frontend âm học khớp paper SV tiếng Việt."""
    sr: int = 16000
    n_mels: int = 80          # 80-dim MFBEs / MFCCs như paper
    n_mfcc: int = 80
    win_ms: float = 25.0      # Hamming 25 ms
    hop_ms: float = 10.0      # shift 10 ms
    pre_emphasis: float = 0.97
    cmvn_win_sec: float = 2.0 # mean-norm trong ~2 s
    fmin: float = 20.0
    fmax: Optional[float] = 7600.0

    @property
    def n_fft(self) -> int:
        return int(round(self.sr * self.win_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sr * self.hop_ms / 1000.0))

    @property
    def win_length(self) -> int:
        return self.n_fft

    @property
    def cmvn_frames(self) -> int:
        return max(1, int(round(self.cmvn_win_sec / (self.hop_ms / 1000.0))))


DEFAULT_CFG = AcousticCfg()


def pre_emphasize(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    if len(y) < 2 or coef <= 0:
        return y.astype(np.float32, copy=False)
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    out[1:] = y[1:] - coef * y[:-1]
    return out


def load_voice(path, sr: int = 16000, trim_db: float = 25.0) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=trim_db)
    if len(yt) < sr * 0.2:
        yt = y
    peak = float(np.max(np.abs(yt))) if len(yt) else 0.0
    if peak > 0:
        yt = yt / peak
    return yt.astype(np.float32)


def extract_mfbe(
    y: np.ndarray,
    cfg: AcousticCfg = DEFAULT_CFG,
    apply_cmvn: bool = True,
) -> np.ndarray:
    """Trả về MFBEs dạng (n_mels, T) — KHÔNG log, KHÔNG DCT (theo paper)."""
    y = pre_emphasize(np.asarray(y, dtype=np.float32), cfg.pre_emphasis)
    # power Mel filterbank energies (linear scale)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
        center=True,
    ).astype(np.float32)
    if apply_cmvn:
        S = sliding_cmvn(S, cfg.cmvn_frames)
    return S


def extract_mfcc_paper(
    y: np.ndarray,
    cfg: AcousticCfg = DEFAULT_CFG,
    apply_cmvn: bool = True,
) -> np.ndarray:
    """MFCC 80-dim cùng frontend (để so sánh với MFBE như trong paper)."""
    y = pre_emphasize(np.asarray(y, dtype=np.float32), cfg.pre_emphasis)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    ).astype(np.float32)
    if apply_cmvn:
        mfcc = sliding_cmvn(mfcc, cfg.cmvn_frames)
    return mfcc


def sliding_cmvn(feats: np.ndarray, win_frames: int) -> np.ndarray:
    """Mean normalization theo cửa sổ trượt (paper: ~2 giây)."""
    # feats: (D, T)
    d, t = feats.shape
    if t == 0:
        return feats
    if t <= win_frames:
        mean = feats.mean(axis=1, keepdims=True)
        return feats - mean

    # prefix sum for O(T) sliding mean
    csum = np.cumsum(feats, axis=1, dtype=np.float64)
    half = win_frames // 2
    out = np.empty_like(feats, dtype=np.float32)
    for i in range(t):
        left = max(0, i - half)
        right = min(t - 1, i + half)
        n = right - left + 1
        total = csum[:, right] - (csum[:, left - 1] if left > 0 else 0)
        out[:, i] = feats[:, i] - (total / n).astype(np.float32)
    return out


def extract_features(
    y: np.ndarray,
    kind: FeatureKind = "mfbe",
    cfg: AcousticCfg = DEFAULT_CFG,
    apply_cmvn: bool = True,
) -> np.ndarray:
    if kind == "mfbe":
        return extract_mfbe(y, cfg=cfg, apply_cmvn=apply_cmvn)
    if kind == "mfcc":
        return extract_mfcc_paper(y, cfg=cfg, apply_cmvn=apply_cmvn)
    raise ValueError(f"Unknown feature kind: {kind}")


def stats_embedding(feats: np.ndarray) -> np.ndarray:
    """Utterance embedding = [mean || std] rồi L2-normalize (thay pooling đơn giản)."""
    # feats: (D, T)
    if feats.size == 0 or feats.shape[1] == 0:
        emb = np.zeros(feats.shape[0] * 2, dtype=np.float32)
        return emb
    mean = feats.mean(axis=1)
    std = feats.std(axis=1)
    emb = np.concatenate([mean, std]).astype(np.float32)
    n = float(np.linalg.norm(emb))
    if n > 1e-8:
        emb /= n
    return emb


def utterance_embedding(
    y: np.ndarray,
    kind: FeatureKind = "mfbe",
    cfg: AcousticCfg = DEFAULT_CFG,
) -> np.ndarray:
    feats = extract_features(y, kind=kind, cfg=cfg, apply_cmvn=True)
    return stats_embedding(feats)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def feature_summary(feats: np.ndarray) -> Dict[str, float]:
    """Tóm tắt số liệu để API/UI."""
    if feats.size == 0:
        return {"n_frames": 0, "n_bands": 0, "energy_mean": 0.0, "energy_std": 0.0}
    return {
        "n_frames": int(feats.shape[1]),
        "n_bands": int(feats.shape[0]),
        "energy_mean": float(np.mean(feats)),
        "energy_std": float(np.std(feats)),
        "band_energy_ratio_low": float(np.mean(feats[:10])) if feats.shape[0] >= 10 else 0.0,
        "band_energy_ratio_high": float(np.mean(feats[-30:])) if feats.shape[0] >= 30 else 0.0,
    }


def compare_mfbe_vs_mfcc(y: np.ndarray, cfg: AcousticCfg = DEFAULT_CFG) -> Dict:
    """Minh họa luận điểm paper: phân bố năng lượng MFBE vs MFCC."""
    mfbe = extract_mfbe(y, cfg=cfg, apply_cmvn=False)
    mfcc = extract_mfcc_paper(y, cfg=cfg, apply_cmvn=False)
    # chuẩn hóa từng frame về [0,1] theo max abs để so sánh phân bố
    def _norm_abs(x: np.ndarray) -> np.ndarray:
        m = np.max(np.abs(x), axis=0, keepdims=True) + 1e-8
        return np.abs(x) / m

    mfbe_n, mfcc_n = _norm_abs(mfbe), _norm_abs(mfcc)
    low_mfbe = float(np.mean(mfbe_n[:10])) if mfbe_n.shape[0] >= 10 else 0.0
    high_mfbe = float(np.mean(mfbe_n[-30:])) if mfbe_n.shape[0] >= 30 else 0.0
    low_mfcc = float(np.mean(mfcc_n[:10])) if mfcc_n.shape[0] >= 10 else 0.0
    high_mfcc = float(np.mean(mfcc_n[-30:])) if mfcc_n.shape[0] >= 30 else 0.0
    return {
        "mfbe": feature_summary(mfbe),
        "mfcc": feature_summary(mfcc),
        "energy_share": {
            "mfbe_first10": round(low_mfbe, 4),
            "mfbe_last30": round(high_mfbe, 4),
            "mfcc_first10": round(low_mfcc, 4),
            "mfcc_last30": round(high_mfcc, 4),
        },
        "note": (
            "MFCC thường dồn năng lượng vào hệ số bậc thấp; "
            "MFBE phân bố đều hơn trên các filter bank — phù hợp CNN/SV tiếng Việt "
            "(Nguyen et al., IEEE Access 2024)."
        ),
    }
