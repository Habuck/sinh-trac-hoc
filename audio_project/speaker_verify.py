"""Vietnamese Speaker Verification dùng Mel-scale Filter Bank Energies (MFBEs).

Pipeline theo Nguyen et al., IEEE Access 2024:
  Audio → pre-emphasis → Hamming 25ms/10ms → 80-dim MFBE → CMVN
  → utterance embedding (stats pooling) → Cosine similarity

Không yêu cầu train ECAPA/ResNet trước: enrollment lưu embedding trung bình;
verification so khớp cosine với ngưỡng. Có chế độ so sánh MFBE vs MFCC.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_project.core import PROJECT_ROOT
from audio_project.mfbe import (
    AcousticCfg,
    DEFAULT_CFG,
    FeatureKind,
    compare_mfbe_vs_mfcc,
    cosine_similarity,
    extract_features,
    feature_summary,
    load_voice,
    stats_embedding,
    utterance_embedding,
)

SPEAKERS_DIR = PROJECT_ROOT / "data" / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

# Ngưỡng mặc định (cosine). Có thể tinh chỉnh theo enrollment thực tế.
DEFAULT_THRESHOLD = 0.72
MIN_DURATION_SEC = 1.0


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    feature_kind: str
    embedding: List[float]
    n_enrollments: int
    created_at: float
    updated_at: float
    cfg: Dict
    meta: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SpeakerProfile":
        return cls(**d)


def _safe_id(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or f"spk_{int(time.time())}"


def _profile_path(speaker_id: str) -> Path:
    return SPEAKERS_DIR / f"{speaker_id}.json"


def list_speakers() -> List[Dict]:
    out = []
    for p in sorted(SPEAKERS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "speaker_id": d.get("speaker_id", p.stem),
                "display_name": d.get("display_name", p.stem),
                "feature_kind": d.get("feature_kind", "mfbe"),
                "n_enrollments": d.get("n_enrollments", 0),
                "updated_at": d.get("updated_at"),
            })
        except Exception:
            continue
    return out


def load_profile(speaker_id: str) -> SpeakerProfile:
    path = _profile_path(speaker_id)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy speaker: {speaker_id}")
    return SpeakerProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def delete_speaker(speaker_id: str) -> bool:
    path = _profile_path(speaker_id)
    if path.exists():
        path.unlink()
        return True
    return False


def _embed_file(
    audio_path: Path,
    kind: FeatureKind = "mfbe",
    cfg: AcousticCfg = DEFAULT_CFG,
) -> Tuple[np.ndarray, Dict]:
    y = load_voice(audio_path, sr=cfg.sr)
    dur = len(y) / float(cfg.sr)
    if dur < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio quá ngắn ({dur:.2f}s). Cần ≥ {MIN_DURATION_SEC:.0f}s giọng nói."
        )
    feats = extract_features(y, kind=kind, cfg=cfg, apply_cmvn=True)
    emb = stats_embedding(feats)
    info = {
        "duration_sec": round(dur, 3),
        "sample_rate": cfg.sr,
        "feature_kind": kind,
        "embedding_dim": int(emb.shape[0]),
        "features": feature_summary(feats),
    }
    return emb, info


def enroll_speaker(
    audio_path: Path,
    display_name: str,
    speaker_id: Optional[str] = None,
    feature_kind: FeatureKind = "mfbe",
    cfg: AcousticCfg = DEFAULT_CFG,
    merge: bool = True,
) -> Dict:
    """Đăng ký / cập nhật hồ sơ giọng nói (enrollment)."""
    sid = speaker_id or _safe_id(display_name)
    emb, info = _embed_file(audio_path, kind=feature_kind, cfg=cfg)
    now = time.time()
    path = _profile_path(sid)

    if merge and path.exists():
        old = load_profile(sid)
        if old.feature_kind != feature_kind:
            raise ValueError(
                f"Speaker '{sid}' đang dùng {old.feature_kind}, "
                f"không thể enroll thêm bằng {feature_kind}."
            )
        old_emb = np.asarray(old.embedding, dtype=np.float32)
        n = old.n_enrollments
        new_emb = (old_emb * n + emb) / (n + 1)
        nn = float(np.linalg.norm(new_emb))
        if nn > 1e-8:
            new_emb = new_emb / nn
        profile = SpeakerProfile(
            speaker_id=sid,
            display_name=display_name or old.display_name,
            feature_kind=feature_kind,
            embedding=new_emb.astype(float).tolist(),
            n_enrollments=n + 1,
            created_at=old.created_at,
            updated_at=now,
            cfg=asdict(cfg),
            meta={"last_enroll": info},
        )
    else:
        profile = SpeakerProfile(
            speaker_id=sid,
            display_name=display_name or sid,
            feature_kind=feature_kind,
            embedding=emb.astype(float).tolist(),
            n_enrollments=1,
            created_at=now,
            updated_at=now,
            cfg=asdict(cfg),
            meta={"last_enroll": info},
        )

    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "speaker_id": profile.speaker_id,
        "display_name": profile.display_name,
        "feature_kind": profile.feature_kind,
        "n_enrollments": profile.n_enrollments,
        "embedding_dim": len(profile.embedding),
        "audio_info": info,
        "profile_path": str(path),
    }


def verify_speaker(
    audio_path: Path,
    speaker_id: str,
    threshold: float = DEFAULT_THRESHOLD,
    cfg: AcousticCfg = DEFAULT_CFG,
) -> Dict:
    """Xác thực 1-1: so utterance với hồ sơ đã enroll (cosine)."""
    profile = load_profile(speaker_id)
    kind: FeatureKind = profile.feature_kind if profile.feature_kind in ("mfbe", "mfcc") else "mfbe"
    emb, info = _embed_file(audio_path, kind=kind, cfg=cfg)
    score = cosine_similarity(emb, np.asarray(profile.embedding, dtype=np.float32))
    accepted = score >= threshold
    return {
        "speaker_id": profile.speaker_id,
        "display_name": profile.display_name,
        "feature_kind": kind,
        "score": round(float(score), 4),
        "threshold": float(threshold),
        "accepted": bool(accepted),
        "decision": "ACCEPT" if accepted else "REJECT",
        "decision_vi": "Khớp giọng" if accepted else "Không khớp",
        "n_enrollments": profile.n_enrollments,
        "audio_info": info,
        "method": "cosine_similarity",
        "reference": (
            "Nguyen et al., Vietnamese Speaker Verification with Mel-scale "
            "Filter Bank Energies and Deep Learning, IEEE Access 2024"
        ),
    }


def identify_speaker(
    audio_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    feature_kind: FeatureKind = "mfbe",
    cfg: AcousticCfg = DEFAULT_CFG,
    top_k: int = 3,
) -> Dict:
    """1-N: tìm speaker gần nhất trong danh sách đã enroll."""
    speakers = list_speakers()
    if not speakers:
        raise ValueError("Chưa có speaker nào được enroll.")

    emb, info = _embed_file(audio_path, kind=feature_kind, cfg=cfg)
    ranked = []
    for s in speakers:
        try:
            prof = load_profile(s["speaker_id"])
            if prof.feature_kind != feature_kind:
                continue
            sc = cosine_similarity(emb, np.asarray(prof.embedding, dtype=np.float32))
            ranked.append({
                "speaker_id": prof.speaker_id,
                "display_name": prof.display_name,
                "score": round(float(sc), 4),
            })
        except Exception:
            continue

    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0] if ranked else None
    accepted = bool(best and best["score"] >= threshold)
    return {
        "feature_kind": feature_kind,
        "threshold": float(threshold),
        "accepted": accepted,
        "best_match": best if accepted else None,
        "top_k": ranked[:top_k],
        "candidates": len(ranked),
        "audio_info": info,
        "decision_vi": (
            f"Nhận diện: {best['display_name']}" if accepted and best
            else "Không khớp với speaker nào"
        ),
    }


def analyze_features(audio_path: Path, cfg: AcousticCfg = DEFAULT_CFG) -> Dict:
    """Phân tích MFBE vs MFCC trên một file (minh họa paper)."""
    y = load_voice(audio_path, sr=cfg.sr)
    cmp = compare_mfbe_vs_mfcc(y, cfg=cfg)
    emb_mfbe = utterance_embedding(y, kind="mfbe", cfg=cfg)
    emb_mfcc = utterance_embedding(y, kind="mfcc", cfg=cfg)
    return {
        "duration_sec": round(len(y) / cfg.sr, 3),
        "cfg": {
            "sr": cfg.sr,
            "n_mels": cfg.n_mels,
            "win_ms": cfg.win_ms,
            "hop_ms": cfg.hop_ms,
            "cmvn_win_sec": cfg.cmvn_win_sec,
        },
        "comparison": cmp,
        "embedding_dim": {
            "mfbe": int(emb_mfbe.shape[0]),
            "mfcc": int(emb_mfcc.shape[0]),
        },
        "intra_cosine_mfbe_vs_mfcc": round(
            cosine_similarity(emb_mfbe, emb_mfcc), 4
        ),
    }


def compare_pair(
    audio_a: Path,
    audio_b: Path,
    feature_kind: FeatureKind = "mfbe",
    threshold: float = DEFAULT_THRESHOLD,
    cfg: AcousticCfg = DEFAULT_CFG,
    also_mfcc: bool = True,
) -> Dict:
    """So khớp 2 utterance (không cần enroll) — MFBE (+ tùy chọn MFCC)."""
    emb_a, info_a = _embed_file(audio_a, kind=feature_kind, cfg=cfg)
    emb_b, info_b = _embed_file(audio_b, kind=feature_kind, cfg=cfg)
    score = cosine_similarity(emb_a, emb_b)
    result = {
        "feature_kind": feature_kind,
        "score": round(float(score), 4),
        "threshold": float(threshold),
        "same_speaker": bool(score >= threshold),
        "decision_vi": "Cùng người nói" if score >= threshold else "Khác người nói",
        "audio_a": info_a,
        "audio_b": info_b,
    }
    if also_mfcc and feature_kind == "mfbe":
        ea, _ = _embed_file(audio_a, kind="mfcc", cfg=cfg)
        eb, _ = _embed_file(audio_b, kind="mfcc", cfg=cfg)
        sc_mfcc = cosine_similarity(ea, eb)
        result["mfcc_score"] = round(float(sc_mfcc), 4)
        result["delta_mfbe_minus_mfcc"] = round(float(score - sc_mfcc), 4)
        result["note"] = (
            "Điểm MFBE thường cao hơn / ổn định hơn MFCC trên cặp cùng speaker "
            "(paper: embedding MFBE compact hơn)."
        )
    return result
