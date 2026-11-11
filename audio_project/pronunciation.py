"""Đánh giá ngữ điệu tiếng Việt — so sánh MFCC, pitch, rhythm, energy qua DTW."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa, numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
HOP_LENGTH = 256
DATASET_DIR: Optional[Path] = None
_DTW_TARGET_FRAMES = 80
_MIN_SEQUENCE_LEN = 3


def _find_dataset_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    for c in [root / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main"
                   / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main",
              root / "760-Hours-Vietnamese-Speech-Data-by-Mobile-Phone-main"]:
        if c.exists() and any(c.glob("*.wav")):
            return c
    for d in root.rglob("*.wav"):
        if any(d.parent.glob("*.txt")):
            return d.parent
    raise FileNotFoundError("Không tìm thấy thư mục dataset Vietnamese Speech.")


def get_dataset_dir() -> Path:
    global DATASET_DIR
    if DATASET_DIR is None:
        DATASET_DIR = _find_dataset_dir()
    return DATASET_DIR


def list_sentences(limit: int = 50) -> List[Dict]:
    ds_dir = get_dataset_dir()
    sentences = []
    for txt_path in sorted(ds_dir.glob("*.txt"))[:limit]:
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        try:
            dur = librosa.get_duration(path=wav_path)
        except Exception:
            dur = 0
        sentences.append({"id": txt_path.stem, "text": text,
                          "wav_file": wav_path.name, "duration_sec": round(dur, 2)})
    return sentences


def _load_and_trim(fp: Path) -> np.ndarray:
    y, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _extract_mfcc(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.vstack([mfcc, librosa.feature.delta(mfcc)]).T


def _extract_pitch(y: np.ndarray) -> np.ndarray:
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return np.nan_to_num(f0, nan=0.0)


def _extract_energy(y: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]


def _ensure_2d(seq: np.ndarray) -> np.ndarray:
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Expected 1-D or 2-D sequence")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_sequence(seq: np.ndarray) -> np.ndarray:
    arr = _ensure_2d(seq)
    if len(arr) == 0:
        return arr
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (arr - mean) / std


def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    arr = _ensure_2d(seq)
    if len(arr) == 0:
        return np.zeros((0, arr.shape[1]), dtype=float)
    if len(arr) == target_len:
        return arr
    if len(arr) == 1:
        return np.repeat(arr, target_len, axis=0)
    idx = np.linspace(0, len(arr) - 1, target_len)
    base = np.arange(len(arr))
    return np.stack([np.interp(idx, base, arr[:, d]) for d in range(arr.shape[1])], axis=1)


def _prepare_for_dtw(seq: np.ndarray, target_len: Optional[int] = _DTW_TARGET_FRAMES) -> np.ndarray:
    arr = _normalize_sequence(seq)
    if target_len and len(arr) > target_len:
        arr = _resample_sequence(arr, target_len)
    return arr


def _has_min_sequence(seq: np.ndarray, min_len: int = _MIN_SEQUENCE_LEN) -> bool:
    return len(_ensure_2d(seq)) >= min_len


def _fast_dtw(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> float:
    seq1 = _prepare_for_dtw(seq1)
    seq2 = _prepare_for_dtw(seq2)
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")
    if n == 1 and m == 1:
        return float(np.linalg.norm(seq1[0] - seq2[0]))
    radius = max(radius, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / n if n > 0 else 1.0
    for i in range(1, n + 1):
        jc = int(round((i - 1) * scale)) + 1
        j_start = max(1, jc - radius)
        j_end = min(m, jc + radius) + 1
        for j in range(j_start, j_end):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    dist = D[n, m] / (n + m)
    return float(dist if np.isfinite(dist) else np.inf)


def _dist_to_score(dist: float, scale: float = 15.0) -> float:
    if not np.isfinite(dist):
        return 0.0
    return float(np.clip(100.0 * np.exp(-dist / scale), 0, 100))


def _pitch_score(f0_ref: np.ndarray, f0_user: np.ndarray) -> Tuple[float, str]:
    rv, uv = f0_ref[f0_ref > 0], f0_user[f0_user > 0]
    if len(rv) < 5 or len(uv) < 5:
        return 70.0, "Không đủ dữ liệu pitch để đánh giá chi tiết."

    ref_median = max(float(np.median(rv)), 1e-6)
    usr_median = max(float(np.median(uv)), 1e-6)
    rs = 12.0 * np.log2(np.clip(rv / ref_median, 1e-6, None))
    us = 12.0 * np.log2(np.clip(uv / usr_median, 1e-6, None))
    score = _dist_to_score(_fast_dtw(rs, us, radius=15), scale=3.0)
    fb = []
    rr = (np.percentile(us, 95) - np.percentile(us, 5)) / (np.percentile(rs, 95) - np.percentile(rs, 5) + 1e-10)
    if rr < 0.6:
        fb.append("Ngữ điệu hơi đơn điệu — thử nhấn nhá rõ hơn các thanh sắc/huyền/hỏi/ngã"); score *= 0.85
    elif rr > 1.5:
        fb.append("Ngữ điệu dao động quá mạnh — thử nói đều hơn"); score *= 0.9
    if abs(np.std(us) - np.std(rs)) > 1.5:
        fb.append("Độ biến thiên cao độ khác mẫu — chú ý thanh điệu từng từ")
    if not fb:
        fb.append("Ngữ điệu tốt! Thanh điệu khá chính xác" if score >= 80
                   else "Ngữ điệu khá ổn, cần cải thiện một số thanh" if score >= 60
                   else "Cần luyện thêm ngữ điệu — nghe kỹ mẫu và bắt chước đường nét lên/xuống")
    return float(np.clip(score, 0, 100)), "; ".join(fb)


def _rhythm_score(y_ref: np.ndarray, y_user: np.ndarray) -> Tuple[float, str]:
    dur_ratio = (len(y_user) / SAMPLE_RATE) / (len(y_ref) / SAMPLE_RATE + 1e-10)
    o_ref = librosa.onset.onset_strength(y=y_ref, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    o_usr = librosa.onset.onset_strength(y=y_user, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    ref_peak = np.max(o_ref) if len(o_ref) else 0.0
    usr_peak = np.max(o_usr) if len(o_usr) else 0.0
    if ref_peak <= 1e-8 or usr_peak <= 1e-8 or not _has_min_sequence(o_ref) or not _has_min_sequence(o_usr):
        base = 65.0 if 0.75 <= dur_ratio <= 1.35 else 55.0
        fb = "Không đủ nhịp onset rõ để đánh giá chính xác; thử ghi âm rõ hơn và giảm tạp âm."
        return base, fb
    o_ref, o_usr = o_ref / ref_peak, o_usr / usr_peak
    score = _dist_to_score(_fast_dtw(o_ref, o_usr, radius=15), scale=5.0)
    fb = []
    if dur_ratio < 0.65:
        fb.append("Nói quá nhanh — thử chậm lại để rõ từng từ"); score *= 0.8
    elif dur_ratio > 1.5:
        fb.append("Nói hơi chậm — thử nói tự nhiên hơn"); score *= 0.9
    if not fb:
        fb.append("Nhịp điệu tự nhiên, tốc độ phù hợp" if score >= 80
                   else "Nhịp điệu tạm ổn, cần đều hơn" if score >= 60
                   else "Cần cải thiện nhịp điệu — nghe mẫu và bắt chước tốc độ")
    return float(np.clip(score, 0, 100)), "; ".join(fb)


def evaluate_pronunciation(user_audio_path: Path, reference_id: str) -> Dict:
    ds_dir = get_dataset_dir()
    ref_wav = ds_dir / f"{reference_id}.wav"
    if not ref_wav.exists():
        raise FileNotFoundError(f"Không tìm thấy audio mẫu: {reference_id}.wav")
    ref_txt = ds_dir / f"{reference_id}.txt"
    ref_text = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else ""

    y_ref, y_user = _load_and_trim(ref_wav), _load_and_trim(user_audio_path)

    mfcc_ref = _extract_mfcc(y_ref)
    mfcc_user = _extract_mfcc(y_user)
    energy_ref = _extract_energy(y_ref)
    energy_user = _extract_energy(y_user)

    mfcc_sc = _dist_to_score(_fast_dtw(mfcc_ref, mfcc_user, 20), 12.0)
    pitch_sc, pitch_fb = _pitch_score(_extract_pitch(y_ref), _extract_pitch(y_user))
    rhythm_sc, rhythm_fb = _rhythm_score(y_ref, y_user)
    energy_sc = _dist_to_score(_fast_dtw(energy_ref, energy_user, 15), 8.0)

    W = {"mfcc": 0.30, "pitch": 0.40, "rhythm": 0.15, "energy": 0.15}
    overall = mfcc_sc * W["mfcc"] + pitch_sc * W["pitch"] + rhythm_sc * W["rhythm"] + energy_sc * W["energy"]

    grade = ("Xuất sắc 🌟" if overall >= 85 else "Tốt 👍" if overall >= 70
             else "Khá 📝" if overall >= 55 else "Trung bình ⚠️" if overall >= 40 else "Cần cố gắng 💪")

    fb = [pitch_fb, rhythm_fb]
    if mfcc_sc < 50: fb.append("Phát âm nguyên âm/phụ âm cần rõ ràng hơn")
    if mfcc_sc >= 75 and pitch_sc >= 75: fb.append("Phát âm và ngữ điệu khá tốt!")

    return {
        "overall_score": round(overall, 1), "grade": grade,
        "mfcc_score": round(mfcc_sc, 1), "pitch_score": round(pitch_sc, 1),
        "rhythm_score": round(rhythm_sc, 1), "energy_score": round(energy_sc, 1),
        "reference_text": ref_text, "reference_id": reference_id,
        "feedback": " | ".join(fb),
        "details": {"weights": W,
                    "user_duration_sec": round(len(y_user) / SAMPLE_RATE, 2),
                    "ref_duration_sec": round(len(y_ref) / SAMPLE_RATE, 2)},
    }
