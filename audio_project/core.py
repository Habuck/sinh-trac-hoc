"""Core: config, preprocessing, features, training, inference, pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import joblib, librosa, numpy as np, pandas as pd, soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
SR = 16000
DUR = 5
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP = 512
RAND = 42
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Preprocessing ───────────────────────────────────────────────────────
def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def preprocess_audio(y: np.ndarray, sr: int = SR, dur: int = DUR) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(y, top_db=20)
    return librosa.util.fix_length(trimmed, size=sr * dur)


def save_audio(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)


def iter_audio_files(base: Path = RAW_DATA_DIR) -> list[Path]:
    return [p for p in base.rglob("*") if p.suffix.lower() in AUDIO_EXT] if base.exists() else []


# ── Features ────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    sr: int = SR
    n_mfcc: int = N_MFCC
    n_mels: int = N_MELS
    n_fft: int = N_FFT
    hop: int = HOP


def extract_features(y: np.ndarray, c: Cfg = None) -> Dict[str, float]:
    c = c or Cfg()
    _m = lambda v: float(np.mean(v))
    mfcc = librosa.feature.mfcc(y=y, sr=c.sr, n_mfcc=c.n_mfcc)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=c.sr, n_fft=c.n_fft, hop_length=c.hop, n_mels=c.n_mels), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=c.sr, n_fft=c.n_fft, hop_length=c.hop)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y, frame_length=c.n_fft, hop_length=c.hop)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=c.sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    return {"mfcc_mean": _m(mfcc), "mel_mean": _m(mel), "chroma_mean": _m(chroma),
            "zcr_mean": _m(zcr), "rms_mean": _m(rms), "f0_mean": _m(f0)}


def build_feature_table(proc_dir: Path = PROCESSED_DATA_DIR, c: Cfg = None) -> pd.DataFrame:
    c = c or Cfg()
    rows = []
    for ld in sorted(proc_dir.iterdir() if proc_dir.exists() else []):
        if not ld.is_dir():
            continue
        for ap in sorted(ld.glob("*.wav")):
            row = extract_features(preprocess_audio(load_audio(ap, c.sr), c.sr), c)
            row["label"] = ld.name
            row["file_path"] = str(ap)
            rows.append(row)
    return pd.DataFrame(rows)


def save_feature_table(df: pd.DataFrame, path: Path = None) -> Path:
    path = path or (FEATURES_DIR / "features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["label", "file_path"]), df["label"]


# ── Training ────────────────────────────────────────────────────────────
@dataclass
class TrainResult:
    model_path: Path
    report: str
    train_size: int
    test_size: int


def train_model(df: pd.DataFrame, path: Path = None) -> TrainResult:
    x, y = split_xy(df)
    xt, xe, yt, ye = train_test_split(
        x, y, test_size=0.2, random_state=RAND,
        stratify=y if len(y.unique()) > 1 else None)
    m = RandomForestClassifier(n_estimators=200, random_state=RAND)
    m.fit(xt, yt)
    rpt = classification_report(ye, m.predict(xe), zero_division=0)
    path = path or (MODELS_DIR / "audio_classifier.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, path)
    return TrainResult(model_path=path, report=rpt, train_size=len(xt), test_size=len(xe))


def load_model(path: Path = None):
    return joblib.load(path or (MODELS_DIR / "audio_classifier.joblib"))


# ── Inference ───────────────────────────────────────────────────────────
def predict_file(audio_path: Path, model, c: Cfg = None) -> str:
    c = c or Cfg()
    feat = extract_features(preprocess_audio(load_audio(audio_path, c.sr), c.sr), c)
    return str(model.predict(pd.DataFrame([feat]))[0])


# ── Pipeline ────────────────────────────────────────────────────────────
def prepare_processed_audio(raw: Path = RAW_DATA_DIR, proc: Path = PROCESSED_DATA_DIR) -> int:
    n = 0
    for src in iter_audio_files(raw):
        try:
            out = proc / src.relative_to(raw).with_suffix(".wav")
            save_audio(out, preprocess_audio(load_audio(src)))
            n += 1
        except Exception:
            continue
    return n


def run_training_pipeline() -> TrainResult:
    ft = build_feature_table()
    if ft.empty:
        raise ValueError("No processed audio. Put files in data/raw/<label>/.")
    save_feature_table(ft)
    return train_model(ft)
