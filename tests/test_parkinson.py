"""Tests cho Parkinson module, Flask API và Raspberry Pi inference.

Chạy:  python -m pytest tests -q
"""
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_project import parkinson as pk  # noqa: E402

SR = pk.SR


def _synthetic_vowel(f0=150.0, seconds=2.0, sr=SR, jitter=0.003, seed=0):
    """Chuỗi xung thanh môn có harmonics — đủ giống nguyên âm để Praat bắt được pitch."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = np.zeros(n)
    t = 0.0
    while t < seconds:
        idx = int(t * sr)
        if idx < n:
            y[idx] = 1.0
        t += (1.0 / f0) * (1 + rng.normal(0, jitter))
    kernel = np.exp(-np.arange(int(0.004 * sr)) / (0.0008 * sr))
    y = np.convolve(y, kernel)[:n]
    y += rng.normal(0, 0.005, n)
    return (y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture(scope="module")
def vowel_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "vowel.wav"
    sf.write(path, _synthetic_vowel(), SR)
    return path


@pytest.fixture(scope="module")
def silence_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "silence.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    return path


# ── Dataset & anti-leakage ───────────────────────────────────────────────────

def test_uci_subject_ids():
    names = pd.Series(["phon_R01_S01_1", "phon_R01_S01_6", "phon_R01_S50_2"])
    assert pk.uci_subject_ids(names).tolist() == ["S01", "S01", "S50"]


def test_uci_dataset_has_subject_groups():
    X, y, groups = pk._load_uci_dataset()
    assert list(X.columns) == pk.UCI_MODEL_COLS
    assert len(X) == len(y) == len(groups) == 195
    # mỗi subject chỉ có một nhãn
    assert (pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].nunique() == 1).all()


def test_group_split_has_no_subject_overlap():
    X, y, groups = pk._load_uci_dataset()
    tr, te = pk.group_train_test_split(X, y, groups)
    assert not set(groups.iloc[tr]) & set(groups.iloc[te])
    assert set(y.iloc[te]) == {0, 1}


# ── Feature extraction ───────────────────────────────────────────────────────

def test_feature_extraction_on_vowel(vowel_wav):
    f = pk.extract_features_from_file(vowel_wav)
    assert abs(f["f0_mean"] - 150) < 10
    for key in pk._AUDIO_TO_UCI:
        assert key in f and math.isfinite(f[key]), key
    assert f["hnr"] > 5
    assert f"mfcc_{pk.N_MFCC}_std" in f


def test_feature_extraction_on_silence_returns_nan_not_crash(silence_wav):
    f = pk.extract_features_from_file(silence_wav)
    assert math.isnan(f["jitter_local"])
    assert pk._quality_warnings(f)


def test_build_model_input_maps_to_training_columns(vowel_wav):
    f = pk.extract_features_from_file(vowel_wav)
    bundle = {"feature_names": pk.UCI_MODEL_COLS, "dataset_type": "uci"}
    X = pk.build_model_input(f, bundle)
    assert list(X.columns) == pk.UCI_MODEL_COLS
    assert X.notna().all().all()


# ── Trained model & inference ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained():
    if not pk.is_model_trained():
        pk.train_parkinson_model(verbose=False)
    return pk.load_parkinson_bundle()


def test_bundle_metadata(trained):
    assert trained["feature_names"] == pk.UCI_MODEL_COLS
    assert trained["split"]["strategy"].startswith("StratifiedGroupKFold")
    assert pk.COMPARISON_CSV_PATH.exists()
    comp = pd.read_csv(pk.COMPARISON_CSV_PATH)
    for col in ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                "Sensitivity", "Specificity", "NestedCV_ROC-AUC"]:
        assert col in comp.columns
    assert len(comp) == 4
    best = comp.loc[comp["NestedCV_ROC-AUC"].idxmax(), "Model"]
    assert trained["model_name"] == best


def test_predicted_label_matches_probability(trained, vowel_wav):
    r = pk.predict_parkinson(vowel_wav, bundle=trained)
    assert (r["prediction"] == "Parkinson") == (r["probability"] >= 0.5)


def test_predict_output_is_json_safe(trained, vowel_wav, silence_wav):
    for path in (vowel_wav, silence_wav):
        r = pk.predict_parkinson(path, bundle=trained)
        assert r["prediction"] in {"Parkinson", "Control"}
        assert 0 <= r["probability"] <= 1
        assert r["audio_info"]["sample_rate"] == SR
        json.dumps(r, allow_nan=False)


# ── Flask API ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(trained):
    import web_app
    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


def test_health_reports_parkinson_and_existing_models(client):
    h = client.get("/api/health").get_json()
    assert h["parkinson_model_ready"] is True
    for key in ("model_ready", "gender_model_ready", "cough_model_ready"):
        assert key in h


def test_predict_requires_file(client):
    r = client.post("/api/parkinson/predict")
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_predict_endpoint(client, vowel_wav):
    data = {"file": (io.BytesIO(vowel_wav.read_bytes()), "vowel.wav")}
    r = client.post("/api/parkinson/predict", data=data, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True
    assert {"prediction", "probability", "model"} <= body.keys()


def test_screen_endpoint_backward_compat(client, vowel_wav):
    data = {"file": (io.BytesIO(vowel_wav.read_bytes()), "vowel.wav")}
    body = client.post("/api/parkinson/screen", data=data,
                       content_type="multipart/form-data").get_json()
    assert body["ok"] is True and "risk_level_vi" in body


def test_model_info_endpoint(client):
    body = client.get("/api/parkinson/model-info").get_json()
    assert body["ok"] is True and len(body["comparison"]) == 4


# ── Raspberry Pi ─────────────────────────────────────────────────────────────

def test_raspi_extracts_multipart_upload(vowel_wav):
    sys.path.insert(0, str(ROOT / "raspberry_pi"))
    import parkinson_inference as pi

    payload = vowel_wav.read_bytes()
    boundary = "XyZ123"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"v.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    data, suffix = pi.extract_upload(f"multipart/form-data; boundary={boundary}", body)
    assert data == payload and suffix == ".wav"
    assert pi.extract_upload("audio/wav", payload) == (payload, ".wav")


# ── Backward compatibility ───────────────────────────────────────────────────

@pytest.mark.parametrize("mod", ["core", "gender", "emotion", "cough", "depression", "pronunciation"])
def test_existing_modules_still_import(mod):
    __import__(f"audio_project.{mod}")
