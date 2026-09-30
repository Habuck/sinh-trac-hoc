"""Flask Backend — Easy Kit Audio Detection."""
import sys, subprocess, os, shutil, tempfile, logging
from pathlib import Path

try:
    import joblib, pandas, librosa
except ImportError:
    print("=> Tự động cài đặt thư viện...", flush=True)
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    except Exception:
        pass

from flask import Flask, jsonify, request, send_from_directory

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# --- AUTO CLEANUP LEGACY FILES ---
_rm_files = [
    "BAO_CAO_NCKH-integrated.html", "BAO_CAO_NCKH.html", "index-backend-connected.html",
    "index-venv311.html", "main.html", "test-simple.html", "skilo tree.md",
    "short term dapper mosquito-html.rar", "simple_backend.py", "main.py", "train_emotion.py",
]
_rm_dirs = ["short term dapper mosquito-html", "venv311"]
for _f in _rm_files:
    try:
        os.remove(BASE_DIR / _f)
    except OSError:
        pass
for _d in _rm_dirs:
    try:
        shutil.rmtree(BASE_DIR / _d)
    except OSError:
        pass
if (BASE_DIR / "new.html").exists() and not (BASE_DIR / "index.html").exists():
    try:
        (BASE_DIR / "new.html").rename(BASE_DIR / "index.html")
    except OSError:
        pass

from audio_project.gender import (GENDER_FEATURE_COLUMNS, load_gender_model, predict_gender,
                                   predict_gender_from_file, train_gender_model)
from audio_project.core import load_model, prepare_processed_audio, run_training_pipeline, predict_file, MODELS_DIR
from audio_project.pronunciation import evaluate_pronunciation, list_sentences, get_dataset_dir
from audio_project.emotion import predict_emotion_from_file
from audio_project.cough import predict_cough, train_cough_model, load_cough_model
from audio_project.depression import screen_depression
from audio_project.parkinson import (
    screen_parkinson,
    predict_parkinson,
    train_parkinson_model,
    is_model_trained as parkinson_model_trained,
    PARKINSON_MODEL_PATH,
)
from audio_project.glucose_alarm import (
    load_glucose_model, predict_glucose_alarm, get_glucose_model_info,
    GLUCOSE_MODEL_PATH,
)

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


def _safe_load(loader):
    try:
        return loader()
    except Exception:
        return None


def _save_temp(file):
    suffix = Path(file.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    path = Path(tmp.name)
    if suffix.lower() in (".webm", ".ogg", ".m4a", ".mp4"):
        wav_path = path.with_suffix(".wav")
        try:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"
                
            subprocess.run([ffmpeg_exe, "-y", "-i", str(path), "-ar", "16000",
                            "-ac", "1", str(wav_path)],
                           capture_output=True, timeout=30, check=True)
            path.unlink(missing_ok=True)
            return wav_path
        except Exception as e:
            logging.warning("ffmpeg convert failed: %s", e)
    return path


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    from audio_project.cough import COUGH_MODEL_PATH
    return jsonify({
        "ok": True,
        "model_ready": _safe_load(load_model) is not None,
        "gender_model_ready": _safe_load(load_gender_model) is not None,
        "cough_model_ready": COUGH_MODEL_PATH.exists(),
        "parkinson_model_ready": parkinson_model_trained(),
    })


@app.route("/api/train", methods=["POST"])
def train():
    try:
        count = prepare_processed_audio()
        r = run_training_pipeline()
        return jsonify({"ok": True, "prepared_count": count, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    model = _safe_load(load_model)
    if not model:
        return jsonify({"ok": False, "error": "Chưa có model. Hãy train trước."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, "prediction": predict_file(tmp, model)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/gender/train", methods=["POST"])
def gender_train():
    try:
        r = train_gender_model()
        return jsonify({"ok": True, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/gender/predict", methods=["POST"])
def gender_predict():
    payload = request.get_json(silent=True) or {}
    missing = [k for k in GENDER_FEATURE_COLUMNS if k not in payload]
    if missing:
        return jsonify({"ok": False, "error": f"Thiếu trường: {missing}"}), 400
    model = _safe_load(load_gender_model)
    if not model:
        return jsonify({"ok": False, "error": "Chưa có gender model. Hãy train trước."}), 400
    try:
        features = {k: float(payload[k]) for k in GENDER_FEATURE_COLUMNS}
        return jsonify({"ok": True, "prediction": predict_gender(features, model=model)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/gender/predict-audio", methods=["POST"])
def gender_predict_audio():
    """Dự đoán giới tính từ file audio (tự động trích xuất đặc trưng)."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    model = _safe_load(load_gender_model)
    if not model:
        return jsonify({"ok": False, "error": "Chưa có gender model. Hãy train trước."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **predict_gender_from_file(tmp, model=model)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/pronunciation/sentences")
def pronunciation_sentences():
    try:
        limit = request.args.get("limit", 50, type=int)
        sents = list_sentences(limit=limit)
        return jsonify({"ok": True, "sentences": sents, "count": len(sents)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/pronunciation/audio/<sentence_id>")
def pronunciation_audio(sentence_id):
    try:
        return send_from_directory(get_dataset_dir(), f"{sentence_id}.wav")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 404


@app.route("/api/pronunciation/evaluate", methods=["POST"])
def pronunciation_evaluate():
    file = request.files.get("file")
    ref_id = request.form.get("reference_id", "").strip()
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    if not ref_id:
        return jsonify({"ok": False, "error": "Thiếu reference_id."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **evaluate_pronunciation(tmp, ref_id)})
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/emotion/predict", methods=["POST"])
def emotion_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **predict_emotion_from_file(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/cough/predict", methods=["POST"])
def cough_predict():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui l\u00f2ng ch\u1ecdn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **predict_cough(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/cough/train", methods=["POST"])
def cough_train():
    try:
        r = train_cough_model()
        return jsonify({"ok": True, "model_path": str(r.model_path),
                        "train_size": r.train_size, "test_size": r.test_size, "report": r.report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/depression/screen", methods=["POST"])
def depression_screen():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui l\u00f2ng ch\u1ecdn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **screen_depression(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Parkinson Voice Screening — INMP441 + Raspberry Pi
# Spec §13, §14, §15
# ══════════════════════════════════════════════════════════════════════

@app.route("/raspi")
def raspi_simulator():
    """Serve trang mô phỏng Raspberry Pi + INMP441."""
    return send_from_directory(BASE_DIR, "raspi_simulator.html")


@app.route("/api/parkinson/screen", methods=["POST"])
def parkinson_screen():
    """Backward-compat: rule-based fallback → ML nếu model có sẵn."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400
    tmp = _save_temp(file)
    try:
        return jsonify({"ok": True, **screen_parkinson(tmp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/parkinson/predict", methods=["POST"])
def parkinson_predict():
    """
    ML-based Parkinson prediction (spec §13).

    Input : multipart/form-data  →  file (audio)
    Output:
        ok          : bool
        prediction  : "Parkinson" | "Control"
        probability : float 0-1
        model       : str — model name used
        risk_pct    : float — probability as percentage
        features_summary : dict
        disclaimer  : str
    """
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400

    if not parkinson_model_trained():
        return jsonify({
            "ok": False,
            "error": "Parkinson model is not trained yet.",
        }), 400

    tmp = _save_temp(file)
    try:
        result = predict_parkinson(tmp)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/parkinson/train", methods=["POST"])
def parkinson_train():
    """
    Train Parkinson ML model on UCI dataset (spec §14).

    Output:
        ok           : bool
        model        : str
        train_size   : int
        test_size    : int
        accuracy     : float
        precision    : float
        recall       : float
        f1           : float
        roc_auc      : float
        sensitivity  : float
        specificity  : float
        model_path   : str
    """
    try:
        result = train_parkinson_model(verbose=False)
        return jsonify({
            "ok"         : True,
            "model"      : result.model_name,
            "train_size" : result.train_size,
            "test_size"  : result.test_size,
            "accuracy"   : round(result.accuracy,    4),
            "precision"  : round(result.precision,   4),
            "recall"     : round(result.recall,      4),
            "f1"         : round(result.f1,          4),
            "roc_auc"    : round(result.roc_auc,     4),
            "sensitivity": round(result.sensitivity, 4),
            "specificity": round(result.specificity, 4),
            "model_path" : result.model_path,
            "report"     : result.report,
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Glucose Alarm Detection (CNN + Mel Spectrogram) ──────────────────────────

@app.route("/api/glucose/health")
def glucose_health():
    """Kiểm tra trạng thái model phát hiện cảnh báo đường huyết thấp."""
    model_ready = GLUCOSE_MODEL_PATH.exists()
    result = {"ok": True, "model_ready": model_ready}
    if model_ready:
        result["model_file"] = GLUCOSE_MODEL_PATH.name
    return jsonify(result)


@app.route("/api/glucose/predict", methods=["POST"])
def glucose_predict():
    """
    Dự đoán xem file audio có chứa tiếng cảnh báo máy đo đường huyết (CGM) không.

    Form-data:
        file          : file audio (WAV, MP3, WebM...)
        threshold     : (optional) float 0-1, mặc định 0.9

    Response JSON:
        ok            : bool
        probability   : float  — xác suất là CGM alarm (0.0 – 1.0)
        is_alarm      : bool
        confidence    : float  — độ chắc chắn (%)
        label         : str    — "glucose_alarm" | "no_alarm"
        rms_energy    : float
        threshold_used: float
    """
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "Vui lòng chọn file audio."}), 400

    if not GLUCOSE_MODEL_PATH.exists():
        return jsonify({
            "ok": False,
            "error": "Chưa có glucose alarm model. Vui lòng đặt file .pth vào thư mục models/"
        }), 400

    threshold = request.form.get("threshold", 0.9, type=float)
    threshold = max(0.0, min(1.0, threshold))  # clamp [0, 1]

    tmp = _save_temp(file)
    try:
        result = predict_glucose_alarm(tmp, confidence_threshold=threshold)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/glucose/model-info")
def glucose_model_info():
    """Trả về thông tin chi tiết model CNN: kiến trúc, params training, metrics."""
    try:
        return jsonify({"ok": True, **get_glucose_model_info()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


def _auto_train():
    """Train missing models on startup."""
    ac = MODELS_DIR / "audio_classifier.joblib"
    gc = MODELS_DIR / "gender_classifier.joblib"
    cc = MODELS_DIR / "cough_classifier.joblib"
    if not gc.exists():
        try:
            r = train_gender_model()
            logging.info("Auto-trained gender model: %s (acc in report)", r.model_path)
        except Exception as e:
            logging.warning("Auto-train gender failed: %s", e)
    if not ac.exists():
        try:
            prepare_processed_audio()
            r = run_training_pipeline()
            logging.info("Auto-trained audio classifier: %s", r.model_path)
        except Exception as e:
            logging.warning("Auto-train audio failed: %s", e)
    if not cc.exists():
        try:
            from audio_project.cough import train_cough_model
            r = train_cough_model()
            logging.info("Auto-trained cough model: %s", r.model_path)
        except Exception as e:
            logging.warning("Auto-train cough failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _auto_train()
    app.run(host="127.0.0.1", port=5000, debug=True)
