#!/usr/bin/env python3
"""
raspberry_pi/parkinson_inference.py
Parkinson's Disease Voice Screening — Raspberry Pi 4 + INMP441 Deployment

Phần cứng:
  - Microphone: INMP441 (I2S MEMS, 24-bit, 16-44.1 kHz)
  - Processor : Raspberry Pi 4 Model B (4GB RAM)
  - Interface : I2S digital audio
  - GPIO      : BCM18 (BCK), BCM19 (LRCLK), BCM20 (DIN)

Pipeline:
    INMP441 → I2S → sounddevice → preprocess → features → SVM → result

Cách dùng trên Raspberry Pi:
    python parkinson_inference.py                 # ghi âm 5 giây rồi predict
    python parkinson_inference.py --file audio.wav  # predict từ file
    python parkinson_inference.py --server        # chạy như HTTP inference server

Yêu cầu cài đặt trên Raspberry Pi:
    sudo apt-get install -y libportaudio2 libsndfile1 python3-numpy python3-scipy
    pip3 install librosa sounddevice soundfile joblib scikit-learn

Cách thiết lập I2S INMP441 trên Raspberry Pi:
    # Thêm vào /boot/config.txt:
    dtoverlay=i2s-mems-mic-overlay
    # Hoặc dùng custom dtoverlay cho INMP441:
    dtoverlay=adau1977-adc

Spec §18 — Raspberry Pi deployment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

# ─── Đường dẫn ────────────────────────────────────────────────────────────────
# Script này nằm trong raspberry_pi/, project root là cha của nó
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_MODEL_PATH  = _PROJECT_DIR / "models" / "parkinson_classifier.joblib"

sys.path.insert(0, str(_PROJECT_DIR))

# ─── Hằng số ─────────────────────────────────────────────────────────────────
SR           = 16_000   # Sample rate (Hz) — khớp với INMP441 native
REC_DURATION = 5        # Giây ghi âm
CHANNELS     = 1        # Mono
DEVICE_NAME  = "inmp441"   # Tên thiết bị ALSA (tùy cấu hình)


# ══════════════════════════════════════════════════════════════════════════════
#  RECORDING (INMP441 via I2S)
# ══════════════════════════════════════════════════════════════════════════════

def record_from_inmp441(
    duration: int = REC_DURATION,
    sr: int = SR,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Ghi âm từ INMP441 qua I2S trên Raspberry Pi.

    INMP441 specs:
      - SNR: 61 dBA, Sensitivity: -26 dBFS
      - Interface: I2S (Left channel, Right channel = zero)
      - Bit depth: 24-bit (right-padded in 32-bit frame)

    Parameters
    ----------
    duration  : thời gian ghi (giây)
    sr        : sample rate (Hz)
    save_path : nếu None, tạo file tạm

    Returns
    -------
    Path tới file WAV đã ghi
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError(
            "Cần cài: pip3 install sounddevice soundfile numpy\n"
            "Và: sudo apt-get install -y libportaudio2"
        )

    print(f"🎙️  Đang ghi âm {duration} giây từ INMP441 (I2S)...")
    print("   Nói 'Aaaa' kéo dài vào microphone...")

    # Ghi âm
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=CHANNELS,
        dtype="float32",
        device=None,   # None = default device; thay bằng ID INMP441 nếu cần
    )
    sd.wait()
    audio = audio.flatten()

    # Chuẩn hóa (INMP441 24-bit có thể cần scale)
    peak = abs(audio).max()
    if peak > 0:
        audio = audio / peak

    # Lưu file
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_path = Path(tmp.name)
        tmp.close()

    sf.write(str(save_path), audio, sr)
    print(f"   ✅ Đã lưu: {save_path} ({len(audio)/sr:.1f}s)")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE (tối ưu cho Raspberry Pi)
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Tải model SVM (joblib). Chỉ dùng model đã train, KHÔNG train trên Pi."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model không tìm thấy: {_MODEL_PATH}\n"
            "Hãy train model trên máy PC trước:\n"
            "  POST /api/parkinson/train\n"
            "Rồi copy models/parkinson_classifier.joblib sang Raspberry Pi."
        )
    import joblib
    return joblib.load(_MODEL_PATH)


def predict_from_audio(audio_path: Path, bundle=None) -> Dict:
    """
    Suy luận Parkinson từ file audio.

    Được tối ưu cho Raspberry Pi:
      - Không load lại model mỗi lần (dùng bundle đã load)
      - Minimal memory footprint
      - Xử lý nhanh (<1 giây trên Pi 4)

    Parameters
    ----------
    audio_path : đường dẫn file WAV
    bundle     : model bundle đã load (tùy chọn, để tái sử dụng)

    Returns
    -------
    dict kết quả prediction
    """
    t0 = time.time()

    if bundle is None:
        bundle = load_model()

    # Import các module trích xuất đặc trưng
    from audio_project.parkinson import (
        preprocess_audio_parkinson,
        extract_parkinson_features,
        _AUDIO_TO_UCI,
    )
    import pandas as pd

    # Preprocess
    y, sr = preprocess_audio_parkinson(audio_path, sr=SR)

    # Trích xuất đặc trưng
    audio_feats = extract_parkinson_features(y, sr)

    # Ánh xạ sang UCI feature space
    pipeline     = bundle["pipeline"]
    feat_names   = bundle["feature_names"]
    dataset_type = bundle.get("dataset_type", "uci")
    model_name   = bundle["model_name"]

    if dataset_type == "uci":
        row = {}
        for audio_key, uci_key in _AUDIO_TO_UCI.items():
            row[uci_key] = audio_feats.get(audio_key, 0.0)
        for col in feat_names:
            if col not in row:
                row[col] = 0.0
        X = pd.DataFrame([{col: row[col] for col in feat_names}])
    else:
        X = pd.DataFrame([{col: audio_feats.get(col, 0.0) for col in feat_names}])

    pred  = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0][1])
    label = "Parkinson" if pred == 1 else "Control"

    elapsed = time.time() - t0

    return {
        "prediction"   : label,
        "probability"  : round(proba, 4),
        "risk_pct"     : round(proba * 100, 2),
        "model"        : model_name,
        "inference_ms" : round(elapsed * 1000, 1),
        "hardware"     : {
            "device"      : "INMP441 (I2S MEMS)",
            "processor"   : "Raspberry Pi 4",
            "sample_rate" : f"{SR} Hz",
            "bit_depth"   : "24-bit → float32",
        },
        "features_summary": {
            "f0_mean"      : round(audio_feats.get("f0_mean", 0.0), 2),
            "jitter_local" : round(audio_feats.get("jitter_local", 0.0), 6),
            "shimmer_local": round(audio_feats.get("shimmer_local", 0.0), 6),
            "hnr"          : round(audio_feats.get("hnr", 0.0), 2),
            "ppe"          : round(audio_feats.get("ppe", 0.0), 4),
            "duration_sec" : round(audio_feats.get("duration_sec", 0.0), 2),
        },
        "disclaimer": (
            "Research and preliminary screening only. "
            "Not a medical diagnostic tool."
        ),
    }


def print_result(result: Dict) -> None:
    """In kết quả ra console."""
    print("\n" + "=" * 52)
    print("  🧠  PARKINSON VOICE SCREENING RESULT")
    print("=" * 52)
    label = result["prediction"]
    prob  = result["risk_pct"]
    if label == "Parkinson":
        print(f"  ⚠️  PREDICTION : {label}")
        print(f"  🔴 PROBABILITY: {prob:.1f}%")
    else:
        print(f"  ✅  PREDICTION : {label}")
        print(f"  🟢 PROBABILITY: {prob:.1f}%")
    print(f"  🤖  MODEL     : {result['model']}")
    print(f"  ⚡  LATENCY   : {result['inference_ms']} ms")
    print("-" * 52)
    print("  📊  Key Features:")
    fs = result.get("features_summary", {})
    for k, v in fs.items():
        print(f"      {k:<16} : {v}")
    print("-" * 52)
    print(f"  ⚠️  {result['disclaimer']}")
    print("=" * 52 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP SERVER (optional — cho phép gọi từ web app)
# ══════════════════════════════════════════════════════════════════════════════

def run_server(host: str = "0.0.0.0", port: int = 5001) -> None:
    """
    Chạy HTTP inference server nhỏ trên Raspberry Pi.

    Endpoint:
        POST /predict   — body: multipart/form-data với field 'file'
        GET  /health    — kiểm tra trạng thái

    Dùng Python built-in http.server để không cần Flask trên Pi.
    """
    import http.server
    import socketserver
    import io
    import cgi

    bundle = load_model()
    print(f"🚀 Parkinson Inference Server @ http://{host}:{port}")
    print(f"   Model: {bundle['model_name']}")
    print(f"   POST /predict  — upload audio file")
    print(f"   GET  /health   — check status")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress default access log

        def do_GET(self):
            if self.path == "/health":
                resp = json.dumps({
                    "ok": True,
                    "model": bundle["model_name"],
                    "hardware": "INMP441 + Raspberry Pi 4",
                })
                self._send(200, resp)
            else:
                self._send(404, '{"ok":false,"error":"not found"}')

        def do_POST(self):
            if self.path == "/predict":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body   = self.rfile.read(length)
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.write(body)
                    tmp.close()
                    result = predict_from_audio(Path(tmp.name), bundle=bundle)
                    os.unlink(tmp.name)
                    self._send(200, json.dumps(result))
                except Exception as e:
                    self._send(500, json.dumps({"ok": False, "error": str(e)}))
            else:
                self._send(404, '{"ok":false,"error":"not found"}')

        def _send(self, code, body):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)

    with socketserver.TCPServer((host, port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n⏹  Server dừng.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parkinson Voice Screening — INMP441 + Raspberry Pi 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parkinson_inference.py                     # record 5s then predict
  python parkinson_inference.py --file voice.wav    # predict from file
  python parkinson_inference.py --duration 8        # record 8 seconds
  python parkinson_inference.py --server            # run HTTP server on :5001
  python parkinson_inference.py --server --port 8080
        """,
    )
    parser.add_argument("--file",     "-f", type=str, help="Path to audio file")
    parser.add_argument("--duration", "-d", type=int, default=REC_DURATION,
                        help=f"Recording duration in seconds (default: {REC_DURATION})")
    parser.add_argument("--server",   "-s", action="store_true",
                        help="Run as HTTP inference server")
    parser.add_argument("--host",     type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port",     type=int, default=5001,
                        help="Server port (default: 5001)")
    parser.add_argument("--json",     action="store_true",
                        help="Output result as JSON")
    args = parser.parse_args()

    if args.server:
        run_server(args.host, args.port)
        return

    # Load model
    try:
        bundle = load_model()
        print(f"Model loaded: {bundle['model_name']} "
              f"(AUC={bundle['metrics'].get('roc_auc', '?'):.3f})")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Get audio
    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"ERROR: File not found: {audio_path}")
            sys.exit(1)
    else:
        # Record from INMP441
        tmp_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            audio_path = record_from_inmp441(
                duration=args.duration,
                sr=SR,
                save_path=tmp_path,
            )
        except Exception as e:
            print(f"ERROR recording: {e}")
            sys.exit(1)

    # Predict
    try:
        result = predict_from_audio(audio_path, bundle=bundle)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)
    except Exception as e:
        print(f"ERROR predicting: {e}")
        sys.exit(1)
    finally:
        # Cleanup temp file if we recorded it
        if not args.file and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
