#!/usr/bin/env python3
"""
raspberry_pi/parkinson_inference.py
Parkinson's Disease Voice Screening — Raspberry Pi 4 + INMP441 Deployment (spec §18)

Phần cứng:
  - Microphone: INMP441 (I2S MEMS, 24-bit, 16-44.1 kHz)
  - Processor : Raspberry Pi 4 Model B (4GB RAM)
  - GPIO      : BCM18 (SCK/BCLK), BCM19 (WS/LRCLK), BCM20 (SD/DIN), L/R → GND

Pipeline (chỉ inference, KHÔNG train trên Pi):
    INMP441 → I2S → sounddevice → preprocess → Praat/Librosa features → model.joblib → result

Cách dùng:
    python parkinson_inference.py                     # ghi âm 5 giây rồi predict
    python parkinson_inference.py --file audio.wav    # predict từ file
    python parkinson_inference.py --list-devices      # liệt kê thiết bị thu âm (tìm INMP441)
    python parkinson_inference.py --device 2          # ghi âm từ thiết bị số 2
    python parkinson_inference.py --server            # HTTP inference server :5001

Cài đặt trên Raspberry Pi OS (64-bit):
    sudo apt-get install -y libportaudio2 libsndfile1
    pip3 install numpy pandas librosa soundfile sounddevice joblib scikit-learn praat-parselmouth

Bật I2S cho INMP441 — thêm vào /boot/firmware/config.txt rồi reboot:
    dtparam=i2s=on
    dtoverlay=googlevoicehat-soundcard

Model phải được train trên PC (POST /api/parkinson/train) rồi copy
models/parkinson_classifier.joblib sang cùng cấu trúc thư mục trên Pi.
Phiên bản scikit-learn trên Pi phải trùng với PC để load được joblib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from audio_project.parkinson import (  # noqa: E402
    PARKINSON_MODEL_PATH,
    SR,
    load_parkinson_bundle,
    predict_parkinson,
)

REC_DURATION = 5   # giây
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ══════════════════════════════════════════════════════════════════════════════
#  RECORDING (INMP441 via I2S)
# ══════════════════════════════════════════════════════════════════════════════

def list_input_devices() -> None:
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}  (in={dev['max_input_channels']}, "
                  f"default_sr={int(dev['default_samplerate'])})")


def record_from_inmp441(duration: int = REC_DURATION, sr: int = SR,
                        device: Optional[int | str] = None,
                        save_path: Optional[Path] = None) -> Path:
    """
    Ghi âm từ INMP441 qua I2S. INMP441 chỉ xuất dữ liệu ở một kênh (L/R → GND = kênh trái),
    nên ghi stereo nếu driver yêu cầu rồi lấy kênh có năng lượng lớn hơn.
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise ImportError("Cần cài: pip3 install sounddevice soundfile numpy "
                          "và sudo apt-get install -y libportaudio2")

    info = sd.query_devices(device, "input")
    channels = min(2, int(info["max_input_channels"])) or 1

    print(f"🎙️  Ghi âm {duration}s từ '{info['name']}' ({channels} kênh, {sr} Hz)...")
    print("   Hãy ngân 'Aaaa' đều giọng vào microphone...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels,
                   dtype="float32", device=device)
    sd.wait()
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio[:, int(np.argmax(np.abs(audio).mean(axis=0)))]
    audio = audio.reshape(-1)

    if save_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_path = Path(tmp.name)
    sf.write(str(save_path), audio, sr)
    print(f"   ✅ Đã lưu: {save_path} ({len(audio) / sr:.1f}s)")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_model() -> Dict:
    """Tải model bundle một lần khi khởi động."""
    try:
        return load_parkinson_bundle()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Không tìm thấy model: {PARKINSON_MODEL_PATH}\n"
            "Train trên PC (POST /api/parkinson/train) rồi copy file joblib sang Raspberry Pi."
        )


def predict_from_audio(audio_path: Path, bundle: Dict) -> Dict:
    """Dùng chung pipeline với web app (audio_project.parkinson.predict_parkinson)."""
    t0 = time.perf_counter()
    result = predict_parkinson(audio_path, bundle=bundle)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["hardware"] = {
        "device": "INMP441 (I2S MEMS)",
        "processor": "Raspberry Pi 4",
        "sample_rate": f"{SR} Hz",
    }
    return result


def print_result(result: Dict) -> None:
    is_pd = result["prediction"] == "Parkinson"
    print("\n" + "=" * 56)
    print("  🧠  PARKINSON VOICE SCREENING RESULT")
    print("=" * 56)
    print(f"  {'⚠️ ' if is_pd else '✅'} PREDICTION : {result['prediction']}")
    print(f"  {'🔴' if is_pd else '🟢'} PROBABILITY: {result['risk_pct']:.1f}%")
    print(f"  🤖 MODEL      : {result['model']}")
    print(f"  ⚡ LATENCY    : {result['inference_ms']} ms")
    print("-" * 56)
    print("  📊 Key Features:")
    for k, v in result.get("features_summary", {}).items():
        print(f"      {k:<16} : {'—' if v is None else v}")
    for w in result.get("warnings", []):
        print(f"  ⚠️  {w}")
    print("-" * 56)
    print(f"  {result['disclaimer']}")
    print("=" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP SERVER — chỉ dùng thư viện chuẩn, không cần Flask trên Pi
# ══════════════════════════════════════════════════════════════════════════════

def extract_upload(content_type: str, body: bytes) -> Tuple[bytes, str]:
    """
    Lấy dữ liệu audio từ request: multipart/form-data (field 'file')
    hoặc raw body (curl --data-binary @voice.wav). Trả về (bytes, suffix).
    """
    if content_type.startswith("multipart/form-data"):
        from email.parser import BytesParser
        from email.policy import HTTP

        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
        )
        for part in msg.iter_parts():
            if part.get_param("name", header="content-disposition") == "file":
                suffix = Path(part.get_filename() or "audio.wav").suffix or ".wav"
                return part.get_payload(decode=True) or b"", suffix
        raise ValueError("Thiếu field 'file' trong multipart form-data.")
    return body, ".wav"


def run_server(host: str = "0.0.0.0", port: int = 5001) -> None:
    """
    Endpoint:
        GET  /health   — trạng thái model
        POST /predict  — multipart field 'file' hoặc raw WAV body
    """
    import http.server

    bundle = load_model()
    print(f"🚀 Parkinson Inference Server @ http://{host}:{port}  (model: {bundle['model_name']})")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": bundle["model_name"],
                                 "hardware": "INMP441 + Raspberry Pi 4"})
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send(404, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self._send(400, {"ok": False, "error": "Body rỗng hoặc quá lớn."})
                return
            tmp_path = None
            try:
                data, suffix = extract_upload(self.headers.get("Content-Type", ""),
                                              self.rfile.read(length))
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                self._send(200, {"ok": True, **predict_from_audio(Path(tmp_path), bundle)})
            except ValueError as e:
                self._send(400, {"ok": False, "error": str(e)})
            except Exception as e:
                self._send(500, {"ok": False, "error": str(e)})
            finally:
                if tmp_path:
                    os.unlink(tmp_path)

        def _send(self, code: int, payload: Dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with http.server.ThreadingHTTPServer((host, port), Handler) as httpd:
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
    )
    parser.add_argument("--file", "-f", type=str, help="Đường dẫn file audio")
    parser.add_argument("--duration", "-d", type=int, default=REC_DURATION,
                        help=f"Thời gian ghi âm (giây, mặc định {REC_DURATION})")
    parser.add_argument("--device", type=str, default=None,
                        help="ID hoặc tên thiết bị thu âm (xem --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="Liệt kê thiết bị thu âm")
    parser.add_argument("--server", "-s", action="store_true", help="Chạy HTTP inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--json", action="store_true", help="In kết quả dạng JSON")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return
    if args.server:
        run_server(args.host, args.port)
        return

    try:
        bundle = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if not args.json:
        auc = bundle.get("metrics", {}).get("nested_cv", {}).get("roc_auc")
        print(f"Model: {bundle['model_name']}" + (f" (nested-CV AUC={auc:.3f})" if auc is not None else ""))

    recorded = None
    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"ERROR: Không tìm thấy file: {audio_path}")
            sys.exit(1)
    else:
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        try:
            audio_path = recorded = record_from_inmp441(args.duration, SR, device=device)
        except Exception as e:
            print(f"ERROR ghi âm: {e}")
            sys.exit(1)

    try:
        result = predict_from_audio(audio_path, bundle)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_result(result)
    except Exception as e:
        print(f"ERROR predict: {e}")
        sys.exit(1)
    finally:
        if recorded is not None:
            recorded.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
