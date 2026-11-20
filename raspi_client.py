#!/usr/bin/env python3
"""Raspberry Pi Client Simulator — Mô phỏng Raspberry Pi + INMP441.

Script này mô phỏng những gì sẽ chạy trên Raspberry Pi thực tế:
  1. Khởi tạo microphone INMP441 qua I2S
  2. Thu âm giọng nói
  3. Gửi file audio lên Flask server qua HTTP
  4. Hiển thị kết quả sàng lọc Parkinson

Trên máy tính: sử dụng sounddevice thay thế I2S driver.
Trên Raspberry Pi thực: thay bằng pyaudio + I2S overlay.

Usage:
  python raspi_client.py                    # Thu âm 5 giây rồi gửi phân tích
  python raspi_client.py --duration 10      # Thu âm 10 giây
  python raspi_client.py --file sample.wav  # Phân tích file có sẵn
  python raspi_client.py --server http://192.168.1.100:5000
"""
import argparse
import json
import os
import sys
import time
import wave
import struct
import tempfile
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
DEFAULT_SERVER = "http://127.0.0.1:5000"
SAMPLE_RATE = 16000
CHANNELS = 1
BIT_DEPTH = 16   # 16-bit (INMP441 outputs 24-bit, downsampled)
CHUNK_SIZE = 1024


def print_header():
    """Print startup banner (simulating Pi boot)."""
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║   🍓 Raspberry Pi 4 — Parkinson Voice Screener             ║
║   Microphone: INMP441 (I2S MEMS)                           ║
║   Version: 2.0  |  Sample Rate: 16kHz  |  24-bit I2S       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    boot_msgs = [
        "[BOOT] Initializing Raspberry Pi 4 Model B...",
        "[I2S]  Loading bcm2835-i2s driver...",
        "[I2S]  INMP441 detected: GPIO18(BCLK) GPIO19(LRCLK) GPIO20(DIN)",
        "[I2S]  Sample rate: 16000 Hz, Channels: 1, Bit depth: 24→16",
        "[MIC]  INMP441 MEMS — SNR: 65 dBA, Sensitivity: -26 dBFS",
        "[NET]  Connecting to analysis server...",
    ]
    for msg in boot_msgs:
        time.sleep(0.15)
        print(f"  {msg}")
    print()


def record_audio(duration: int, output_path: Path) -> Path:
    """Thu âm qua microphone (mô phỏng INMP441 I2S).

    Trên máy tính: dùng sounddevice hoặc pyaudio.
    Trên Raspberry Pi: thay bằng I2S capture.
    """
    print(f"[REC]  🎙️  Recording {duration}s of sustained vowel 'Aaa...'")
    print(f"[REC]  Output: {output_path}")
    print(f"[REC]  {'─' * 50}")

    try:
        import sounddevice as sd
        print("[REC]  Using sounddevice (simulating INMP441)")
        print(f"[REC]  ● Recording... speak now!", flush=True)

        # Countdown display
        audio_data = sd.rec(
            int(SAMPLE_RATE * duration),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )

        for i in range(duration):
            remaining = duration - i
            bar = "█" * (duration - remaining) + "░" * remaining
            print(f"\r[REC]  [{bar}] {remaining}s remaining  ", end="", flush=True)
            time.sleep(1)
        print(f"\r[REC]  [{'█' * duration}] Done!            ")

        sd.wait()

        # Save WAV
        with wave.open(str(output_path), "w") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        file_size = output_path.stat().st_size / 1024
        print(f"[REC]  ■ Saved: {file_size:.1f} KB")
        return output_path

    except ImportError:
        print("[WARN] sounddevice not installed. Using pyaudio fallback...")
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            print(f"[REC]  ● Recording... speak now!", flush=True)
            frames = []
            for i in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                # Progress
                progress = i / int(SAMPLE_RATE / CHUNK_SIZE * duration)
                if int(progress * 10) > int((i - 1) / int(SAMPLE_RATE / CHUNK_SIZE * duration) * 10):
                    bar_len = int(progress * 30)
                    print(f"\r[REC]  [{'█' * bar_len}{'░' * (30 - bar_len)}] {int(progress*100)}%", end="", flush=True)

            print(f"\r[REC]  [{'█' * 30}] 100% — Done!")
            stream.stop_stream()
            stream.close()
            pa.terminate()

            with wave.open(str(output_path), "w") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            file_size = output_path.stat().st_size / 1024
            print(f"[REC]  ■ Saved: {file_size:.1f} KB")
            return output_path

        except ImportError:
            print("[ERR]  Neither sounddevice nor pyaudio available!")
            print("[ERR]  Install: pip install sounddevice  OR  pip install pyaudio")
            sys.exit(1)


def send_to_server(audio_path: Path, server_url: str) -> dict:
    """Gửi file audio lên Flask server để phân tích Parkinson."""
    import urllib.request
    import mimetypes

    url = f"{server_url}/api/parkinson/screen"
    print(f"\n[API]  Sending audio to: {url}")
    print(f"[API]  File: {audio_path.name} ({audio_path.stat().st_size / 1024:.1f} KB)")

    # Build multipart form data
    boundary = "----RaspberryPiBoundary" + str(int(time.time()))
    content_type = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'.encode())
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    with open(audio_path, "rb") as f:
        body.extend(f.read())
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                print("[API]  ✅ Analysis complete!")
                return result
            else:
                print(f"[API]  ❌ Server error: {result.get('error')}")
                return result
    except Exception as e:
        print(f"[API]  ❌ Connection failed: {e}")
        print(f"[API]  Make sure the server is running: python web_app.py")
        return {"ok": False, "error": str(e)}


def display_results(data: dict):
    """Hiển thị kết quả sàng lọc Parkinson."""
    if not data.get("ok"):
        print(f"\n❌ Error: {data.get('error', 'Unknown error')}")
        return

    score = data.get("risk_score", 0)
    level = data.get("risk_level_vi", "N/A")
    indicators = data.get("indicators", [])
    features = data.get("features", {})

    # Score gauge (ASCII art)
    gauge_len = 40
    filled = int(score / 100 * gauge_len)
    if score >= 60:
        color_start, color_end = "\033[91m", "\033[0m"  # red
    elif score >= 30:
        color_start, color_end = "\033[93m", "\033[0m"  # yellow
    else:
        color_start, color_end = "\033[92m", "\033[0m"  # green

    print(f"\n{'═' * 60}")
    print(f"  KẾT QUẢ SÀNG LỌC PARKINSON")
    print(f"{'═' * 60}")
    print(f"\n  Điểm nguy cơ: {color_start}{score}%{color_end}  {level}")
    print(f"  [{color_start}{'█' * filled}{'░' * (gauge_len - filled)}{color_end}]")

    # Features
    print(f"\n{'─' * 60}")
    print(f"  ĐẶC TRƯNG GIỌNG NÓI")
    print(f"{'─' * 60}")
    key_features = [
        ("F0 Mean", f"{features.get('f0_mean', 0):.1f} Hz"),
        ("F0 Range", f"{features.get('f0_range', 0):.1f} Hz"),
        ("Jitter (local)", f"{features.get('jitter_local', 0):.6f}"),
        ("Jitter (RAP)", f"{features.get('jitter_rap', 0):.6f}"),
        ("Shimmer (local)", f"{features.get('shimmer_local', 0):.6f}"),
        ("Shimmer (dB)", f"{features.get('shimmer_db', 0):.4f}"),
        ("HNR", f"{features.get('hnr', 0):.1f} dB"),
        ("NHR", f"{features.get('nhr', 0):.6f}"),
        ("PPE", f"{features.get('ppe', 0):.4f}"),
        ("Speech Rate", f"{features.get('speech_rate', 0):.1f} onset/s"),
        ("Pause Ratio", f"{features.get('pause_ratio', 0):.2%}"),
    ]
    for name, val in key_features:
        print(f"  {name:<20s} {val}")

    # Indicators
    if indicators:
        print(f"\n{'─' * 60}")
        print(f"  DẤU HIỆU PHÁT HIỆN ({len(indicators)})")
        print(f"{'─' * 60}")
        for ind in indicators:
            icon = "🔴" if ind["severity"] == "cao" else "🟡"
            print(f"  {icon} {ind['feature']}")
            print(f"     {ind['detail']}")

    # Advice
    print(f"\n{'─' * 60}")
    print(f"  💡 {data.get('advice', '')}")
    print(f"{'─' * 60}")
    print(f"  ⚠️  {data.get('disclaimer', '')}")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Parkinson Voice Screener (Simulator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python raspi_client.py                          # Thu âm 5s, gửi phân tích
  python raspi_client.py --duration 10            # Thu âm 10 giây
  python raspi_client.py --file my_voice.wav      # Phân tích file có sẵn
  python raspi_client.py --server http://pi:5000  # Server khác
        """,
    )
    parser.add_argument("--duration", type=int, default=5, help="Thời gian thu âm (giây)")
    parser.add_argument("--file", type=str, help="File audio có sẵn (bỏ qua thu âm)")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="URL Flask server")
    parser.add_argument("--output", type=str, help="Lưu file WAV ra đường dẫn cụ thể")
    args = parser.parse_args()

    print_header()

    # Check server connection
    print(f"[NET]  Server: {args.server}")
    try:
        import urllib.request
        urllib.request.urlopen(f"{args.server}/api/health", timeout=5)
        print("[NET]  ✅ Server connected!")
    except Exception:
        print("[NET]  ⚠️  Server not reachable. Results may fail.")

    if args.file:
        # Use existing file
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"[ERR]  File not found: {audio_path}")
            sys.exit(1)
        print(f"\n[FILE] Using existing audio: {audio_path}")
    else:
        # Record
        if args.output:
            audio_path = Path(args.output)
        else:
            audio_path = Path(tempfile.mktemp(suffix=".wav"))
        record_audio(args.duration, audio_path)

    # Send and analyze
    result = send_to_server(audio_path, args.server)
    display_results(result)

    # Cleanup temp file
    if not args.file and not args.output:
        try:
            audio_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
