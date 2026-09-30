#!/bin/bash
# ==============================================================================
# Script khởi chạy ứng dụng ParkiScan trên Raspberry Pi 4 + Microphone INMP441
# Đề tài 19: Voice-based Parkinson's Disease Screening
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/parkinson-voice-screening"

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

echo "🚀 Đang khởi động giao diện Desktop ParkiScan..."
python3 app.py
