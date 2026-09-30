#!/bin/bash
# ==============================================================================
# Script khởi chạy ứng dụng ParkiScan trên Raspberry Pi 4 + Microphone INMP441
# Đề tài 19: Voice-based Parkinson's Disease Screening
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "   ParkiScan: Khởi chạy trên Raspberry Pi 4 + INMP441 Microphone"
echo "======================================================================"

# Kiểm tra ALSA audio recording tool
if ! command -v arecord &> /dev/null; then
    echo "⚠️ arecord chưa được cài đặt. Đang cài đặt alsa-utils..."
    sudo apt-get update && sudo apt-get install -y alsa-utils
fi

# Kiểm tra Python 3 và Tkinter
if ! python3 -c "import tkinter" &> /dev/null; then
    echo "⚠️ python3-tk chưa được cài đặt. Đang cài đặt..."
    sudo apt-get update && sudo apt-get install -y python3-tk
fi

# Kiểm tra DISPLAY cho giao diện GUI
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

echo "🚀 Đang khởi động giao diện Desktop ParkiScan..."
python3 app.py
