"""
app.py — Ứng dụng Desktop chạy Local cho Nhận diện giọng nói Parkinson & Sinh trắc học.

Chủ đề 19: Voice-based Parkinson's Disease Screening & Sinh trắc học giọng nói
Phần cứng : Microphone INMP441 (I2S MEMS) + Raspberry Pi 4 (hoặc PC Microphone)
Công nghệ : Trích xuất MFCC / Jitter / Shimmer / HNR + Multi-model ML (LR/SVM/RF/GB)
            Tích hợp Nhận diện Giới tính, Cảm xúc, Tiếng ho, Âm vực từ Đồ án 2.1.

Giao diện gồm 6 tab chuyên sâu:
  🎙️ Sàng lọc & Phân tích       — Ghi âm / Tải file → Waveform → Sàng lọc Parkinson + Sinh trắc
  🧬 Sinh trắc học Giọng nói     — Nhận diện Giới tính, Cảm xúc, Tiếng ho, Hồ sơ âm vực
  📊 Phổ đặc trưng & MFCC        — Biểu đồ phân tích chuyên sâu + Spectrogram + F0 Contour
  🤖 Huấn luyện Đa Mô hình (ML)  — So sánh 4 model: LR/SVM/RF/GB + GridSearchCV + GroupKFold CV
  📈 Đánh giá & So sánh Model     — ROC Curve, Confusion Matrix, Feature Importance, Bảng metrics
  📋 Lịch sử Sàng lọc             — SQLite database, thống kê, xuất CSV
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
import tempfile
import time
import queue
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

# ─── Tkinter ─────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ─── NumPy / Matplotlib ──────────────────────────────────────────────────────
import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Audio recording & playback ──────────────────────────────────────────────
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# ─── Local modules ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from features import extract_features
from model import (
    train_model, train_all_models, predict_from_audio_features,
    is_model_trained, load_model, MODEL_PATH, RESULTS_DIR,
    UCI_FEATURE_COLS, MODEL_LABELS,
)
import biometrics
import history

# ─── Màu sắc (Dark theme hiện đại & chuyên nghiệp) ───────────────────────────
C = {
    "bg"     : "#0b0f1a",
    "bg2"    : "#111827",
    "bg3"    : "#1e2535",
    "card"   : "#161d2e",
    "pri"    : "#6366f1",
    "pri_d"  : "#4f46e5",
    "cyan"   : "#06b6d4",
    "green"  : "#10b981",
    "orange" : "#f59e0b",
    "red"    : "#ef4444",
    "pink"   : "#ec4899",
    "purple" : "#a855f7",
    "txt"    : "#e2e8f0",
    "txt2"   : "#94a3b8",
    "txt3"   : "#64748b",
    "border" : "#1e293b",
    "shadow" : "#000000",
}

FONT_H1 = ("Segoe UI", 16, "bold")
FONT_H2 = ("Segoe UI", 13, "bold")
FONT_H3 = ("Segoe UI", 11, "bold")
FONT_B  = ("Segoe UI", 10, "bold")
FONT_N  = ("Segoe UI", 10)
FONT_S  = ("Segoe UI", 9)
FONT_M  = ("Consolas", 9)

REC_DURATION  = 5          # Số giây ghi âm mặc định
SAMPLE_RATE   = 16_000     # Tần số chuẩn (16 kHz) — tương thích INMP441 & Librosa

# ─── Màu biểu đồ ─────────────────────────────────────────────────────────────
MPL_STYLE = {
    "axes.facecolor"   : C["card"],
    "figure.facecolor" : C["bg2"],
    "axes.edgecolor"   : C["border"],
    "axes.labelcolor"  : C["txt2"],
    "xtick.color"      : C["txt3"],
    "ytick.color"      : C["txt3"],
    "grid.color"       : C["border"],
    "text.color"       : C["txt"],
}
if HAS_MPL:
    plt.rcParams.update(MPL_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper widgets
# ══════════════════════════════════════════════════════════════════════════════

def card(parent, **kwargs) -> tk.Frame:
    kw = dict(bg=C["card"], relief="flat", bd=0)
    kw.update(kwargs)
    return tk.Frame(parent, **kw)


def btn(parent, text, command, color=None, **kwargs) -> tk.Button:
    bg = color or C["pri"]
    kw = dict(
        text=text, command=command, bg=bg, fg="#ffffff",
        font=FONT_B, relief="flat", bd=0, cursor="hand2",
        padx=14, pady=8, activebackground=_darken(bg, 0.8),
        activeforeground="#ffffff",
    )
    kw.update(kwargs)
    b = tk.Button(parent, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_darken(bg, 0.88)) if b["state"] != "disabled" else None)
    b.bind("<Leave>", lambda e: b.config(bg=bg) if b["state"] != "disabled" else None)
    return b


def _darken(hex_color: str, factor: float = 0.85) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#4f46e5"
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def separator(parent, **kw) -> tk.Frame:
    f = tk.Frame(parent, bg=C["border"], height=1)
    f.update_idletasks()
    return f


def _blend_alpha(hex_col: str, bg: str, a: float) -> str:
    def parse(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    fc, bc = parse(hex_col), parse(bg)
    r = int(fc[0] * a + bc[0] * (1 - a))
    g = int(fc[1] * a + bc[1] * (1 - a))
    b = int(fc[2] * a + bc[2] * (1 - a))
    return f"#{r:02x}{g:02x}{b:02x}"


# ══════════════════════════════════════════════════════════════════════════════
#  Waveform Widget — Vẽ dạng sóng âm mượt mà
# ══════════════════════════════════════════════════════════════════════════════

class WaveformWidget(tk.Canvas):
    """Canvas hiển thị dạng sóng âm thanh trực tiếp (Waveform)."""

    def __init__(self, parent, height=80, **kw):
        kw.setdefault("bg", C["bg3"])
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", C["border"])
        super().__init__(parent, height=height, **kw)
        self._audio: Optional[np.ndarray] = None
        self.bind("<Configure>", lambda e: self.redraw())

    def set_audio(self, audio: Optional[np.ndarray]):
        self._audio = audio
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        mid_y = h / 2
        # Đường trục giữa
        self.create_line(0, mid_y, w, mid_y, fill=C["border"], dash=(2, 3))

        if self._audio is None or len(self._audio) == 0:
            self.create_text(w // 2, h // 2,
                             text="Chưa có dữ liệu sóng âm (Ghi âm hoặc Tải file để xem)",
                             fill=C["txt3"], font=FONT_S)
            return

        # Downsample để vẽ vừa chiều rộng màn hình
        n_pts = min(w, len(self._audio))
        step = max(1, len(self._audio) // w)
        sub = self._audio[::step][:w]
        max_abs = np.max(np.abs(self._audio)) + 1e-6
        scale = (h * 0.42) / max_abs

        # Vẽ dải phong bì biên độ (amplitude envelope)
        for x, val in enumerate(sub):
            y_top = mid_y - abs(val) * scale
            y_bot = mid_y + abs(val) * scale
            self.create_line(x, y_top, x, y_bot, fill=C["cyan"], width=1)

        # Vẽ đường trung tâm sóng
        pts = []
        for x, val in enumerate(sub):
            pts.extend([x, mid_y - val * scale])
        if len(pts) >= 4:
            self.create_line(*pts, fill="#ffffff", width=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Animated Orb (visualizer trạng thái)
# ══════════════════════════════════════════════════════════════════════════════

class OrbWidget(tk.Canvas):
    def __init__(self, parent, size=110, **kw):
        kw.setdefault("bg", C["card"])
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, width=size, height=size, **kw)
        self.size   = size
        self.cx     = size // 2
        self.cy     = size // 2
        self.r_base = size // 2 - 10
        self._phase = 0.0
        self._state = "idle"
        self._job   = None
        self._draw()

    def _draw(self):
        self.delete("all")
        if self._state == "idle":
            r, color, glow = self.r_base, C["pri"], C["pri"]
        elif self._state == "listening":
            r     = self.r_base + int(6 * np.sin(self._phase))
            color = C["cyan"]
            glow  = C["cyan"]
        elif self._state == "thinking":
            r     = self.r_base - 5 + int(5 * abs(np.sin(self._phase)))
            color = C["orange"]
            glow  = C["orange"]
        elif self._state == "done_pd":
            r, color, glow = self.r_base, C["red"], C["red"]
        elif self._state == "done_ok":
            r, color, glow = self.r_base, C["green"], C["green"]
        else:
            r, color, glow = self.r_base, C["pri"], C["pri"]

        # Glow vòng ngoài
        for d in range(16, 2, -3):
            alpha = int(50 * (1 - d / 16))
            g_col = _blend_alpha(glow, C["card"], alpha / 255)
            self.create_oval(self.cx - r - d, self.cy - r - d,
                             self.cx + r + d, self.cy + r + d,
                             fill=g_col, outline="")
        # Nhân tâm
        self.create_oval(self.cx - r, self.cy - r,
                         self.cx + r, self.cy + r,
                         fill=color, outline="")

    def set_state(self, state: str):
        self._state = state
        self._phase = 0.0
        if state in ("listening", "thinking"):
            self._animate()
        else:
            if self._job:
                self.after_cancel(self._job)
                self._job = None
            self._draw()

    def _animate(self):
        self._phase += 0.15
        self._draw()
        self._job = self.after(40, self._animate)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Local Application Window
# ══════════════════════════════════════════════════════════════════════════════

class ParkinsonApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("🧠 ParkiScan — Sàng lọc Bệnh Parkinson & Sinh Trắc Học Giọng Nói")
        self.geometry("1180x830")
        self.minsize(1000, 720)
        self.config(bg=C["bg"])

        self._recording = False
        self._playing = False
        self._rec_audio: Optional[np.ndarray] = None
        self._last_feats: Optional[Dict] = None
        self._last_result: Optional[Dict] = None
        self._last_bio: Optional[Dict] = None
        self._devices_list: List[Tuple[any, str]] = []
        self._q: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

        # Tự động kiểm tra trạng thái model
        self.after(200, self._check_model_on_start)

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._style_notebook()

        # Tab 1: Phân tích & Sàng lọc Parkinson
        t1 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t1, text="  🎙️ Sàng lọc & Phân tích  ")
        self._build_tab_analyze(t1)

        # Tab 2: Tiện ích Sinh trắc học (Giới tính, Cảm xúc, Tiếng ho, Âm vực)
        t2 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t2, text="  🧬 Sinh trắc học Giọng nói  ")
        self._build_tab_biometrics(t2)

        # Tab 3: Đặc trưng âm thanh (MFCC / Jitter / Shimmer)
        t3 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t3, text="  📊 Phổ đặc trưng & MFCC  ")
        self._build_tab_features(t3)

        # Tab 4: Huấn luyện Machine Learning (Đa mô hình)
        t4 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t4, text="  🤖 Huấn luyện ML  ")
        self._build_tab_train(t4)

        # Tab 5: Đánh giá & So sánh Model (ROC, CM, Feature Importance)
        t5 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t5, text="  📈 Đánh giá Model  ")
        self._build_tab_evaluation(t5)

        # Tab 6: Lịch sử Sàng lọc (SQLite)
        t6 = tk.Frame(self._nb, bg=C["bg"])
        self._nb.add(t6, text="  📋 Lịch sử Sàng lọc  ")
        self._build_tab_history(t6)

    def _style_notebook(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=C["bg"],
                        borderwidth=0,
                        tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
                        background=C["bg3"],
                        foreground=C["txt3"],
                        font=("Segoe UI", 10, "bold"),
                        padding=[16, 9],
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", C["card"])],
                  foreground=[("selected", C["pri"])],
                  expand=[("selected", [1, 1, 1, 0])])
        style.configure("TCombobox",
                        fieldbackground=C["bg3"],
                        background=C["bg3"],
                        foreground=C["txt"],
                        arrowcolor=C["cyan"])

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["bg2"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        inner = tk.Frame(hdr, bg=C["bg2"])
        inner.pack(fill="both", expand=True, padx=20)

        # Logo icon
        logo_box = tk.Frame(inner, bg=C["pri"], width=36, height=36)
        logo_box.pack(side="left", anchor="center", pady=14)
        logo_box.pack_propagate(False)
        tk.Label(logo_box, text="🧠", bg=C["pri"], font=("Segoe UI", 16)).pack(expand=True)

        # Tiêu đề
        title_box = tk.Frame(inner, bg=C["bg2"])
        title_box.pack(side="left", anchor="center", padx=10)
        tk.Label(title_box, text="ParkiScan Desktop — Sinh Trắc Học & Sàng Lọc Bệnh Parkinson",
                 bg=C["bg2"], fg=C["txt"], font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(title_box, text="Đề tài 19: Microphone INMP441 + Raspberry Pi 4 + SVM + Trắc lượng Giới tính & Cảm xúc",
                 bg=C["bg2"], fg=C["txt3"], font=FONT_S).pack(anchor="w")

        # Badges bên phải
        self._badge_var = tk.StringVar(value="⚪ Đang kiểm tra model…")
        self._badge_lbl = tk.Label(inner, textvariable=self._badge_var,
                                   bg=C["bg3"], fg=C["orange"],
                                   font=FONT_S, padx=12, pady=5)
        self._badge_lbl.pack(side="right", anchor="center", padx=4)

        tk.Label(inner, text="🍓 Hỗ trợ Raspberry Pi 4 + INMP441",
                 bg=C["bg3"], fg=C["cyan"], font=FONT_S,
                 padx=12, pady=5).pack(side="right", anchor="center", padx=4)

    # ── Tab 1: Sàng lọc & Phân tích ──────────────────────────────────────────

    def _build_tab_analyze(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        # ── Cột trái: Điều khiển thu âm & Phần cứng ──
        left = tk.Frame(parent, bg=C["bg"], padx=4, pady=6)
        left.grid(row=0, column=0, sticky="nsew")

        # Visualizer Orb
        orb_card = card(left)
        orb_card.pack(fill="x", padx=6, pady=4)
        tk.Frame(orb_card, bg=C["card"], height=8).pack()
        self._orb = OrbWidget(orb_card, size=110)
        self._orb.pack()
        self._orb_lbl = tk.Label(orb_card, text="Sẵn sàng thu nhận tín hiệu",
                                 bg=C["card"], fg=C["txt3"], font=FONT_S)
        self._orb_lbl.pack(pady=(4, 10))

        # Điều khiển thu âm & File
        btn_card = card(left)
        btn_card.pack(fill="x", padx=6, pady=4)
        tk.Label(btn_card, text="🎙️ Nguồn âm thanh đầu vào",
                 bg=C["card"], fg=C["txt"], font=FONT_B,
                 padx=14, pady=8).pack(anchor="w")
        separator(btn_card).pack(fill="x", padx=12)

        # Chọn thiết bị Microphone
        mic_box = tk.Frame(btn_card, bg=C["card"])
        mic_box.pack(fill="x", padx=14, pady=(6, 4))
        tk.Label(mic_box, text="Chọn Microphone:", bg=C["card"], fg=C["txt3"],
                 font=FONT_S).pack(anchor="w")
        
        dev_row = tk.Frame(mic_box, bg=C["card"])
        dev_row.pack(fill="x", pady=2)
        self._dev_combo = ttk.Combobox(dev_row, state="readonly", font=FONT_S)
        self._dev_combo.pack(side="left", fill="x", expand=True)
        tk.Button(dev_row, text="🔄", command=self._refresh_audio_devices,
                  bg=C["bg3"], fg=C["txt"], relief="flat", bd=0,
                  padx=6, pady=2, cursor="hand2").pack(side="right", padx=(4, 0))
        self._refresh_audio_devices()

        # Nút ghi âm & tải file
        self._rec_btn = btn(btn_card, "🎤  Bắt đầu Ghi âm (5s)", self._toggle_record,
                            color=C["cyan"])
        self._rec_btn.pack(fill="x", padx=14, pady=4)

        btn(btn_card, "📂  Tải file âm thanh (.wav/.mp3)", self._load_file,
            color=C["bg3"]).pack(fill="x", padx=14, pady=4)

        self._play_btn = btn(btn_card, "▶️  Nghe lại âm thanh", self._toggle_play,
                             color=C["bg3"])
        self._play_btn.pack(fill="x", padx=14, pady=4)
        self._play_btn.config(state="disabled")

        self._analyze_btn = btn(btn_card, "🔬  Phân tích Toàn diện & Sinh trắc học", self._run_analyze,
                                color=C["pri"])
        self._analyze_btn.pack(fill="x", padx=14, pady=(4, 8))
        self._analyze_btn.config(state="disabled")

        # Trạng thái
        self._status_var = tk.StringVar(value="Đang chờ giọng nói đầu vào…")
        status_lbl = tk.Label(btn_card, textvariable=self._status_var,
                              bg=C["card"], fg=C["txt3"], font=FONT_S,
                              wraplength=260, justify="center")
        status_lbl.pack(padx=14, pady=(0, 10))

        # Thông tin phần cứng nhúng
        hw_card = card(left)
        hw_card.pack(fill="x", padx=6, pady=4)
        tk.Label(hw_card, text="⚙️ Cấu hình Phần cứng đề tài",
                 bg=C["card"], fg=C["txt2"], font=FONT_B,
                 padx=14, pady=8).pack(anchor="w")
        separator(hw_card).pack(fill="x", padx=12)

        hw_info = [
            ("Microphone", "INMP441 (I2S MEMS)"),
            ("Vi xử lý",   "Raspberry Pi 4 Model B"),
            ("Sample Rate","16 000 Hz / Mono"),
            ("Giao thức",  "I2S Digital PCM"),
            ("Mô hình",    "SVM (RBF Kernel)"),
        ]
        for k, v in hw_info:
            row = tk.Frame(hw_card, bg=C["card"])
            row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=k + ":", bg=C["card"], fg=C["txt3"],
                     font=FONT_S, width=12, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=C["card"], fg=C["txt"],
                     font=FONT_S).pack(side="left")

        btn(hw_card, "📖 Xem Sơ đồ nối dây Raspberry Pi + INMP441", self._show_raspi_guide,
            color=C["bg3"], font=FONT_S, pady=4).pack(fill="x", padx=14, pady=(8, 8))

        # ── Cột phải: Kết quả, Waveform & Bảng đặc trưng ──
        right = tk.Frame(parent, bg=C["bg"], padx=4, pady=6)
        right.grid(row=0, column=1, sticky="nsew")

        # Kết quả phân tích
        res_card = card(right)
        res_card.pack(fill="x", padx=6, pady=4)
        tk.Label(res_card, text="📋 Kết quả Sàng lọc Parkinson & Sinh Trắc Học",
                 bg=C["card"], fg=C["txt"], font=FONT_H2,
                 padx=16, pady=10).pack(anchor="w")
        separator(res_card).pack(fill="x", padx=12)

        self._result_frame = tk.Frame(res_card, bg=C["card"])
        self._result_frame.pack(fill="x", padx=16, pady=10)
        self._show_placeholder()

        # Waveform card
        wf_card = card(right)
        wf_card.pack(fill="x", padx=6, pady=4)
        wf_hdr = tk.Frame(wf_card, bg=C["card"])
        wf_hdr.pack(fill="x", padx=16, pady=(8, 2))
        tk.Label(wf_hdr, text="🌊 Dạng sóng âm thanh (Waveform)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3).pack(side="left")
        self._audio_dur_lbl = tk.Label(wf_hdr, text="0.0s | 16 kHz",
                                       bg=C["card"], fg=C["cyan"], font=FONT_M)
        self._audio_dur_lbl.pack(side="right")
        separator(wf_card).pack(fill="x", padx=12, pady=(2, 6))

        self._waveform = WaveformWidget(wf_card, height=75)
        self._waveform.pack(fill="x", padx=14, pady=(0, 8))

        # Bảng đặc trưng nhanh (MFCC, Jitter, Shimmer, F0)
        feat_card = card(right)
        feat_card.pack(fill="both", expand=True, padx=6, pady=4)
        tk.Label(feat_card, text="🔢 Các thông số âm học trích xuất (Acoustic Features)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=16, pady=8).pack(anchor="w")
        separator(feat_card).pack(fill="x", padx=12)

        self._feat_frame = tk.Frame(feat_card, bg=C["card"])
        self._feat_frame.pack(fill="both", expand=True, padx=12, pady=6)

        # Medical Disclaimer
        disc = tk.Frame(right, bg=C["bg"])
        disc.pack(fill="x", padx=6, pady=4)
        tk.Label(disc, text=(
            "⚠️ Tuyên bố miễn trừ trách nhiệm y tế: Hệ thống chỉ phục vụ mục đích nghiên cứu và sàng lọc ban đầu, "
            "không phải công cụ chẩn đoán y khoa và không thay thế thăm khám chuyên môn của bác sĩ chuyên khoa thần kinh."
        ), bg=C["bg"], fg=C["txt3"], font=("Segoe UI", 8),
                 wraplength=640, justify="left").pack(anchor="w", padx=4)

    def _show_placeholder(self):
        for w in self._result_frame.winfo_children():
            w.destroy()
        tk.Label(self._result_frame,
                 text="Chưa có kết quả phân tích.\nHãy bấm 'Bắt đầu Ghi âm' hoặc 'Tải file âm thanh' rồi bấm 'Phân tích'.",
                 bg=C["card"], fg=C["txt3"], font=FONT_N,
                 justify="center").pack(pady=16)

    def _show_result(self, result: Dict, bio: Optional[Dict] = None):
        for w in self._result_frame.winfo_children():
            w.destroy()

        prob   = result["probability"]
        label_ = result["label"]
        is_pd  = result["prediction"] == 1

        bar_bg   = C["red"] if is_pd else C["green"]
        bar_text = f"{'⚠️ ' if is_pd else '✅ '}{label_}  —  {prob:.1f}% xác suất nguy cơ"

        risk_bar = tk.Frame(self._result_frame, bg=bar_bg, height=44, relief="flat")
        risk_bar.pack(fill="x", pady=(0, 8))
        risk_bar.pack_propagate(False)
        tk.Label(risk_bar, text=bar_text, bg=bar_bg, fg="#ffffff",
                 font=("Segoe UI", 12, "bold")).pack(expand=True)

        # Progress bar
        tk.Label(self._result_frame, text="Mức độ rủi ro Parkinson:", bg=C["card"],
                 fg=C["txt3"], font=FONT_S).pack(anchor="w")

        pb_outer = tk.Frame(self._result_frame, bg=C["bg3"], height=10)
        pb_outer.pack(fill="x", pady=(3, 8))
        pb_outer.pack_propagate(False)

        fill_pct = min(prob / 100, 1.0)
        pb_inner = tk.Frame(pb_outer, bg=bar_bg, height=10)
        pb_inner.place(relwidth=fill_pct, relheight=1.0)

        # Sinh trắc học giọng nói tóm tắt (Gender, Emotion, Cough, Pitch)
        if bio:
            bio_box = tk.Frame(self._result_frame, bg=C["card"])
            bio_box.pack(fill="x", pady=(4, 8))

            tk.Label(bio_box, text="🧬 Tiện ích Sinh trắc học giọng nói (Phát hiện tự động):",
                     bg=C["card"], fg=C["cyan"], font=FONT_B).pack(anchor="w", pady=(0, 4))
            
            bio_row = tk.Frame(bio_box, bg=C["card"])
            bio_row.pack(fill="x")

            # Gender badge
            g_data = bio["gender"]
            g_is_male = g_data["gender"].lower() == "male"
            g_bg = "#2563eb" if g_is_male else "#db2777"
            f_g = tk.Frame(bio_row, bg=g_bg, padx=8, pady=4)
            f_g.pack(side="left", padx=(0, 6))
            tk.Label(f_g, text=f"👤 Giới tính: {g_data['label_vi']} ({g_data['confidence']}%)",
                     bg=g_bg, fg="#ffffff", font=FONT_S).pack()

            # Emotion badge
            e_data = bio["emotion"]
            f_e = tk.Frame(bio_row, bg="#7c3aed", padx=8, pady=4)
            f_e.pack(side="left", padx=(0, 6))
            tk.Label(f_e, text=f"😊 Cảm xúc: {e_data['label_vi']} ({e_data['confidence']}%)",
                     bg="#7c3aed", fg="#ffffff", font=FONT_S).pack()

            # Cough badge
            c_data = bio["cough"]
            f_c = tk.Frame(bio_row, bg="#059669", padx=8, pady=4)
            f_c.pack(side="left", padx=(0, 6))
            tk.Label(f_c, text=f"🩺 Hô hấp: {c_data['label_vi']}",
                     bg="#059669", fg="#ffffff", font=FONT_S).pack()

            # Pitch profile badge
            p_data = bio["pitch_profile"]
            f_p = tk.Frame(bio_row, bg=C["bg3"], padx=8, pady=4)
            f_p.pack(side="left")
            tk.Label(f_p, text=f"🎵 Âm vực: {p_data['pitch_range']}",
                     bg=C["bg3"], fg=C["txt"], font=FONT_S).pack()

        # Lời khuyên
        if prob >= 60:
            advice = "Phát hiện nhiều bất thường âm học (Jitter/Shimmer cao). Khuyến nghị thăm khám chuyên khoa Thần kinh."
        elif prob >= 30:
            advice = "Phát hiện một vài chỉ số nghi vấn ở mức nhẹ. Nên ghi âm lại trong môi trường yên tĩnh hoặc theo dõi định kỳ."
        else:
            advice = "Tần số cơ bản F0, độ rung Jitter và biên độ Shimmer hoàn toàn bình thường. Không phát hiện dấu hiệu rối loạn giọng nói."

        tk.Label(self._result_frame, text=advice, bg=C["card"], fg=C["txt2"],
                 font=FONT_N, wraplength=580, justify="left").pack(anchor="w", pady=(4, 0))

    def _show_quick_features(self, feats: Dict):
        for w in self._feat_frame.winfo_children():
            w.destroy()

        keys = [
            ("F0 cơ bản (mean)",   "f0_mean",       "Hz"),
            ("Độ lệch chuẩn F0",  "f0_std",        "Hz"),
            ("Jitter (local)",     "jitter_local",  ""),
            ("Jitter (RAP)",       "jitter_rap",    ""),
            ("Shimmer (local)",    "shimmer_local", ""),
            ("Shimmer (dB)",       "shimmer_db",    "dB"),
            ("HNR (Độ trong)",     "hnr",           "dB"),
            ("NHR (Nhiễu ồn)",     "nhr",           ""),
            ("PPE (Entropy cao độ)","ppe",          ""),
            ("Tốc độ phát âm",    "speech_rate",   "onset/s"),
            ("Tỷ lệ ngắt nghỉ",   "pause_ratio",   "%"),
            ("MFCC 1 (Trung bình)","mfcc_1_mean",   ""),
        ]
        cols = 4
        for idx, (name, key, unit) in enumerate(keys):
            r, c = divmod(idx, cols)
            val = feats.get(key, 0.0)
            cell = tk.Frame(self._feat_frame, bg=C["bg3"], relief="flat", bd=0)
            cell.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
            self._feat_frame.columnconfigure(c, weight=1)
            tk.Label(cell, text=name, bg=C["bg3"], fg=C["txt3"],
                     font=("Segoe UI", 8)).pack(anchor="w", padx=6, pady=(4, 0))
            if unit == "%":
                display = f"{val*100:.1f}%"
            elif unit:
                display = f"{val:.2f} {unit}"
            else:
                display = f"{val:.4f}"
            tk.Label(cell, text=display, bg=C["bg3"], fg=C["txt"],
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(0, 4))

    # ── Tab 2: Tiện ích Sinh trắc học Giọng nói (MỚI) ────────────────────────

    def _build_tab_biometrics(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        # ── Cột trái: Nhận diện Giới tính & Phân tích Hô hấp ──
        left = tk.Frame(parent, bg=C["bg"], padx=6, pady=8)
        left.grid(row=0, column=0, sticky="nsew")

        # 1. Card Nhận diện Giới tính
        self._bio_gender_card = card(left)
        self._bio_gender_card.pack(fill="x", padx=6, pady=4)
        tk.Label(self._bio_gender_card, text="👤 Nhận diện Giới tính (Gender Recognition)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=8).pack(anchor="w")
        separator(self._bio_gender_card).pack(fill="x", padx=12)

        self._bio_gender_content = tk.Frame(self._bio_gender_card, bg=C["card"])
        self._bio_gender_content.pack(fill="x", padx=14, pady=10)
        self._show_gender_placeholder()

        # 2. Card Phân tích Tiếng ho & Hô hấp
        self._bio_cough_card = card(left)
        self._bio_cough_card.pack(fill="both", expand=True, padx=6, pady=4)
        tk.Label(self._bio_cough_card, text="🩺 Phân tích Tiếng ho & Đường hô hấp (Cough Analysis)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=8).pack(anchor="w")
        separator(self._bio_cough_card).pack(fill="x", padx=12)

        self._bio_cough_content = tk.Frame(self._bio_cough_card, bg=C["card"])
        self._bio_cough_content.pack(fill="both", expand=True, padx=14, pady=10)
        self._show_cough_placeholder()

        # ── Cột phải: Nhận diện Cảm xúc & Hồ sơ Âm vực ──
        right = tk.Frame(parent, bg=C["bg"], padx=6, pady=8)
        right.grid(row=0, column=1, sticky="nsew")

        # 3. Card Cảm xúc giọng nói
        self._bio_emo_card = card(right)
        self._bio_emo_card.pack(fill="x", padx=6, pady=4)
        tk.Label(self._bio_emo_card, text="😊 Phân tích Cảm xúc Giọng nói (Speech Emotion)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=8).pack(anchor="w")
        separator(self._bio_emo_card).pack(fill="x", padx=12)

        self._bio_emo_content = tk.Frame(self._bio_emo_card, bg=C["card"])
        self._bio_emo_content.pack(fill="x", padx=14, pady=10)
        self._show_emotion_placeholder()

        # 4. Card Âm vực & Thanh điệu
        self._bio_pitch_card = card(right)
        self._bio_pitch_card.pack(fill="both", expand=True, padx=6, pady=4)
        tk.Label(self._bio_pitch_card, text="🎵 Trắc lượng Âm vực & Độ cao thanh đới (Voice Profile)",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=8).pack(anchor="w")
        separator(self._bio_pitch_card).pack(fill="x", padx=12)

        self._bio_pitch_content = tk.Frame(self._bio_pitch_card, bg=C["card"])
        self._bio_pitch_content.pack(fill="both", expand=True, padx=14, pady=10)
        self._show_pitch_placeholder()

    def _show_gender_placeholder(self):
        for w in self._bio_gender_content.winfo_children():
            w.destroy()
        tk.Label(self._bio_gender_content,
                 text="Chưa có dữ liệu.\nHãy ghi âm hoặc tải file âm thanh ở Tab 1 rồi bấm Phân tích.",
                 bg=C["card"], fg=C["txt3"], font=FONT_N).pack(pady=12)

    def _show_cough_placeholder(self):
        for w in self._bio_cough_content.winfo_children():
            w.destroy()
        tk.Label(self._bio_cough_content,
                 text="Chưa có dữ liệu hô hấp.",
                 bg=C["card"], fg=C["txt3"], font=FONT_N).pack(pady=12)

    def _show_emotion_placeholder(self):
        for w in self._bio_emo_content.winfo_children():
            w.destroy()
        tk.Label(self._bio_emo_content,
                 text="Chưa có dữ liệu cảm xúc giọng nói.",
                 bg=C["card"], fg=C["txt3"], font=FONT_N).pack(pady=12)

    def _show_pitch_placeholder(self):
        for w in self._bio_pitch_content.winfo_children():
            w.destroy()
        tk.Label(self._bio_pitch_content,
                 text="Chưa có dữ liệu trắc lượng âm vực.",
                 bg=C["card"], fg=C["txt3"], font=FONT_N).pack(pady=12)

    def _update_biometrics_tab(self, bio: Dict):
        # 1. Cập nhật Gender
        for w in self._bio_gender_content.winfo_children():
            w.destroy()
        g = bio["gender"]
        is_male = g["gender"].lower() == "male"
        badge_bg = "#2563eb" if is_male else "#db2777"
        
        # Gender Header Badge
        hdr = tk.Frame(self._bio_gender_content, bg=badge_bg, height=40)
        hdr.pack(fill="x", pady=(0, 8))
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"👤 Giới tính dự đoán: {g['label_vi'].upper()}  ({g['confidence']}%)",
                 bg=badge_bg, fg="#fff", font=FONT_B).pack(expand=True)

        # Bar so sánh Nam vs Nữ
        p_row = tk.Frame(self._bio_gender_content, bg=C["card"])
        p_row.pack(fill="x", pady=2)
        tk.Label(p_row, text=f"Nam: {g['scores'].get('male', 0)*100:.1f}%", bg=C["card"], fg="#60a5fa", font=FONT_S).pack(side="left")
        tk.Label(p_row, text=f"Nữ: {g['scores'].get('female', 0)*100:.1f}%", bg=C["card"], fg="#f472b6", font=FONT_S).pack(side="right")
        
        bar_frame = tk.Frame(self._bio_gender_content, bg=C["bg3"], height=8)
        bar_frame.pack(fill="x", pady=(2, 8))
        bar_frame.pack_propagate(False)
        male_pct = g["scores"].get("male", 0.5)
        tk.Frame(bar_frame, bg="#2563eb").place(relx=0, rely=0, relwidth=male_pct, relheight=1.0)
        tk.Frame(bar_frame, bg="#db2777").place(relx=male_pct, rely=0, relwidth=1.0 - male_pct, relheight=1.0)

        # Bảng 6 thông số voice.csv
        tk.Label(self._bio_gender_content, text="Đặc trưng âm học chuẩn voice.csv:", bg=C["card"], fg=C["txt3"], font=FONT_S).pack(anchor="w")
        tbl = tk.Frame(self._bio_gender_content, bg=C["card"])
        tbl.pack(fill="x", pady=4)
        g_feat_labels = [
            ("Tần số TB (meanfreq)", f"{g['features']['meanfreq']} kHz"),
            ("Độ lệch tần số (sd)",  f"{g['features']['sd']} kHz"),
            ("Trọng tâm phổ (centroid)", f"{g['features']['centroid']} kHz"),
            ("Tần số cơ bản (meanfun)", f"{g['features']['meanfun']*1000:.1f} Hz"),
            ("Dải tứ phân vị (IQR)", f"{g['features']['IQR']} kHz"),
            ("Tần số trung vị (median)", f"{g['features']['median']} kHz"),
        ]
        for idx, (lbl_name, val_str) in enumerate(g_feat_labels):
            r, c = divmod(idx, 2)
            cell = tk.Frame(tbl, bg=C["bg3"], padx=6, pady=4)
            cell.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
            tbl.columnconfigure(c, weight=1)
            tk.Label(cell, text=lbl_name, bg=C["bg3"], fg=C["txt3"], font=("Segoe UI", 8)).pack(anchor="w")
            tk.Label(cell, text=val_str, bg=C["bg3"], fg=C["txt"], font=FONT_B).pack(anchor="w")

        # 2. Cập nhật Cough
        for w in self._bio_cough_content.winfo_children():
            w.destroy()
        c_res = bio["cough"]
        c_bg = "#059669" if c_res["prediction"] == "not_cough" else "#d97706"
        c_box = tk.Frame(self._bio_cough_content, bg=c_bg, height=36)
        c_box.pack(fill="x", pady=(0, 8))
        c_box.pack_propagate(False)
        tk.Label(c_box, text=f"🩺 Trạng thái: {c_res['label_vi']} ({c_res['confidence']}%)",
                 bg=c_bg, fg="#fff", font=FONT_B).pack(expand=True)

        if c_res["prediction"] == "not_cough":
            c_desc = "Mẫu giọng nói ngân /a/ liên tục, không phát hiện xung kích âm thanh đặc trưng của tiếng ho."
        elif c_res["prediction"] == "dry_cough":
            c_desc = "Phát hiện xung kích âm tần số cao tương ứng với dấu hiệu ho khan."
        else:
            c_desc = "Phát hiện tiếng thở khò khè hoặc xung kích ẩm tương ứng với ho có đờm."
        tk.Label(self._bio_cough_content, text=c_desc, bg=C["card"], fg=C["txt2"],
                 font=FONT_N, wraplength=480, justify="left").pack(anchor="w")

        # 3. Cập nhật Emotion
        for w in self._bio_emo_content.winfo_children():
            w.destroy()
        e_res = bio["emotion"]
        e_top = tk.Frame(self._bio_emo_content, bg="#7c3aed", height=38)
        e_top.pack(fill="x", pady=(0, 8))
        e_top.pack_propagate(False)
        tk.Label(e_top, text=f"😊 Cảm xúc chủ đạo: {e_res['label_vi'].upper()} ({e_res['confidence']}%)",
                 bg="#7c3aed", fg="#fff", font=FONT_B).pack(expand=True)

        # Thanh phân bố các cảm xúc
        tk.Label(self._bio_emo_content, text="Phân bố các sắc thái cảm xúc:", bg=C["card"], fg=C["txt3"], font=FONT_S).pack(anchor="w")
        for emo_k, emo_vi in biometrics.EMOTION_LABEL_VI.items():
            pct = e_res["all_scores"].get(emo_k, 0.0)
            row = tk.Frame(self._bio_emo_content, bg=C["card"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=emo_vi, bg=C["card"], fg=C["txt2"], font=("Segoe UI", 8), width=16, anchor="w").pack(side="left")
            pb = tk.Frame(row, bg=C["bg3"], height=8)
            pb.pack(side="left", fill="x", expand=True, padx=4)
            pb.pack_propagate(False)
            fill_col = "#a855f7" if pct == e_res["confidence"] else C["pri"]
            tk.Frame(pb, bg=fill_col).place(relwidth=min(pct/100.0, 1.0), relheight=1.0)
            tk.Label(row, text=f"{pct:.1f}%", bg=C["card"], fg=C["txt3"], font=("Segoe UI", 8), width=6).pack(side="right")

        # 4. Cập nhật Pitch Profile
        for w in self._bio_pitch_content.winfo_children():
            w.destroy()
        p = bio["pitch_profile"]
        p_row1 = tk.Frame(self._bio_pitch_content, bg=C["bg3"], padx=10, pady=8)
        p_row1.pack(fill="x", pady=4)
        tk.Label(p_row1, text="Phân loại chất giọng:", bg=C["bg3"], fg=C["txt3"], font=FONT_S).pack(anchor="w")
        tk.Label(p_row1, text=f"🎵 {p['pitch_range']}", bg=C["bg3"], fg=C["cyan"], font=FONT_H3).pack(anchor="w")

        p_tbl = tk.Frame(self._bio_pitch_content, bg=C["card"])
        p_tbl.pack(fill="x", pady=4)
        items = [
            ("Tần số cơ bản F0", f"{p['f0_mean_hz']} Hz"),
            ("Trọng tâm phổ", f"{p['spectral_centroid_khz']} kHz"),
            ("Độ rộng phổ (IQR)", f"{p['iqr_khz']} kHz"),
        ]
        for idx, (k_lbl, v_str) in enumerate(items):
            cell = tk.Frame(p_tbl, bg=C["bg3"], padx=6, pady=4)
            cell.pack(fill="x", pady=2)
            tk.Label(cell, text=k_lbl + ":", bg=C["bg3"], fg=C["txt3"], font=FONT_S).pack(side="left")
            tk.Label(cell, text=v_str, bg=C["bg3"], fg=C["txt"], font=FONT_B).pack(side="right")

    # ── Tab 3: Biểu đồ đặc trưng âm học & MFCC ───────────────────────────────

    def _build_tab_features(self, parent):
        if not HAS_MPL:
            tk.Label(parent, text="Cần cài đặt matplotlib để vẽ biểu đồ: pip install matplotlib",
                     bg=C["bg"], fg=C["txt3"], font=FONT_N).pack(expand=True)
            return

        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.pack(fill="x", padx=16, pady=8)
        tk.Label(ctrl, text="Biểu đồ phân tích Âm học chuyên sâu (MFCC / Jitter / Shimmer / F0)",
                 bg=C["bg"], fg=C["txt"], font=FONT_H2).pack(side="left")
        btn(ctrl, "🔄 Cập nhật biểu đồ", self._update_feature_chart,
            color=C["pri"]).pack(side="right")

        self._chart_frame = tk.Frame(parent, bg=C["bg2"])
        self._chart_frame.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self._draw_empty_chart()

    def _draw_empty_chart(self):
        if not HAS_MPL:
            return
        fig = Figure(figsize=(10, 5), dpi=96)
        fig.patch.set_facecolor(C["bg2"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["card"])
        ax.text(0.5, 0.5, "Chưa có dữ liệu âm thanh.\nHãy ghi âm hoặc tải file rồi bấm Phân tích ở Tab 1.",
                ha="center", va="center", color=C["txt3"],
                fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        self._embed_chart(fig)

    def _update_feature_chart(self):
        if not HAS_MPL:
            return
        if self._last_feats is None:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu đặc trưng. Hãy phân tích âm thanh trước.")
            return
        self._draw_feature_chart(self._last_feats)

    def _draw_feature_chart(self, feats: Dict):
        if not HAS_MPL:
            return

        fig = Figure(figsize=(11, 6), dpi=96)
        fig.patch.set_facecolor(C["bg2"])
        fig.subplots_adjust(wspace=0.32, hspace=0.45, top=0.92, bottom=0.08,
                            left=0.06, right=0.97)

        axes = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4),
        ]

        # 1. F0 & HNR
        ax1 = axes[0]
        f0_keys = ["f0_mean", "f0_min", "f0_max", "f0_range", "hnr"]
        f0_names = ["F0 Mean", "F0 Min", "F0 Max", "F0 Range", "HNR (dB)"]
        f0_vals = [feats.get(k, 0.0) for k in f0_keys]
        ax1.bar(range(len(f0_names)), f0_vals, color=C["pri"], width=0.55)
        ax1.set_title("Tần số cơ bản F0 (Hz) & Độ trong HNR (dB)", color=C["txt"], fontsize=9, pad=5)
        ax1.set_xticks(range(len(f0_names)))
        ax1.set_xticklabels(f0_names, fontsize=7, color=C["txt3"])

        # 2. Jitter
        ax2 = axes[1]
        jit_keys = ["jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp"]
        jit_names = ["Local", "RAP", "PPQ5", "DDP"]
        jit_vals = [feats.get(k, 0.0) * 100 for k in jit_keys]
        ax2.bar(range(len(jit_names)), jit_vals, color=C["cyan"], width=0.5)
        ax2.set_title("Độ bất ổn chu kỳ Jitter (%)", color=C["txt"], fontsize=9, pad=5)
        ax2.set_xticks(range(len(jit_names)))
        ax2.set_xticklabels(jit_names, fontsize=7, color=C["txt3"])

        # 3. Shimmer
        ax3 = axes[2]
        shim_keys = ["shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5"]
        shim_names = ["Local (%)", "dB", "APQ3 (%)", "APQ5 (%)"]
        shim_vals = [
            feats.get("shimmer_local", 0.0) * 100,
            feats.get("shimmer_db", 0.0),
            feats.get("shimmer_apq3", 0.0) * 100,
            feats.get("shimmer_apq5", 0.0) * 100,
        ]
        ax3.bar(range(len(shim_names)), shim_vals, color=C["orange"], width=0.5)
        ax3.set_title("Độ bất ổn biên độ Shimmer", color=C["txt"], fontsize=9, pad=5)
        ax3.set_xticks(range(len(shim_names)))
        ax3.set_xticklabels(shim_names, fontsize=7, color=C["txt3"])

        # 4. MFCC 1-13 (Mel-Frequency Cepstral Coefficients)
        ax4 = axes[3]
        mfcc_vals = [feats.get(f"mfcc_{i+1}_mean", 0.0) for i in range(13)]
        mfcc_labels = [f"C{i+1}" for i in range(13)]
        ax4.bar(range(13), mfcc_vals, color=C["green"], width=0.6)
        ax4.set_title("13 Hệ số MFCC Mean (Mel-Frequency Cepstral Coefficients)", color=C["txt"], fontsize=9, pad=5)
        ax4.set_xticks(range(13))
        ax4.set_xticklabels(mfcc_labels, fontsize=7, color=C["txt3"])

        for ax in axes:
            ax.set_facecolor(C["card"])
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])
            ax.tick_params(colors=C["txt3"], labelsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3, color=C["border"])

        self._embed_chart(fig)

    def _embed_chart(self, fig):
        for w in self._chart_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 4: Huấn luyện SVM ───────────────────────────────────────────────

    def _build_tab_train(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        left = tk.Frame(parent, bg=C["bg"], padx=4, pady=8)
        left.grid(row=0, column=0, sticky="nsew")

        info_card = card(left)
        info_card.pack(fill="x", padx=6, pady=4)
        tk.Label(info_card, text="📂 Bộ dữ liệu UCI Parkinson",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=10).pack(anchor="w")
        separator(info_card).pack(fill="x", padx=12)

        info_rows = [
            ("Nguồn",      "UCI ML Repository (Oxford)"),
            ("Tổng số mẫu","195 recordings"),
            ("Bệnh nhân",  "31 đối tượng (PD + Đối chứng)"),
            ("Đặc trưng",  "22 acoustic features"),
            ("Phân loại",  "Support Vector Machine (SVM)"),
            ("Kernel",     "RBF (Radial Basis Function)"),
            ("Tối ưu",     "GridSearchCV (C, gamma)"),
        ]
        for k, v in info_rows:
            row = tk.Frame(info_card, bg=C["card"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=k + ":", bg=C["card"], fg=C["txt3"],
                     font=FONT_S, width=12, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=C["card"], fg=C["txt"],
                     font=FONT_B).pack(side="left")

        # Trạng thái Model
        self._model_status_var = tk.StringVar(value="⚪ Đang kiểm tra…")
        ms_card = card(left)
        ms_card.pack(fill="x", padx=6, pady=4)
        tk.Label(ms_card, text="📦 Trạng thái Model",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=8).pack(anchor="w")
        separator(ms_card).pack(fill="x", padx=12)
        tk.Label(ms_card, textvariable=self._model_status_var,
                 bg=C["card"], fg=C["orange"], font=FONT_B,
                 padx=14, pady=8).pack(anchor="w")

        # Nút huấn luyện
        self._train_btn = btn(left, "🚀  Bắt đầu Huấn luyện SVM", self._run_train,
                              color=C["pri"])
        self._train_btn.pack(fill="x", padx=10, pady=(10, 4))

        self._train_prog = ttk.Progressbar(left, mode="indeterminate")
        self._train_prog.pack(fill="x", padx=10, pady=4)

        self._train_status_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self._train_status_var,
                 bg=C["bg"], fg=C["txt3"], font=FONT_S,
                 wraplength=260).pack(padx=10, pady=4)

        # Cột phải: Log / Report
        right = tk.Frame(parent, bg=C["bg"], padx=4, pady=8)
        right.grid(row=0, column=1, sticky="nsew")

        log_card = card(right)
        log_card.pack(fill="both", expand=True, padx=6, pady=4)
        tk.Label(log_card, text="📋 Báo cáo Đánh giá Huấn luyện & Cross-Validation",
                 bg=C["card"], fg=C["txt"], font=FONT_H3,
                 padx=14, pady=10).pack(anchor="w")
        separator(log_card).pack(fill="x", padx=12)

        txt_frame = tk.Frame(log_card, bg=C["card"])
        txt_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self._train_log = tk.Text(
            txt_frame, bg=C["bg3"], fg=C["txt"], font=FONT_M,
            relief="flat", bd=0, insertbackground=C["txt"],
            state="disabled", wrap="word", spacing1=2,
        )
        sb = ttk.Scrollbar(txt_frame, command=self._train_log.yview)
        self._train_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._train_log.pack(fill="both", expand=True)

        self._log("🧠 ParkiScan — Console Huấn luyện SVM trên UCI Dataset\n")
        self._log("=" * 60 + "\n")
        self._log(f"Đường dẫn dataset: {DATA_PATH}\n")
        self._log(f"Đường dẫn lưu model: {MODEL_PATH}\n")
        self._log("-" * 60 + "\n")
        if is_model_trained():
            self._log("✅ Model SVM đã sẵn sàng. Bạn có thể huấn luyện lại bất cứ lúc nào.\n")
        else:
            self._log("⚠️ Chưa phát hiện model. Hãy bấm 'Bắt đầu Huấn luyện SVM'!\n")

    def _log(self, text: str):
        self._train_log.config(state="normal")
        self._train_log.insert("end", text)
        self._train_log.see("end")
        self._train_log.config(state="disabled")

    # ── Audio Recording & Devices ────────────────────────────────────────────

    def _refresh_audio_devices(self):
        devices = [("auto", "Tự động (Mặc định hệ thống / INMP441)")]
        if HAS_AUDIO:
            try:
                devs = sd.query_devices()
                for idx, d in enumerate(devs):
                    if d.get("max_input_channels", 0) > 0:
                        name = d.get("name", f"Microphone {idx}")
                        short_name = name[:36] + ("…" if len(name) > 36 else "")
                        devices.append((idx, f"[{idx}] {short_name}"))
            except Exception:
                pass
        self._devices_list = devices
        self._dev_combo["values"] = [lbl for _, lbl in devices]
        self._dev_combo.current(0)

    def _toggle_record(self):
        if self._recording:
            self._recording = False
            self._rec_btn.config(text="🎤  Bắt đầu Ghi âm (5s)", bg=C["cyan"])
        else:
            if not HAS_AUDIO:
                messagebox.showerror("Thiếu thư viện",
                    "Cần cài đặt sounddevice và soundfile để ghi âm:\npip install sounddevice soundfile")
                return
            self._recording = True
            self._rec_btn.config(text="⏹  Đang ghi âm (Bấm để dừng)", bg=C["red"])
            self._orb.set_state("listening")
            self._orb_lbl.config(text="Đang ghi âm giọng nói…", fg=C["cyan"])
            self._status_var.set("🎙️ Đang ghi âm trong 5 giây… Hãy phát âm /a/ liên tục")
            self._analyze_btn.config(state="disabled")
            self._play_btn.config(state="disabled")
            threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        try:
            choice_idx = self._dev_combo.current()
            dev_id = self._devices_list[choice_idx][0]
            dev_arg = None if dev_id == "auto" else dev_id

            # Ghi âm qua sounddevice
            audio = sd.rec(int(REC_DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1,
                           device=dev_arg, dtype="float32")
            sd.wait()
            self._rec_audio = audio.flatten()
            self._q.put(("record_done", None))
        except Exception as e:
            # Fallback trên Raspberry Pi: arecord
            if sys.platform.startswith("linux") and shutil.which("arecord"):
                try:
                    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_wav.close()
                    cmd = ["arecord", "-D", "plughw:1,0", "-r", str(SAMPLE_RATE),
                           "-c", "1", "-f", "S24_LE", "-d", str(REC_DURATION), tmp_wav.name]
                    subprocess.run(cmd, check=True, timeout=REC_DURATION + 2)
                    y, _ = sf.read(tmp_wav.name, dtype="float32")
                    os.unlink(tmp_wav.name)
                    self._rec_audio = y
                    self._q.put(("record_done", None))
                    return
                except Exception:
                    pass
            self._q.put(("record_error", str(e)))

    def _toggle_play(self):
        if self._playing:
            sd.stop()
            self._playing = False
            self._play_btn.config(text="▶️  Nghe lại âm thanh", bg=C["bg3"])
        else:
            if self._rec_audio is None:
                return
            self._playing = True
            self._play_btn.config(text="⏹  Dừng phát âm thanh", bg=C["orange"])
            def _play():
                try:
                    sd.play(self._rec_audio, SAMPLE_RATE)
                    sd.wait()
                finally:
                    self._q.put(("play_done", None))
            threading.Thread(target=_play, daemon=True).start()

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Chọn file âm thanh mẫu",
            filetypes=[("Audio files", "*.wav *.mp3 *.ogg *.flac *.webm"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            import librosa
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self._rec_audio = y
            dur = len(y) / SAMPLE_RATE
            self._waveform.set_audio(self._rec_audio)
            self._audio_dur_lbl.config(text=f"{dur:.1f}s | {SAMPLE_RATE} Hz")
            self._status_var.set(f"✅ Đã tải: {Path(path).name} ({dur:.1f}s)")
            self._analyze_btn.config(state="normal")
            self._play_btn.config(state="normal")
            self._orb.set_state("idle")
            self._orb_lbl.config(text="Âm thanh đã sẵn sàng", fg=C["green"])
        except Exception as e:
            messagebox.showerror("Lỗi đọc file", f"Không thể tải file âm thanh:\n{e}")

    def _run_analyze(self):
        if self._rec_audio is None:
            messagebox.showwarning("Thông báo", "Vui lòng ghi âm hoặc tải file âm thanh trước!")
            return
        if not is_model_trained():
            messagebox.showwarning("Chưa có model",
                "Mô hình SVM chưa được huấn luyện.\nVui lòng vào tab 'Huấn luyện SVM' để huấn luyện trước!")
            return
        self._analyze_btn.config(state="disabled")
        self._orb.set_state("thinking")
        self._orb_lbl.config(text="Đang phân tích toàn diện…", fg=C["orange"])
        self._status_var.set("🔬 Đang phân tích MFCC, Jitter, Shimmer, Giới tính, Cảm xúc…")
        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, self._rec_audio, SAMPLE_RATE)
            tmp_path = tmp.name
            tmp.close()

            # 1. Trích xuất đặc trưng Parkinson (51 đặc trưng) & dự đoán SVM
            feats = extract_features(tmp_path)
            parkinson_res = predict_from_audio_features(feats)

            # 2. Tiện ích Sinh trắc học: Giới tính, Cảm xúc, Tiếng ho, Âm vực
            bio_res = biometrics.get_complete_biometrics(tmp_path)

            os.unlink(tmp_path)
            self._q.put(("analyze_done", (feats, parkinson_res, bio_res)))
        except Exception as e:
            self._q.put(("analyze_error", str(e)))

    def _run_train(self):
        self._train_btn.config(state="disabled")
        self._train_prog.start(10)
        self._train_status_var.set("⏳ Đang huấn luyện SVM qua GridSearchCV (5-Fold CV)…")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results = train_model(verbose=True)
            log_text = buf.getvalue()
            self._q.put(("train_done", (results, log_text)))
        except Exception as e:
            self._q.put(("train_error", str(e)))

    # ── Queue Polling ────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg, data = self._q.get_nowait()
                self._handle_msg(msg, data)
        except queue.Empty:
            pass
        self.after(40, self._poll_queue)

    def _handle_msg(self, msg: str, data):
        if msg == "record_done":
            self._recording = False
            self._rec_btn.config(text="🎤  Bắt đầu Ghi âm (5s)", bg=C["cyan"])
            self._orb.set_state("idle")
            self._orb_lbl.config(text="Ghi âm hoàn tất", fg=C["green"])
            self._status_var.set("✅ Đã ghi âm xong! Bấm 'Phân tích' để xem kết quả")
            self._waveform.set_audio(self._rec_audio)
            dur = len(self._rec_audio) / SAMPLE_RATE if self._rec_audio is not None else 0
            self._audio_dur_lbl.config(text=f"{dur:.1f}s | {SAMPLE_RATE} Hz")
            self._analyze_btn.config(state="normal")
            self._play_btn.config(state="normal")

        elif msg == "record_error":
            self._recording = False
            self._rec_btn.config(text="🎤  Bắt đầu Ghi âm (5s)", bg=C["cyan"])
            self._orb.set_state("idle")
            self._orb_lbl.config(text="Lỗi ghi âm", fg=C["red"])
            self._status_var.set("❌ Không thể ghi âm từ thiết bị đã chọn")
            messagebox.showerror("Lỗi Microphone",
                f"Không thể thu âm từ microphone:\n{data}\n\nHãy kiểm tra quyền truy cập Microphone hoặc chọn thiết bị khác.")

        elif msg == "play_done":
            self._playing = False
            self._play_btn.config(text="▶️  Nghe lại âm thanh", bg=C["bg3"])

        elif msg == "analyze_done":
            feats, parkinson_res, bio_res = data
            self._last_feats  = feats
            self._last_result = parkinson_res
            self._last_bio    = bio_res

            is_pd = parkinson_res["prediction"] == 1
            self._orb.set_state("done_pd" if is_pd else "done_ok")
            self._orb_lbl.config(
                text="⚠️ Nguy cơ cao" if is_pd else "✅ Khỏe mạnh",
                fg=C["red"] if is_pd else C["green"],
            )
            g_lbl = bio_res["gender"]["label_vi"]
            g_conf = bio_res["gender"]["confidence"]
            self._status_var.set(f"Parkinson: {parkinson_res['label']} ({parkinson_res['probability']:.1f}%) | Giới tính: {g_lbl} ({g_conf}%)")

            # Cập nhật kết quả Tab 1
            self._show_result(parkinson_res, bio_res)
            self._show_quick_features(feats)
            self._analyze_btn.config(state="normal")

            # Cập nhật Tab 2 (Sinh trắc học chi tiết)
            self._update_biometrics_tab(bio_res)

            # Cập nhật Tab 3 (Biểu đồ MFCC/Jitter/Shimmer)
            if HAS_MPL:
                self._draw_feature_chart(feats)

        elif msg == "analyze_error":
            self._orb.set_state("idle")
            self._orb_lbl.config(text="Lỗi phân tích", fg=C["red"])
            self._status_var.set("❌ Phân tích thất bại")
            self._analyze_btn.config(state="normal")
            messagebox.showerror("Lỗi phân tích", f"Quá trình trích xuất đặc trưng gặp lỗi:\n{data}")

        elif msg == "train_done":
            results, log_text = data
            self._train_prog.stop()
            self._train_btn.config(state="normal")
            self._train_status_var.set("✅ Huấn luyện SVM hoàn tất!")
            self._model_status_var.set("✅ Đã sẵn sàng (CV AUC: " + results.get("cv_auc", "N/A") + ")")
            self._badge_var.set("✅ Model SVM sẵn sàng")
            self._badge_lbl.config(fg=C["green"])

            self._log("\n" + "=" * 60 + "\n")
            self._log(log_text)
            self._log(f"\n📈 CV Accuracy : {results.get('cv_accuracy')}\n")
            self._log(f"📈 CV ROC-AUC  : {results.get('cv_auc')}\n")
            self._log(f"📈 Best params : {results.get('best_params')}\n")
            self._log(f"💾 File lưu    : {MODEL_PATH}\n")
            self._log("=" * 60 + "\n")

        elif msg == "train_error":
            self._train_prog.stop()
            self._train_btn.config(state="normal")
            self._train_status_var.set("❌ Lỗi trong quá trình huấn luyện!")
            self._log(f"\n❌ Lỗi: {data}\n")
            messagebox.showerror("Lỗi huấn luyện", str(data))

    def _check_model_on_start(self):
        if is_model_trained():
            self._badge_var.set("✅ Model SVM sẵn sàng")
            self._badge_lbl.config(fg=C["green"])
            self._model_status_var.set("✅ Đã huấn luyện sẵn")
        else:
            self._badge_var.set("⚠️ Chưa có model")
            self._badge_lbl.config(fg=C["orange"])
            self._model_status_var.set("⚠️ Chưa có model — Cần bấm Huấn luyện")

    def _show_raspi_guide(self):
        win = tk.Toplevel(self)
        win.title("🍓 Hướng dẫn Phần cứng: Raspberry Pi 4 + Microphone INMP441")
        win.geometry("640x520")
        win.config(bg=C["bg2"])
        win.transient(self)

        tk.Label(win, text="🍓 Sơ đồ kết nối Microphone INMP441 (I2S) với Raspberry Pi 4",
                 bg=C["bg2"], fg=C["txt"], font=FONT_H2, padx=16, pady=12).pack(anchor="w")

        body = tk.Text(win, bg=C["card"], fg=C["txt"], font=FONT_M,
                       relief="flat", bd=0, padx=16, pady=12)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        guide_text = """
1. BẢNG NỐI DÂY (PINOUT I2S DIGITAL AUDIO):
----------------------------------------------------------------------
  Chân INMP441        Chân Raspberry Pi 4           Chức năng
----------------------------------------------------------------------
  VDD          -----> Pin 1  (3.3V Power)           Nguồn 3.3V
  GND          -----> Pin 6  (GND)                  Nối đất
  SD (DOUT)    -----> Pin 38 (GPIO 20 / PCM_DIN)    Dữ liệu I2S
  SCK (BCLK)   -----> Pin 12 (GPIO 18 / PCM_CLK)    Xung nhịp Bit Clock
  WS (LRCLK)   -----> Pin 35 (GPIO 19 / PCM_FS)     Word Select (Kênh L/R)
  L/R          -----> Pin 9  (GND)                  Chọn kênh Trái (Left)
----------------------------------------------------------------------

2. KÍCH HOẠT DRIVER I2S TRÊN RASPBERRY PI OS:
- Chỉnh sửa file cấu hình boot:
    sudo nano /boot/config.txt   (hoặc /boot/firmware/config.txt)
- Thêm vào cuối file dòng sau:
    dtoverlay=googlevoicehat-soundcard
    dtparam=i2s=on
- Khởi động lại Raspberry Pi:
    sudo reboot

3. KIỂM TRA THIẾT BỊ TRÊN RASPBERRY PI:
- Kiểm tra danh sách microphone:
    arecord -l
- Ghi âm thử nghiệm 5 giây:
    arecord -D plughw:1,0 -r 16000 -c 1 -f S24_LE -d 5 test_inmp441.wav

4. KHỞI CHẠY ỨNG DỤNG TRÊN RASPBERRY PI:
    python app.py
Ứng dụng sẽ tự động chọn micro I2S INMP441 để sàng lọc giọng nói!
"""
        body.insert("1.0", guide_text)
        body.config(state="disabled")

        btn(win, "Đóng cửa sổ", win.destroy, color=C["pri"]).pack(pady=(0, 12))


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = BASE_DIR / "data" / "parkinsons.data"

if __name__ == "__main__":
    app = ParkinsonApp()
    app.mainloop()
