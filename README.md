# Hệ Thống Sinh Trắc Học & Phân Tích Giọng Nói (Đồ án 2.1)
## Voice Biometrics & Disease Screening System

Hệ thống phân tích giọng nói toàn diện phát triển trên nền tảng Python/Flask, mở rộng với module chuyên sâu **Sàng lọc Bệnh Parkinson qua Giọng nói (Parkinson's Disease Voice Screening)** sử dụng Machine Learning và triển khai phần cứng nhúng Raspberry Pi 4 + Microphone INMP441.

---

## Cấu trúc Hệ thống và Các Phân hệ (Modules)

```
Đồ án 2.1
│
├── Audio Analysis (Xử lý tín hiệu & đặc trưng cơ bản)
├── Gender Recognition (Nhận diện giới tính từ giọng nói)
├── Emotion Recognition (Nhận diện cảm xúc: vui, buồn, giận, ...)
├── Pronunciation Analysis (Phân tích phát âm & âm vực)
├── Cough Analysis (Phân tích tiếng ho & bệnh lý đường hô hấp)
├── Depression Screening (Sàng lọc dấu hiệu trầm cảm qua giọng nói)
└── Parkinson's Disease Voice Screening (Sàng lọc hỗ trợ chẩn đoán Parkinson)
```

---

## Chi tiết Module: Parkinson's Disease Voice Screening

### 1. Problem Statement (Bài toán)
Bệnh Parkinson là một bệnh lý thoái hóa thần kinh tiến triển, ảnh hưởng nghiêm trọng đến hệ thống vận động. Khoảng 90% bệnh nhân Parkinson có biểu hiện rối loạn phát âm âm thanh (Dysphonia) ở giai đoạn sớm, bao gồm: run thanh đới, giảm cường độ âm thanh, đơn điệu cao độ và âm sắc bị nghẹt. Phương pháp sàng lọc truyền thống đòi hỏi thăm khám chuyên khoa tốn kém và khó tiếp cận tại vùng xa. 

Hệ thống này cung cấp giải pháp **sàng lọc sớm không xâm lấn qua phân tích âm học giọng nói**, giúp phát hiện các bất thường tiềm ẩn trước khi các triệu chứng vận động rõ ràng xuất hiện.

### 2. Dataset (Tập dữ liệu)
- **Tập dữ liệu chuẩn**: [UCI Machine Learning Repository - Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) được thu thập bởi Đại học Oxford (Little et al., 2007).
- **Quy mô**: 195 bản ghi âm nguyên âm kéo dài `/a/` từ **32 mã bệnh nhân** trong file (24 Parkinson, 8 khỏe mạnh — 147 / 48 bản ghi). Mã bệnh nhân lấy từ cột `name` (`phon_R01_S01_1` → `S01`).
- **Đặc trưng dùng cho model**: 16 thuộc tính MDVP tái tạo được từ audio bằng Praat — `Fo, Fhi, Flo, Jitter(%), Jitter(Abs), RAP, PPQ, DDP, Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA, NHR, HNR`. Các độ đo phi tuyến (RPDE, DFA, spread1, spread2, D2, PPE) bị loại vì không tính lại được từ audio mới; nếu giữ, lúc suy luận phải điền giá trị giả.
- **Chống rò rỉ dữ liệu (Data Leakage)**: mọi bản ghi của một bệnh nhân nằm trọn trong train hoặc test (`StratifiedGroupKFold`). GridSearchCV và nested CV cũng chia theo bệnh nhân. Hàm `group_train_test_split` kiểm tra không có bệnh nhân nào xuất hiện ở cả hai tập.
- **Audio riêng (tùy chọn)**: đặt file vào `data/raw/parkinson/{parkinson,control}/<maBenhNhan>_<n>.wav`. Khi không có `data/parkinsons.data`, hệ thống trích xuất đặc trưng (lưu ở `data/features/parkinson_features.csv`) và gom nhóm theo tiền tố tên file.

### 3. Audio Preprocessing (Tiền xử lý âm thanh)
1. Chuyển đổi định dạng âm thanh đa kênh (stereo) sang đơn kênh (mono).
2. Chuẩn hóa tần số lấy mẫu về **16,000 Hz** (tối ưu hóa cho giọng nói con người và tương thích microphone INMP441).
3. Chuẩn hóa biên độ theo đỉnh (peak normalization).
4. Khử khoảng lặng đầu/cuối bằng `librosa.effects.trim(top_db=25)`.
5. File `.webm/.ogg/.m4a` từ trình duyệt được ffmpeg chuyển sang WAV 16 kHz mono trước khi xử lý.

### 4. Feature Extraction (Trích xuất đặc trưng)
`extract_parkinson_features()` trả về 54 giá trị cho mỗi bản ghi:
- **Praat (`praat-parselmouth`)** — cùng thuật toán chu kỳ-đến-chu kỳ với các cột MDVP của UCI:
  - F0: mean, std, min, max, range (+ pitch mean / variance / range)
  - Jitter: local, local absolute, RAP, PPQ5, DDP
  - Shimmer: local, local dB, APQ3, APQ5, APQ11, DDA
  - HNR (harmonicity cc, dB) và NHR (tính từ HNR từng frame hữu thanh)
- **Librosa**: 13 MFCC (mean + std), Spectral Centroid / Bandwidth / Rolloff, ZCR, RMS Energy.
- **Metadata**: thời lượng, tỉ lệ frame hữu thanh (dùng để cảnh báo bản ghi kém chất lượng).

Giá trị không đo được (vd. audio im lặng) là `NaN` và được `SimpleImputer` xử lý, không bị gán 0.

### 5. Machine Learning (Mô hình học máy)
Pipeline: `SimpleImputer(median) → StandardScaler → Classifier`, siêu tham số chọn bằng `GridSearchCV` (CV theo bệnh nhân, scoring ROC-AUC):
- **Logistic Regression** (`class_weight="balanced"`, C ∈ {0.01, 0.1, 1, 10})
- **SVM RBF** (`class_weight="balanced"`, C × γ)
- **Random Forest** (300 cây, `class_weight="balanced"`, max_depth × min_samples_leaf)
- **Gradient Boosting** (n_estimators × max_depth × learning_rate)

Quy trình đánh giá:
1. **Test giữ riêng**: ~25% bệnh nhân (8 người / 49 bản ghi, trong đó 2 người khỏe mạnh) không tham gia train.
2. **Nested CV theo bệnh nhân** (5 fold ngoài × 4 fold trong) trên toàn bộ 32 người: mỗi bản ghi được dự đoán bởi model chưa từng thấy bệnh nhân đó.
3. Model triển khai là model có **nested-CV ROC-AUC cao nhất**. Nhãn dự đoán = xác suất Parkinson ≥ 0.5.

Pipeline tốt nhất được lưu tại `models/parkinson_classifier.joblib` (kèm tên cột, metrics, cách chia dữ liệu).

### 6. Model Evaluation (Kết quả đánh giá mô hình)

Số liệu dưới đây sinh ra từ lần chạy `train_parkinson_model()` (random_state = 42, chạy lại cho ra kết quả y hệt), lưu tại `results/parkinson_model_comparison.csv`.

**Nested CV theo bệnh nhân — 32 người, 195 bản ghi (ước lượng chính):**

| Model | Accuracy | F1 | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **64.10%** | **72.66%** | **0.797** | **63.27%** | **66.67%** |
| SVM (RBF) | 70.77% | 81.79% | 0.595 | 87.07% | 20.83% |
| Random Forest | 67.69% | 78.50% | 0.714 | 78.23% | 35.42% |
| Gradient Boosting | 73.85% | 83.50% | 0.601 | 87.76% | 31.25% |

**Test giữ riêng — 8 người, 49 bản ghi (2 người khỏe mạnh):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 71.43% | 100.00% | 62.16% | 76.67% | 0.973 | 62.16% | 100.00% |
| SVM (RBF) | 75.51% | 75.51% | 100.00% | 86.05% | 0.966 | 100.00% | 0.00% |
| Random Forest | 75.51% | 87.88% | 78.38% | 82.86% | 0.838 | 78.38% | 66.67% |
| Gradient Boosting | 63.27% | 85.19% | 62.16% | 71.88% | 0.592 | 62.16% | 66.67% |

Cách đọc:
- Tập test giữ riêng chỉ có 2 người khỏe mạnh, nên Specificity ở bảng thứ hai chỉ có thể nhận vài giá trị (0%, 66.67%, 100%…) và dao động rất mạnh. Hãy dùng số nested CV làm kết quả chính.
- Accuracy/F1 cao của SVM và Gradient Boosting chủ yếu đến từ việc đoán gần như mọi mẫu là Parkinson (75% dữ liệu là Parkinson). Specificity 20–35% cho thấy điều đó.
- README trước đây báo ROC-AUC ≈ 0.96 với cách chia ngẫu nhiên theo bản ghi; con số đó bị thổi phồng vì cùng một bệnh nhân có mặt ở cả train và test.

Biểu đồ (vẽ từ dự đoán nested CV): `results/confusion_matrix.png`, `results/roc_curve.png`.

### 7. API Endpoints (Flask RESTful API)

Các endpoint nhận audio dùng `multipart/form-data` với field **`file`**.

| Phương thức | Endpoint | Chức năng | Kết quả trả về |
|---|---|---|---|
| `GET` | `/api/health` | Trạng thái hệ thống | `{"ok": true, "parkinson_model_ready": true, "model_ready": ..., "gender_model_ready": ..., "cough_model_ready": ...}` |
| `POST` | `/api/parkinson/predict` | Sàng lọc bằng ML model | `{"ok": true, "prediction": "Control", "probability": 0.35, "model": "LogisticRegression", "features_summary": {...}, "audio_info": {"duration_sec", "sample_rate", "original_sample_rate"}, "warnings": [...]}` — chưa train: `{"ok": false, "error": "Parkinson model is not trained yet."}` |
| `POST` | `/api/parkinson/screen` | Tương thích ngược: dùng ML nếu có, ngược lại rule-based theo ngưỡng MDVP | Như trên + `risk_level_vi`, `advice`, `indicators` |
| `POST` | `/api/parkinson/train` | Huấn luyện lại (~30–60 giây) | `model, train_size, test_size, train_subjects, test_subjects, accuracy, precision, recall, f1, roc_auc, sensitivity, specificity, cv_roc_auc, nested_cv, model_path, comparison` |
| `GET` | `/api/parkinson/model-info` | Metrics của model đã train | `model, metrics, comparison, split` |

Prediction **không** tự động train model.

### 8. Web Interface (Giao diện người dùng)
Section **Parkinson Voice Screening** trong `index.html`:
- **Ghi âm trực tiếp** (5 giây, `MediaRecorder`, waveform thời gian thực) hoặc **upload** `.wav/.mp3/.flac/.ogg/.m4a/.webm`.
- **Kết quả**: Prediction (Parkinson/Control), thanh xác suất, model đã dùng và nested-CV AUC/Sensitivity/Specificity.
- **Waveform** của audio đã phân tích, **thời lượng**, **sample rate** gốc và sau xử lý.
- **Đặc trưng Praat**: F0 mean/std, Jitter, Shimmer, HNR, NHR, tỉ lệ hữu thanh; cảnh báo khi bản ghi quá ngắn hoặc không đủ đoạn hữu thanh.
- **Nút Train** hiển thị metrics nested CV và test giữ riêng.
- **Medical disclaimer** song ngữ.

### 9. Raspberry Pi Architecture (Phần cứng nhúng)
Thiết kế module suy luận độc lập tại `raspberry_pi/parkinson_inference.py`:
- **Phần cứng**:
  - Vi xử lý: **Raspberry Pi 4 Model B** (Quad-core Cortex-A72 @ 1.5 GHz, 4GB RAM) chạy Raspberry Pi OS 64-bit.
  - Microphone: **INMP441** (I2S Digital MEMS Microphone, 24-bit, SNR 61 dBA).
- **Sơ đồ chân nối (I2S Pinout)**:
  - `SCK` (BCLK) → GPIO 18 (Pin 12)
  - `WS` (LRCLK) → GPIO 19 (Pin 35)
  - `SD` (DOUT)  → GPIO 20 (Pin 38)
  - `L/R`        → GND (kênh trái)
  - `VDD`        → 3.3V (Pin 1)
  - `GND`        → GND (Pin 6)
- **Bật I2S** (`/boot/firmware/config.txt`): `dtparam=i2s=on` và `dtoverlay=googlevoicehat-soundcard`, sau đó dùng `--list-devices` để tìm ID của INMP441.
- **Chỉ suy luận**: model train trên PC rồi copy `models/parkinson_classifier.joblib` sang Pi; phiên bản scikit-learn trên Pi phải trùng với PC. Script dùng chung `predict_parkinson()` với web app nên tiền xử lý và đặc trưng giống hệt nhau.
- **Chế độ chạy**:
  - CLI: `python parkinson_inference.py` (ghi 5 giây) hoặc `--file voice.wav`, `--device <id>`, `--json`
  - HTTP server chỉ dùng thư viện chuẩn: `--server --port 5001`, với `GET /health` và `POST /predict` (multipart field `file` hoặc raw WAV body)
- **Độ trễ**: khoảng 1 giây cho 3 giây audio trên MacBook (Apple Silicon); trên Pi 4 sẽ chậm hơn, cần đo thực tế.

### 10. Limitations (Hạn chế của hệ thống)
1. **Dữ liệu rất nhỏ**: 32 người, chỉ 8 người khỏe mạnh. Nested-CV ROC-AUC ≈ 0.80 với Sensitivity/Specificity ≈ 63%/67% — chưa đủ tin cậy để dùng ngoài mục đích nghiên cứu. Việc chọn model tốt nhất trong 4 model dựa trên nested CV làm con số này lạc quan nhẹ.
2. **Lệch miền thu âm (domain shift)**: UCI ghi âm bằng micro chuyên dụng trong môi trường kiểm soát; INMP441 hay micro máy tính ở 16 kHz có đáp ứng tần số và nhiễu khác. Model chưa được kiểm định trên bản ghi từ INMP441.
3. **Chỉ nguyên âm kéo dài `/a/`**: không áp dụng cho giọng nói liên tục; người dùng phải ngân đều 3–5 giây.
4. **Đặc trưng phi tuyến bị loại**: RPDE, DFA, spread1/2, D2, PPE (vốn có giá trị phân biệt cao trong nghiên cứu gốc) không được dùng vì không tái tạo được từ audio mới.
5. **Phân biệt bệnh lý khác**: khàn giọng do tuổi tác, viêm thanh quản… có thể gây dương tính giả.

### 11. Medical Disclaimer (Tuyên bố miễn trừ trách nhiệm y tế)
> **Tiếng Việt:** Hệ thống chỉ phục vụ mục đích nghiên cứu khoa học và sàng lọc ban đầu, không phải là công cụ chẩn đoán y khoa và tuyệt đối không thay thế cho việc chẩn đoán, đánh giá chuyên môn của bác sĩ chuyên khoa thần kinh.
>
> **English:** This system is intended for research and preliminary screening purposes only. It is not a medical diagnostic tool and should not replace professional medical evaluation.

---

## Hướng dẫn Cài đặt & Sử dụng

### 1. Yêu cầu môi trường
- Python 3.9 - 3.11
- Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy ứng dụng Web
```bash
python web_app.py
```
Truy cập giao diện tại: `http://localhost:5000`

### 3. Huấn luyện model Parkinson
```bash
python -c "from audio_project.parkinson import train_parkinson_model; train_parkinson_model()"
# hoặc bấm nút Train trên giao diện / POST /api/parkinson/train
```

### 4. Chạy kiểm thử toàn diện
```bash
python -m pytest tests -q   # unit + API + Raspberry Pi tests
python verify_all.py        # kiểm tra nhanh 10 phase
```

### 5. Triển khai trên Raspberry Pi 4
```bash
# Trên Raspberry Pi:
cd raspberry_pi
python parkinson_inference.py --server --port 5001
```
