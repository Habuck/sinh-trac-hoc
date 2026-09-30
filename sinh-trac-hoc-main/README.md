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
- **Quy mô**: 195 bản ghi âm nguyên âm kéo dài `/a/` từ 31 đối tượng (23 người bệnh Parkinson, 8 người đối chứng khỏe mạnh).
- **Đặc trưng**: 22 thuộc tính âm học bao gồm các biến thể tần số, jitter, shimmer, tỉ lệ ồn-hài âm và các độ đo phi tuyến (RPDE, DFA, PPE).
- **Chống rò rỉ dữ liệu (Data Leakage)**: Sử dụng kỹ thuật chia mẫu theo nhóm bệnh nhân (`GroupShuffleSplit` / `StratifiedKFold`) đảm bảo không rò rỉ bản ghi cùng đối tượng vào cả tập train và test.

### 3. Audio Preprocessing (Tiền xử lý âm thanh)
1. Chuyển đổi định dạng âm thanh đa kênh (stereo) sang đơn kênh (mono).
2. Chuẩn hóa tần số lấy mẫu về **16,000 Hz** (tối ưu hóa cho giọng nói con người và tương thích microphone INMP441).
3. Khử khoảng lặng (silence trimming) với ngưỡng năng lượng thích ứng bằng `librosa.effects.trim(top_db=25)`.
4. Chuẩn hóa biên độ năng lượng (Peak Normalization & RMS Normalization).

### 4. Feature Extraction (Trích xuất đặc trưng)
Hệ thống trích xuất **51 đặc trưng âm học** chuyên sâu:
- **Tần số cơ bản F0**: Mean, Std, Min, Max, Range (sử dụng giải thuật YIN / PYIN).
- **Jitter (Bất ổn định chu kỳ)**:
  - `Jitter (local)`: Sai phân chu kỳ lân cận.
  - `Jitter (RAP)`: Relative Average Perturbation (bộ lọc 3 điểm).
  - `Jitter (PPQ5)`: Period Perturbation Quotient (bộ lọc 5 điểm).
  - `Jitter (DDP)`: Difference of Differences of Periods.
- **Shimmer (Bất ổn định biên độ)**:
  - `Shimmer (local)`, `Shimmer (dB)`, `Shimmer (APQ3)`, `Shimmer (APQ5)`.
- **Tỷ lệ Hài âm & Nhiễu**:
  - `HNR (Harmonics-to-Noise Ratio)` đo độ trong của giọng nói.
  - `NHR (Noise-to-Harmonics Ratio)` đo độ nhiễu dây thanh.
- **Đặc trưng Phổ & Âm sắc**:
  - 13 hệ số MFCC (Mean & Standard Deviation = 26 giá trị).
  - Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate (ZCR), RMS Energy.

### 5. Machine Learning (Mô hình học máy)
Pipeline học máy được chuẩn hóa bằng `StandardScaler` và `GridSearchCV`:
- **Support Vector Machine (SVM)** với RBF kernel tối ưu hóa ($C$, $\gamma$).
- **Logistic Regression** (L2 regularization baseline).
- **Random Forest Classifier** (100 cây quyết định, chống overfitting).
- **Gradient Boosting Classifier** (tối ưu hóa sai số tuần tự).

Toàn bộ pipeline được tuần tự hóa và lưu trữ tại `models/parkinson_classifier.joblib`.

### 6. Model Evaluation (Kết quả đánh giá mô hình)

Kết quả thử nghiệm thực tế trên tập kiểm tra độc lập (Subject-Aware Evaluation):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| **SVM (RBF Kernel)** | **89.74%** | **93.10%** | **93.10%** | **93.10%** | **0.9621** | **93.10%** | **80.00%** |
| Random Forest | 92.31% | 93.33% | 96.55% | 94.92% | 0.9621 | 96.55% | 80.00% |
| Logistic Regression | 92.31% | 93.33% | 96.55% | 94.92% | 0.9241 | 96.55% | 80.00% |
| Gradient Boosting | 84.62% | 89.66% | 89.66% | 89.66% | 0.8931 | 89.66% | 70.00% |

- Biểu đồ đánh giá được tự động xuất tại:
  - `results/confusion_matrix.png`
  - `results/roc_curve.png`
  - `results/parkinson_model_comparison.csv`

### 7. API Endpoints (Flask RESTful API)

| Phương thức | Endpoint | Chức năng | Tham số / Body | Kết quả trả về |
|---|---|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái hệ thống | Không | `{"ok": true, "parkinson_model_ready": true, ...}` |
| `POST` | `/api/parkinson/predict` | Sàng lọc file âm thanh | `audio` (file wav/webm/mp3) | `{"ok": true, "prediction": "Parkinson", "probability": 0.82, "model": "SVM"}` |
| `POST` | `/api/parkinson/screen` | Sàng lọc chi tiết + đặc trưng | `audio` (file) | Kèm toàn bộ 51 đặc trưng, F0, Jitter, Shimmer |
| `POST` | `/api/parkinson/train` | Huấn luyện lại mô hình | Không | Toàn bộ metrics (Accuracy, ROC-AUC, Sensitivity, ...) |

### 8. Web Interface (Giao diện người dùng)
Giao diện trực quan tích hợp trong `index.html`:
- **Ghi âm trực tiếp**: Sử dụng Web Audio API và `MediaRecorder`, vẽ biểu đồ sóng âm thời gian thực (Waveform Oscilloscope).
- **Tải lên tệp**: Hỗ trợ `.wav`, `.mp3`, `.ogg`, `.webm`, `.flac`.
- **Hiển thị kết quả trực quan**: Thẻ chỉ số nguy cơ (Risk Badge), thanh tiến trình xác suất (Probability Bar), bảng tổng kết các đặc trưng âm học quan trọng (F0, Jitter, Shimmer, HNR).
- **Điều khiển huấn luyện**: Nút "Huấn luyện lại mô hình" trực tiếp trên giao diện web.

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
- **Tối ưu hóa**: Trọng lượng nhẹ, chỉ thực thi suy luận (`inference only`), độ trễ phản hồi < 0.5 giây. Hỗ trợ chạy chế độ CLI hoặc REST Microservice trên cổng `:5001`.

### 10. Limitations (Hạn chế của hệ thống)
1. **Quy mô tập dữ liệu**: Tập dữ liệu UCI tương đối nhỏ (195 bản ghi), chủ yếu tập trung vào nguyên âm ngân dài `/a/`, chưa bao gồm giọng nói liên tục hay đoạn hội thoại tự do.
2. **Nhiễu môi trường**: Microphone MEMS có thể nhạy cảm với tạp âm phòng và tiếng ồn nền nếu không được trang bị bộ lọc cách âm.
3. **Phân biệt bệnh lý đồng mắc**: Một số triệu chứng khàn giọng, run thanh đới do tuổi già hoặc viêm thanh quản cấp tính có thể gây dương tính giả nếu không có bác sĩ chuyên khoa thẩm định.

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

### 3. Chạy kiểm thử toàn diện
```bash
python verify_all.py
```

### 4. Triển khai trên Raspberry Pi 4
```bash
# Trên Raspberry Pi:
cd raspberry_pi
python parkinson_inference.py --server --port 5001
```
