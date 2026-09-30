# Parkinson's Disease Voice Screening — Đặc tả mở rộng cho Đồ án 2.1

> **Dành cho AI / coding agent đọc file này:**
> Đây là đặc tả kỹ thuật cho việc **mở rộng** một dự án đã tồn tại, **không phải** yêu cầu tạo dự án mới. Hãy đọc toàn bộ file này trước khi viết bất kỳ dòng code nào, sau đó thực hiện tuần tự theo các Phase ở mục 22.

## 0. Bối cảnh

Dự án hiện có: **Đồ án 2.1**, repository GitHub `Habuck/Do-an-2.1`.

Repository hiện tại đã có một hệ thống phân tích âm thanh bằng Python/Flask, bao gồm:

```
web_app.py
audio_project/core.py
audio_project/gender.py
audio_project/emotion.py
audio_project/cough.py
audio_project/depression.py
audio_project/pronunciation.py
audio_project/voice.csv
data/raw
data/processed
data/features
index.html
requirements.txt
```

**Không được tạo một project độc lập mới.**

Hãy **phân tích, kế thừa và phát triển trực tiếp từ kiến trúc/code hiện tại của Đồ án 2.1** để mở rộng thành một module mới:

**Tên đề tài (VN):** Hệ thống sàng lọc bệnh Parkinson dựa trên phân tích giọng nói và Machine Learning
**Tên đề tài (EN):** Voice-Based Parkinson's Disease Screening Using Machine Learning

---

## 1. Mục tiêu phát triển

Biến hệ thống audio hiện tại của Đồ án 2.1 thành một hệ thống có thêm khả năng theo pipeline sau:

```
Audio recording
  ↓
Audio preprocessing
  ↓
Voice feature extraction
  ↓
Parkinson's model
  ↓
Prediction
  ↓
Risk / probability
  ↓
Web interface
```

- Người dùng có thể **upload** hoặc **ghi âm giọng nói trực tiếp**.
- Hệ thống phân tích các đặc trưng của giọng nói và đưa ra kết quả sàng lọc: **Control** hoặc **Parkinson's**.
- Đây là hệ thống **screening / research**, **không được gọi là công cụ chẩn đoán y khoa**.

---

## 2. Quy tắc quan trọng

### Không phá vỡ hệ thống hiện tại

- Không được xóa hoặc viết lại các module hiện có nếu không cần thiết.
- Các chức năng hiện tại sau **phải tiếp tục hoạt động**:
  - `emotion`
  - `gender`
  - `cough`
  - `pronunciation`
  - `depression`
  - audio processing (core)

### Cách tích hợp

- Parkinson phải được xây dựng thành **module độc lập nhưng tích hợp vào hệ thống hiện tại**.
- Ưu tiên tái sử dụng:
  - audio preprocessing
  - Librosa
  - model loading
  - Flask API
  - frontend
  - cấu trúc `audio_project`
  - hệ thống `data/`
  - `requirements.txt` hiện tại

---

## 3. Trước khi code

Trước tiên phải **đọc và phân tích toàn bộ repository hiện tại**. Đặc biệt kiểm tra:

```
web_app.py
audio_project/core.py
audio_project/voice.csv
audio_project/gender.py
audio_project/emotion.py
audio_project/cough.py
audio_project/depression.py
audio_project/pronunciation.py
requirements.txt
index.html
```

Cần xác định:

1. Pipeline audio hiện tại hoạt động thế nào.
2. Model hiện tại được train như thế nào.
3. Feature extraction đang sử dụng những gì.
4. Model được lưu ở đâu.
5. Flask API hiện tại tổ chức thế nào.
6. Frontend gọi API ra sao.
7. Có thể tái sử dụng phần nào cho Parkinson.

> **Không được đoán kiến trúc. Phải đọc code hiện tại trước khi sửa.**

---

## 4. Parkinson module

Tạo module mới: `audio_project/parkinson.py`

Module này chịu trách nhiệm cho toàn bộ pipeline sau:

```
load Parkinson dataset
  ↓
preprocess audio
  ↓
extract features
  ↓
train model
  ↓
evaluate model
  ↓
save model
  ↓
predict new audio
```

Không đưa logic Parkinson vào `core.py` nếu không cần thiết.

---

## 5. Dataset

Thiết kế pipeline để hỗ trợ dataset Parkinson voice.

Ưu tiên các dataset nghiên cứu Parkinson Voice có:

- audio recordings
- Parkinson / control labels
- patient / subject identifier (nếu có)

Nếu dataset hiện tại chưa chứa Parkinson audio, tạo cấu trúc:

```
data/
├── raw/
│   └── parkinson/
│       ├── control/
│       └── parkinson/
│
├── processed/
│   └── parkinson/
│
└── features/
    └── parkinson_features.csv
```

> Không commit dataset y tế có vấn đề bản quyền/quyền riêng tư vào repository nếu không được phép.

---

## 6. Tránh Data Leakage (bắt buộc)

Nếu một bệnh nhân có nhiều recording:

```
Patient A
├── audio1
├── audio2
└── audio3
```

thì **toàn bộ recording của Patient A phải nằm trong cùng một split**.

Không được:

```
Patient A audio1 → train
Patient A audio2 → test
```

Nếu dataset có patient ID, sử dụng:

- `GroupShuffleSplit`, hoặc
- `GroupKFold`

để đảm bảo đánh giá chính xác.

---

## 7. Audio preprocessing

Tái sử dụng preprocessing hiện có của Đồ án 2.1 nếu phù hợp. Pipeline:

```
Input Audio
  ↓
Convert to mono
  ↓
Resample
  ↓
Normalize amplitude
  ↓
Remove silence
  ↓
Optional noise reduction
  ↓
Feature extraction
```

Chuẩn hóa sample rate, ưu tiên **16 kHz** nếu phù hợp với dataset.

---

## 8. Parkinson voice features

Trích xuất các đặc trưng liên quan đến giọng nói Parkinson.

### MFCC
- MFCC 1–13
- mean
- std

### Fundamental Frequency
- F0 mean
- F0 std
- F0 min
- F0 max

### Voice quality
Cố gắng trích xuất:
- Jitter
- Shimmer
- HNR

Nếu Librosa không cung cấp trực tiếp jitter/shimmer một cách phù hợp, **sử dụng phương pháp/library phù hợp thay vì tạo công thức giả**.

### Spectral features
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy

### Pitch-related features
- Pitch mean
- Pitch variance
- Pitch range

Tạo **feature vector cho mỗi recording**.

---

## 9. Machine Learning

Implement và so sánh các model:

- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting / XGBoost nếu phù hợp

Pipeline:

```
Audio
  ↓
Features
  ↓
Imputation (nếu cần)
  ↓
StandardScaler
  ↓
Classifier
```

Lưu pipeline bằng `joblib` để inference sử dụng đúng preprocessing lúc training.

Model nên được lưu tại:

```
models/
└── parkinson_classifier.joblib
```

---

## 10. Deep Learning — tùy chọn

Sau khi traditional ML hoạt động ổn định, có thể bổ sung:

```
Audio
  ↓
Mel-Spectrogram
  ↓
CNN
  ↓
Parkinson / Control
```

- **Không được ưu tiên CNN trước khi pipeline ML truyền thống hoạt động ổn định.**
- Nếu dataset nhỏ, phải đánh giá nguy cơ overfitting.

---

## 11. Evaluation

Bắt buộc tính:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Sensitivity
- Specificity

Tạo:
- Confusion Matrix
- ROC Curve

Đặc biệt báo cáo **Sensitivity** và **Specificity** vì đây là bài toán screening.

> Không tự tạo số liệu. Tất cả số liệu phải lấy từ quá trình training/evaluation thực tế.

---

## 12. Model comparison

Tạo bảng tự động với các cột sau:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| SVM | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |
| CNN (optional) | | | | | |

Lưu kết quả vào:

```
results/
└── parkinson_model_comparison.csv
```

và nếu phù hợp:

```
results/
├── confusion_matrix.png
└── roc_curve.png
```

---

## 13. Flask API — Prediction

Tích hợp Parkinson vào `web_app.py`. Thêm endpoint:

```
POST /api/parkinson/predict
```

**Input:** audio file

**Output (thành công):**
```json
{
  "ok": true,
  "prediction": "Parkinson",
  "probability": 0.82,
  "model": "SVM"
}
```

**Output (model chưa tồn tại):**
```json
{
  "ok": false,
  "error": "Parkinson model is not trained yet."
}
```

> Không tự động train model mỗi lần người dùng prediction.

---

## 14. Flask API — Training

Thêm:

```
POST /api/parkinson/train
```

API phải trả về:

- train size
- test size
- model
- accuracy
- precision
- recall
- f1
- roc_auc
- sensitivity
- specificity
- model path

---

## 15. Flask API — Health

Mở rộng:

```
GET /api/health
```

để trả thêm:

```json
{
  "parkinson_model_ready": true
}
```

> Không làm mất các trạng thái model hiện tại.

---

## 16. Frontend

Tích hợp thêm một section mới vào `index.html`: **Parkinson Voice Screening**.

Giao diện (mockup):

```
┌─────────────────────────────────────┐
│      Parkinson Voice Screening       │
│                                       │
│         [ Upload Audio ]             │
│                                       │
│               hoặc                   │
│                                       │
│       [ 🎙 Start Recording ]         │
│                                       │
│         [ Analyze Voice ]            │
└─────────────────────────────────────┘
```

Sau khi phân tích, hiển thị **Voice Analysis Result**:

- Prediction: `Parkinson` / `Control`
- Screening Probability: ví dụ `82%`
- Model: ví dụ `SVM`

Hiển thị thêm:
- waveform
- duration
- sample rate
- extracted feature summary (nếu phù hợp)
- confidence/probability

---

## 17. Medical disclaimer

Trong giao diện Parkinson **phải hiển thị** disclaimer sau:

**Tiếng Anh:**
> This system is intended for research and preliminary screening purposes only. It is not a medical diagnostic tool and should not replace professional medical evaluation.

**Tiếng Việt:**
> Hệ thống chỉ phục vụ mục đích nghiên cứu và sàng lọc ban đầu, không phải công cụ chẩn đoán y khoa và không thay thế đánh giá của chuyên gia y tế.

---

## 18. Raspberry Pi — giai đoạn mở rộng

Sau khi hệ thống PC/web hoạt động ổn định, thiết kế module:

```
Raspberry Pi
  ↓
INMP441 microphone
  ↓
Record voice
  ↓
Preprocessing
  ↓
Feature extraction
  ↓
Parkinson model
  ↓
Result
```

Tạo:

```
raspberry_pi/
└── parkinson_inference.py
```

Code phải **tối ưu cho inference**, không train model trên Raspberry Pi.

---

## 19. Cấu trúc project sau khi phát triển

Giữ cấu trúc hiện tại nhưng mở rộng thành:

```
Do-an-2.1/
│
├── audio_project/
│   ├── core.py
│   ├── gender.py
│   ├── emotion.py
│   ├── cough.py
│   ├── depression.py
│   ├── pronunciation.py
│   ├── parkinson.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── raw_medical/
│
├── models/
│   └── parkinson_classifier.joblib
│
├── results/
│   ├── parkinson_model_comparison.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── raspberry_pi/
│   └── parkinson_inference.py
│
├── notebooks/
│   └── parkinson_analysis.ipynb
│
├── web_app.py
├── index.html
├── requirements.txt
└── README.md
```

> Không bắt buộc phải tạo notebook nếu project hiện tại không sử dụng notebook; có thể triển khai bằng Python scripts.

---

## 20. README

Cập nhật README để mô tả rõ cấu trúc các module:

```
Đồ án 2.1
│
├── Audio Analysis
├── Gender Recognition
├── Emotion Recognition
├── Pronunciation Analysis
├── Cough Analysis
├── Depression Screening
└── Parkinson's Disease Voice Screening
```

Phần Parkinson trong README phải có các mục sau:

1. Problem statement
2. Dataset
3. Audio preprocessing
4. Feature extraction
5. Machine Learning
6. Model evaluation
7. API
8. Web interface
9. Raspberry Pi architecture
10. Limitations
11. Medical disclaimer

---

## 21. Quan trọng nhất: phát triển từ code hiện tại

**Không được làm kiểu:**

```
Project cũ
  ↓
bỏ đi
  ↓
Project Parkinson mới
```

**Mà phải làm:**

```
                    ĐỒ ÁN 2.1
                        │
                 Audio Processing
                        │
          ┌─────────────┴─────────────┐
          │                           │
   Existing modules            Parkinson Module
          │                           │
 ┌────────┼────────┐                  │
Gender  Emotion  Cough           Parkinson
Pronunciation  Depression             │
          └─────────────┬─────────────┘
                         │
                 ML Classification
                         │
                     Flask API
                         │
                   Web Interface
                         │
                   Raspberry Pi
```

Mục tiêu là biến **Parkinson Voice Screening** thành một **module phát triển tiếp theo của Đồ án 2.1**, tận dụng tối đa code và kiến trúc đã có.

---

## 22. Quy trình thực hiện (bắt buộc theo thứ tự)

**Không viết toàn bộ code một lần.** Thực hiện theo thứ tự:

| Phase | Nội dung |
|---|---|
| 1 | Phân tích repository hiện tại |
| 2 | Phân tích dataset và thiết kế Parkinson module |
| 3 | Implement preprocessing + feature extraction |
| 4 | Implement training |
| 5 | Evaluation và chống data leakage |
| 6 | Lưu model + inference |
| 7 | Tích hợp Flask API |
| 8 | Tích hợp frontend |
| 9 | Testing toàn bộ hệ thống |
| 10 | Raspberry Pi deployment |

Sau **mỗi phase**, AI phải:

- [ ] Chạy test
- [ ] Kiểm tra lỗi
- [ ] Không phá vỡ module cũ
- [ ] Báo cáo file nào được tạo/sửa
- [ ] Giải thích ngắn gọn thay đổi

> **Chỉ chuyển sang phase tiếp theo khi phase hiện tại hoạt động.**
