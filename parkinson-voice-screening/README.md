# ParkiScan — Ứng dụng Desktop Sàng lọc Bệnh Parkinson & Sinh Trắc Học Giọng Nói
## Đề tài 19: Voice-based Parkinson's Disease Screening & Voice Biometrics (Desktop GUI)

Ứng dụng độc lập chạy **Local (Native Desktop GUI)** trên nền tảng Python / Tkinter, **không cần chạy qua web hay trình duyệt**, tương thích hoàn toàn trên cả **Windows PC** và **Raspberry Pi 4 Model B** kết hợp **Microphone INMP441 (I2S MEMS)**.

Dự án kế thừa và tích hợp các tiện ích sinh trắc học giọng nói cốt lõi từ Đồ án 2.1 (`sinh-trac-hoc`), kết hợp thuật toán Học máy SVM phục vụ sàng lọc bệnh Parkinson.

---

## 📌 Thông tin Đề tài
- **Chủ đề**: 19. Nhận diện giọng nói để hỗ trợ chẩn đoán rối loạn giọng nói/Parkinson (Voice-based Parkinson's Disease Screening).
- **Phần cứng**: Microphone module INMP441 (I2S MEMS) + Raspberry Pi 4 (hoặc Microphone máy tính).
- **Công nghệ**:
  - Trích xuất đặc trưng âm học: **MFCC** (13 hệ số), **Jitter** (chu kỳ), **Shimmer** (biên độ), **HNR/NHR**, **F0** (pYIN), **PPE**.
  - Phân loại học máy: **SVM (Support Vector Machine - RBF Kernel)** huấn luyện trên bộ dữ liệu chuẩn **UCI Parkinson's Voice Dataset** (ROC-AUC > 0.96).
  - Tích hợp Sinh trắc học: **Nhận diện Giới tính (Gender)** (KNN + 6 đặc trưng voice.csv), **Phân tích Cảm xúc (Speech Emotion)** (DTW-MFCC), **Phân loại Tiếng ho & Hô hấp (Cough)**, **Trắc lượng Âm vực**.

---

## 🌟 4 Tab Chức năng Chuyên sâu của Ứng dụng Desktop Local

### Tab 1: 🎙️ Sàng lọc & Phân tích
- **Ghi âm trực tiếp / Tải file**: Thu âm giọng nói mẫu (nguyên âm /a/ ngân dài) qua PC Microphone hoặc Raspberry Pi INMP441.
- **Dạng sóng âm thanh (Waveform)**: Vẽ biểu đồ sóng âm thời gian thực kèm thời lượng và tần số lấy mẫu.
- **Nghe lại âm thanh (Playback)**: Nút phát lại audio để người dùng nghe và kiểm tra chất lượng giọng thu.
- **Kết quả Sàng lọc Parkinson**: Thẻ báo mức rủi ro (Cao / Trung bình / Thấp) kèm xác suất % và lời khuyên y khoa.
- **Sinh trắc học tóm tắt**: 4 huy hiệu kết quả nhận diện tự động (Giới tính, Cảm xúc, Hô hấp, Âm vực).
- **Bảng 12 thông số âm học nhanh**: F0 Mean, Std, Jitter, Shimmer, HNR, NHR, Tốc độ phát âm, Tỷ lệ ngắt nghỉ.

### Tab 2: 🧬 Sinh trắc học Giọng nói (Kế thừa từ Đồ án 2.1)
- **Nhận diện Giới tính (Gender Recognition)**:
  - Dự đoán Nam / Nữ kèm độ tin cậy %.
  - Thước đo phân bố xác suất Nam vs Nữ.
  - Bảng 6 thông số âm học chuẩn `voice.csv`: Tần số trung bình (`meanfreq`), Độ lệch chuẩn (`sd`), Trọng tâm phổ (`centroid`), Tần số cơ bản (`meanfun`), Khoảng tứ phân vị (`IQR`), Tần số trung vị (`median`).
- **Phân tích Cảm xúc Giọng nói (Speech Emotion Analysis)**:
  - Cảm xúc chủ đạo: Bình thường, Vui vẻ, Buồn, Tức giận, Lo âu / Sợ hãi, Ngạc nhiên.
  - Thanh đo phân bố phần trăm xác suất của cả 6 sắc thái cảm xúc.
- **Phân tích Tiếng ho & Đường hô hấp (Cough Analysis)**:
  - Phân loại: Mẫu bình thường (không ho), Ho khan, Ho có đờm.
- **Trắc lượng Âm vực & Thanh đới (Voice Profile)**:
  - Phân loại chất giọng: Trầm (Bass/Baritone), Trung bình (Tenor/Alto), Cao (Soprano).
  - Tần số cơ bản trung bình và độ mở rộng phổ tần số.

### Tab 3: 📊 Phổ đặc trưng & MFCC
- Biểu đồ phân tích 4 nhóm âm học chuyên sâu:
  1. Tần số cơ bản F0 & HNR (dB).
  2. Bất ổn định chu kỳ Jitter (Local, RAP, PPQ5, DDP).
  3. Bất ổn định biên độ Shimmer (Local, dB, APQ3, APQ5).
  4. 13 Hệ số MFCC Mean (Mel-Frequency Cepstral Coefficients).

### Tab 4: 🤖 Huấn luyện SVM (UCI)
- Huấn luyện lại mô hình SVM RBF Kernel trực tiếp từ dữ liệu `data/parkinsons.data`.
- Tự động tối ưu hóa siêu tham số ($C, \gamma$) bằng `GridSearchCV` với 5-Fold Stratified Cross-Validation.
- Xem ma trận nhầm lẫn (Confusion Matrix), điểm ROC-AUC (> 0.96) và Báo cáo phân loại (Classification Report).

---

## 🚀 Hướng dẫn Khởi chạy

### Cách 1: Trên Windows PC
1. **Cài đặt thư viện** (nếu chưa có):
   ```bash
   pip install -r requirements.txt
   ```
2. **Khởi chạy ứng dụng**:
   - Nhấp đúp chuột vào file **`Chay_App_Local.bat`**
   - Hoặc chạy từ Terminal:
     ```bash
     python app.py
     ```

### Cách 2: Trên Raspberry Pi 4 + Microphone INMP441
1. **Nối dây theo chuẩn I2S**:
   - `VDD` ➔ Pin 1 (3.3V) | `GND` ➔ Pin 6 (GND)
   - `SD` ➔ Pin 38 (GPIO 20) | `SCK` ➔ Pin 12 (GPIO 18) | `WS` ➔ Pin 35 (GPIO 19) | `L/R` ➔ Pin 9 (GND)
2. **Kích hoạt I2S**: Thêm `dtoverlay=googlevoicehat-soundcard` vào `/boot/config.txt` và `sudo reboot`.
3. **Khởi chạy**:
   ```bash
   bash chay_raspberry_pi.sh
   # hoặc:
   python3 app.py
   ```

---

## 📋 Tuyên bố Miễn trừ Trách nhiệm Y tế (Medical Disclaimer)
> Hệ thống này chỉ phục vụ mục đích học tập, nghiên cứu khoa học và sàng lọc hỗ trợ ban đầu. Đây không phải là thiết bị chẩn đoán y khoa chuyên nghiệp và tuyệt đối không thay thế cho việc thăm khám, đánh giá lâm sàng của các bác sĩ chuyên khoa thần kinh.
