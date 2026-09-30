

<p align="center">
  <img src="https://img.shields.io/badge/👨‍💻 Developer-Prince%20S-007bff?style=for-the-badge&logo=github">
  <img src="https://img.shields.io/badge/📊 Accuracy-94.87%25-27ae60?style=for-the-badge&logo=google-analytics">
  <img src="https://img.shields.io/badge/🖥️ Framework-Gradio-f39c12?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/🤖 Model-LightGBM-16a085?style=for-the-badge&logo=lightning">
  <img src="https://img.shields.io/badge/📄 License-MIT-7f8c8d?style=for-the-badge">
</p>



# 🎙️ Parkinson Voice Detection App
Developed by Prince S


This project predicts early signs of Parkinson’s disease using **machine learning + voice analysis**.  
Users can record or upload a short audio sample, and the model instantly displays the **Parkinson’s risk percentage**.

The system is built with a **LightGBM model** and a clean **Gradio interface**, making it fast, easy, and ideal for early detection.

---

## 🎥 App Demo
<p align="center">
  <img src="Screenshot 2025-11-12 233556.png" width="700">
</p>

---

## 🧠 Features
- 🎤 Record or upload voice samples  
- ⚡ Instant Parkinson’s prediction  
- 📊 Displays confidence percentage  
- 🧠 LightGBM model trained on MFCC, jitter, shimmer, and spectral features  
- 🔊 Audio processing with Librosa & Parselmouth  
- 🖥️ Modern & simple Gradio interface  
- 💻 Works locally on any computer  

---

## ⚙️ Technologies Used
- Python  
- Gradio  
- LightGBM  
- Librosa  
- Parselmouth  
- NumPy / Scikit-Learn  
- UCI Parkinson’s Voice Dataset  

---

## 📦 Installation (Run Locally)
1. Clone the repository:
   git clone https://github.com/prince-s19/Parkinson-Voice-Detection-App.git

   cd Parkinson-Voice-Detection-App


2. Create & activate virtual environment:

     python -m venv .venv
   
   ..venv\Scripts\activate


4. Install dependencies:
   
   pip install -r requirements.txt


6. Run the application:


  python ParkinsonVoiceDetector.py

5. Open Gradio link in browser (example):
http://127.0.0.1:7860

---

## 📈 Model Performance
<p align="center">
  <img src="Screenshot 2025-11-12 193246.png" width="600">
</p>

---

## 🖼️ Sample Predictions

### 🧠 Parkinson’s Detected
<p align="center">
  <img src="Screenshot 2025-11-12 233337.png" width="650">
</p>

### ✅ Voice Appears Normal
<p align="center">
  <img src="Screenshot 2025-11-12 233301.png" width="650">
</p>

---

## 🔍 How It Works
1. User records or uploads a 3–5 second voice sample  
2. Audio is trimmed, normalized, and processed  
3. Features such as MFCC, jitter, shimmer, spectral contrast are extracted  
4. LightGBM model analyzes the features  
5. The system outputs the **risk percentage**  

---

## 🎯 Purpose
The goal is to create an **accessible, non-invasive, low-cost** solution for early Parkinson’s screening using voice biomarkers.

---

## ✨ Future Improvements
- Cloud deployment (Render / Hugging Face)  
- Dedicated mobile app  
- Doctor dashboard for patient tracking  
- Support for multiple languages  

---

## 👤 Developer
Prince S 


