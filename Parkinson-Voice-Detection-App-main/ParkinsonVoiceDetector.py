import os
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from utils.features import extract_features

# === Paths ===
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Trained Model ===
try:
    model = joblib.load("models/parkinsons_best_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    scaler = joblib.load("models/scaler.pkl")
    MODEL_ACCURACY = 94.87  # From your training
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    raise SystemExit(e)

# === Prediction Function ===
def predict_parkinson(audio_path):
    """
    Takes an audio file path or tuple (sr, np.ndarray) and predicts Parkinson's probability.
    """
    try:
        # If the audio is raw data (tuple), save temporarily
        if isinstance(audio_path, tuple):
            y, sr = audio_path
            temp_path = os.path.join(UPLOAD_FOLDER, "temp_recording.wav")
            sf.write(temp_path, y, sr)
            audio_path = temp_path

        # Extract features
        features = extract_features(audio_path)
        X = pd.DataFrame([features])
        X = X.reindex(columns=feature_names, fill_value=0)
        X_scaled = scaler.transform(X)

        # Predict (average over 3 runs for stability)
        preds = [model.predict(X_scaled)[0] for _ in range(3)]
        probas = [model.predict_proba(X_scaled)[0][1] for _ in range(3)]
        final_pred = int(round(np.mean(preds)))
        avg_proba = float(np.mean(probas)) * 100

        if final_pred == 1:
            msg = f"🧠 Parkinson’s Detected!\n\nConfidence: {avg_proba:.2f}%"
            color = "red"
        else:
            msg = f"✅ Voice Appears Normal\n\nConfidence: {avg_proba:.2f}%"
            color = "green"

        return msg, f"Model Accuracy: {MODEL_ACCURACY}%"
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# === Gradio UI ===
description = """
### 🎙 Parkinson’s Voice Detector
Detects Parkinson’s disease from a short voice clip using AI.<br>
Upload or record a 5-second sample — ideally saying "aaaah" clearly.<br><br>
"""

app = gr.Interface(
    fn=predict_parkinson,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Record or Upload Voice"),
    outputs=[
        gr.Textbox(label="Prediction Result", lines=4, show_copy_button=True),
        gr.Textbox(label="Model Info", interactive=False)
    ],
    title="🧠 Parkinson’s Voice Detector",
    description=description,
    theme=gr.themes.Soft(primary_hue="blue"),
    examples=None,
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch(server_port=7860, show_api=False, share=False)
