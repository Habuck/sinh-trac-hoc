import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.features import extract_features
from pathlib import Path
import librosa
import soundfile as sf   # ✅ added

# === Load and re-extract features from the dataset ===
data_path = Path("data/parkinsons.data")
df = pd.read_csv(data_path)

if "name" in df.columns:
    df = df.drop(columns=["name"])

# For compatibility: simulate real audio features
print("🎧 Extracting synthetic audio-like features...")
synthetic = []
for i in range(len(df)):
    # simulate random waveform using base jitter/shimmer scaling
    jitter = df.iloc[i].get("MDVP:Jitter(%)", 0.01)
    shimmer = df.iloc[i].get("MDVP:Shimmer", 0.05)
    f0 = df.iloc[i].get("MDVP:Fo(Hz)", 150)
    sr = 22050
    y = np.sin(2 * np.pi * f0 * np.linspace(0, 1, sr))
    y = y + np.random.randn(sr) * (jitter + shimmer)
    tmp_path = f"temp_audio_{i}.wav"

    sf.write(tmp_path, y, sr)  # ✅ fixed line

    features = extract_features(tmp_path)
    features["status"] = df.iloc[i]["status"]
    synthetic.append(features)
    os.remove(tmp_path)

audio_df = pd.DataFrame(synthetic)
print("✅ Feature extraction complete:", audio_df.shape)

X = audio_df.drop(columns=["status"])
y = audio_df["status"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/parkinsons_best_model.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("🎯 Model retrained and saved for real-audio prediction!")
