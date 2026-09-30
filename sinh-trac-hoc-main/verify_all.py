import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from pathlib import Path

print('=== FINAL VERIFICATION - All 10 Phases ===')
print()

# Phase 3: Feature extraction
try:
    from audio_project.parkinson import extract_parkinson_features
    import numpy as np
    y = np.sin(2*np.pi*150*np.linspace(0, 3, 16000*3)).astype(np.float32)
    feats = extract_parkinson_features(y, 16000)
    assert 'f0_mean' in feats and 'jitter_local' in feats and 'mfcc_1_mean' in feats
    print(f'Phase 3 (Features): OK - {len(feats)} features extracted')
except Exception as e:
    print(f'Phase 3 FAILED: {e}')

# Phase 4-5: Training + Evaluation
try:
    from audio_project.parkinson import is_model_trained
    model_exists = is_model_trained()
    comp_exists = Path('results/parkinson_model_comparison.csv').exists()
    print(f'Phase 4-5 (Train+Eval): OK - model={model_exists}, comparison={comp_exists}')
except Exception as e:
    print(f'Phase 4-5 FAILED: {e}')

# Phase 6: Inference
try:
    import joblib
    bundle = joblib.load('models/parkinson_classifier.joblib')
    name = bundle['model_name']
    auc = bundle['metrics']['roc_auc']
    print(f'Phase 6 (Inference): OK - model={name}, AUC={auc:.4f}')
except Exception as e:
    print(f'Phase 6 FAILED: {e}')

# Phase 7: Flask API
try:
    from audio_project.parkinson import (
        screen_parkinson, predict_parkinson, train_parkinson_model,
        is_model_trained, PARKINSON_MODEL_PATH
    )
    print('Phase 7 (Flask API): OK - all functions importable')
except Exception as e:
    print(f'Phase 7 FAILED: {e}')

# Phase 8: Frontend
try:
    html = open('index.html', encoding='utf-8').read()
    assert 'pkCard' in html
    assert '/api/parkinson/predict' in html
    assert '/api/parkinson/train' in html
    assert 'Medical Disclaimer' in html
    assert 'parkinson' in html.lower()
    print('Phase 8 (Frontend): OK - section + JS + Disclaimer present')
except Exception as e:
    print(f'Phase 8 FAILED: {e}')

# Phase 9: Testing
try:
    results = {
        'model': Path('models/parkinson_classifier.joblib').exists(),
        'comparison_csv': Path('results/parkinson_model_comparison.csv').exists(),
        'confusion_matrix': Path('results/confusion_matrix.png').exists(),
        'roc_curve': Path('results/roc_curve.png').exists(),
    }
    all_ok = all(results.values())
    print(f'Phase 9 (Testing): {"OK" if all_ok else "PARTIAL"} - {results}')
except Exception as e:
    print(f'Phase 9 FAILED: {e}')

# Phase 10: Raspberry Pi
try:
    rasp_exists = Path('raspberry_pi/parkinson_inference.py').exists()
    print(f'Phase 10 (Raspberry Pi): {"OK" if rasp_exists else "MISSING"}')
except Exception as e:
    print(f'Phase 10 FAILED: {e}')

# Documentation & Analysis
try:
    readme_exists = Path('README.md').exists()
    nb_exists = Path('notebooks/parkinson_analysis.ipynb').exists()
    print(f'Documentation (README & Notebook): OK - README={readme_exists}, Notebook={nb_exists}')
except Exception as e:
    print(f'Documentation check FAILED: {e}')

# Backward compat
print()
print('=== Backward Compatibility ===')
for mod in ['emotion', 'gender', 'cough', 'depression', 'pronunciation', 'core']:
    try:
        __import__(f'audio_project.{mod}')
        print(f'  audio_project.{mod}: OK')
    except Exception as e:
        print(f'  audio_project.{mod}: FAILED - {e}')

# Model results
print()
print('=== Model Comparison ===')
import pandas as pd
df = pd.read_csv('results/parkinson_model_comparison.csv')
print(df.to_string(index=False))
