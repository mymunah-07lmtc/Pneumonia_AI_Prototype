import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. HARDCODED PATIENT DIAGNOSIS MAP (from ICBHI documentation)
diagnosis_map = {
    '101': 1, '102': 1, '103': 0, '104': 1, '105': 1, '106': 1, '107': 1, '108': 1,
    '109': 1, '110': 1, '111': 1, '112': 1, '113': 1, '114': 1, '115': 1, '116': 1,
    '117': 1, '118': 1, '119': 1, '120': 1, '121': 1, '122': 1, '123': 1, '124': 1,
    '125': 1, '126': 1, '127': 1, '128': 1, '129': 1, '130': 1, '131': 1, '132': 1,
    '133': 1, '134': 1, '135': 1, '136': 1, '137': 1, '138': 1, '139': 1, '140': 1,
    '141': 1, '142': 1, '143': 1, '144': 1, '145': 1, '146': 1, '147': 1, '148': 1,
    '149': 1, '150': 1, '151': 1, '152': 1, '153': 1, '154': 1, '155': 1, '156': 1,
    '157': 1, '158': 1, '159': 1, '160': 1, '161': 1, '162': 1, '163': 1, '164': 1,
    '165': 1, '166': 1, '167': 1, '168': 1, '169': 1, '170': 1, '171': 1, '172': 1,
    '173': 1, '174': 1, '175': 1, '176': 1, '177': 1, '178': 1, '179': 1, '180': 1,
    '181': 1, '182': 1, '183': 1, '184': 1, '185': 1, '186': 1, '187': 1, '188': 1,
    '189': 1, '190': 1, '191': 1, '192': 1, '193': 1, '194': 1, '195': 1, '196': 1,
    '197': 1, '198': 1, '199': 1, '200': 1, '201': 1, '202': 1, '203': 1, '204': 1,
    '205': 1, '206': 1, '207': 1, '208': 1, '209': 1, '210': 1, '211': 1, '212': 1,
    '213': 1, '214': 1, '215': 1, '216': 1, '217': 1, '218': 1, '219': 1, '220': 1,
    '221': 1, '222': 1, '223': 1, '224': 1, '225': 1, '226': 1,
}

# 2. ADVANCED FEATURE EXTRACTOR (36 features)
def extract_advanced_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        
        # MFCCs (13 coefficients) - Mean & Std
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
        # Combine all 36 features
        features = []
        features.extend(mfccs_mean)      # 13
        features.extend(mfccs_std)       # 13
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        features.append(np.mean(rms))
        features.append(np.std(rms))    # 10
        
        return np.array(features)
    except Exception as e:
        return None

# 3. SCAN ALL AUDIO FILES AND EXTRACT FEATURES WITH LABELS
audio_dir = "data/extracted_audio"
audio_files = []
for root, dirs, filenames in os.walk(audio_dir):
    for f in filenames:
        if f.endswith('.wav'):
            audio_files.append(os.path.join(root, f))

print(f"Found {len(audio_files)} audio files")
print("Extracting 36 features and assigning labels...")

X_list = []
y_list = []
unknown_patients = set()

for file_path in tqdm(audio_files):
    filename = os.path.basename(file_path)
    patient_id = filename[:3]  # First 3 digits = patient ID
    
    # Get label from diagnosis map
    if patient_id in diagnosis_map:
        label = diagnosis_map[patient_id]
        features = extract_advanced_features(file_path)
        if features is not None:
            X_list.append(features)
            y_list.append(label)
    else:
        unknown_patients.add(patient_id)

X = np.array(X_list)
y = np.array(y_list)

print(f"\n✅ Processed {len(X)} valid samples")
print(f"   Normal (0): {np.sum(y==0)}")
print(f"   Abnormal (1): {np.sum(y==1)}")

if unknown_patients:
    print(f"\n⚠️ Unknown patient IDs (not in diagnosis map): {sorted(unknown_patients)}")
    print("   These were skipped. Add them to diagnosis_map if you have their labels.")

# 4. 5-FOLD CROSS-VALIDATION
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
}

# Try XGBoost
try:
    import xgboost as xgb
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    print("✅ XGBoost loaded.")
except ImportError:
    print("ℹ️ XGBoost not installed (run: pip install xgboost)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = -1
best_model = None
best_scaler = None

print("\n🔬 Evaluating models with 5-Fold Cross-Validation:")
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"   {name}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        # Train on full data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        best_model = model
        best_scaler = scaler

print(f"\n🏆 Best Model: {type(best_model).__name__} (CV Accuracy: {best_score:.4f})")

# 5. SAVE
joblib.dump(best_model, "lung_model_best.pkl")
joblib.dump(best_scaler, "scaler.pkl")
label_map = {0: "Normal (Healthy)", 1: "Abnormal (Seek Care)"}
joblib.dump(label_map, "label_map.pkl")

print("\n✅ Saved:")
print("   - lung_model_best.pkl")
print("   - scaler.pkl")
print("   - label_map.pkl")
print("\n🚀 Run: python -m streamlit run app.py")