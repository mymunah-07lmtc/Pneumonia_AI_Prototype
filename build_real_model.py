import os
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD METADATA
if os.path.exists("icbhi_diagnosis.csv"):
    metadata = pd.read_csv("icbhi_diagnosis.csv")
    print("✅ Loaded metadata from icbhi_diagnosis.csv")
else:
    print("❌ Metadata file not found. Download it first.")
    exit()

print("\n📊 Metadata preview:")
print(metadata.head())
print(f"\n📋 Columns: {list(metadata.columns)}")

# Determine which column is Patient ID and which is Diagnosis
patient_col = None
diagnosis_col = None

for col in metadata.columns:
    if 'patient' in col.lower() or 'id' in col.lower():
        patient_col = col
    if 'diagnosis' in col.lower() or 'disease' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
        diagnosis_col = col

if patient_col is None:
    patient_col = metadata.columns[0]  # Assume first column is patient ID
if diagnosis_col is None:
    diagnosis_col = metadata.columns[1]  # Assume second column is diagnosis

print(f"🔍 Using: Patient ID = '{patient_col}', Diagnosis = '{diagnosis_col}'")

# 2. LOAD YOUR AUDIO FEATURES
df_features = pd.read_csv("lung_sound_features.csv")
df_features['Patient_ID'] = df_features['Filename'].str[:3].astype(int)

# 3. MERGE WITH METADATA
metadata[patient_col] = metadata[patient_col].astype(int)
df_merged = df_features.merge(metadata[[patient_col, diagnosis_col]], left_on='Patient_ID', right_on=patient_col, how='left')

# Drop rows with no diagnosis
df_clean = df_merged.dropna(subset=[diagnosis_col])
print(f"\n✅ Merged: {len(df_clean)} samples with diagnoses")

# 4. CREATE BINARY LABEL (0 = Healthy/Normal, 1 = Any disease)
def map_to_binary(diagnosis):
    diagnosis = str(diagnosis).strip().lower()
    # Normal/Healthy cases
    if diagnosis in ['normal', 'healthy', 'control', 'normal_control', 'healthy_control']:
        return 0
    else:
        return 1

df_clean['Binary_Label'] = df_clean[diagnosis_col].apply(map_to_binary)

print(f"\n📊 Class Distribution:")
print(f"   Normal (0): {sum(df_clean['Binary_Label'] == 0)}")
print(f"   Abnormal (1): {sum(df_clean['Binary_Label'] == 1)}")

# 5. EXTRACT ADVANCED FEATURES
audio_dir = "data/extracted_audio"
print("\n🔬 Extracting 36 features from audio files...")

X_list = []
y_list = []

for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
    fname = row['Filename']
    
    # Find the file path
    file_path = None
    for root, dirs, files in os.walk(audio_dir):
        if fname in files:
            file_path = os.path.join(root, fname)
            break
    
    if file_path is None:
        continue
    
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        
        # MFCCs (13 coefficients) - Mean & Std (26 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Spectral features (Mean & Std) (10 features)
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
        features.append(np.std(rms))    # 10 (Total 36)
        
        X_list.append(features)
        y_list.append(int(row['Binary_Label']))
    except Exception as e:
        continue

X = np.array(X_list)
y = np.array(y_list)

print(f"\n✅ Final dataset: {len(X)} samples")
print(f"   Normal: {np.sum(y==0)}")
print(f"   Abnormal: {np.sum(y==1)}")

if len(X) == 0:
    print("❌ No samples extracted. Check if audio files exist.")
    exit()

# 6. CROSS-VALIDATION
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
}

try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
except ImportError:
    print("ℹ️ XGBoost not installed. Run: pip install xgboost")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = -1
best_pipeline = None

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
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        best_pipeline = {
            'model': model,
            'scaler': scaler,
            'score': mean_score
        }

print(f"\n🏆 Best Model: {type(best_pipeline['model']).__name__} (CV Accuracy: {best_pipeline['score']:.4f})")

# 7. SAVE
joblib.dump(best_pipeline['model'], "lung_model_best.pkl")
joblib.dump(best_pipeline['scaler'], "scaler.pkl")
label_map = {0: "Normal (Healthy)", 1: "Abnormal (Seek Care)"}
joblib.dump(label_map, "label_map.pkl")

print("\n✅ Model saved successfully!")
print("   - lung_model_best.pkl")
print("   - scaler.pkl")
print("   - label_map.pkl")
print("\n🚀 Run: python -m streamlit run app.py")