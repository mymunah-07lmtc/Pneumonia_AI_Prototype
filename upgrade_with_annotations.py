import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your features
df = pd.read_csv("lung_sound_features.csv")

# Get patient ID from filename
df['Patient_ID'] = df['Filename'].str[:3]

# Load annotation files (they should be in the same folder as the audio)
audio_dir = "data/extracted_audio"
annotation_files = [f for f in os.listdir(audio_dir) if f.endswith('.txt')]

# Map each audio file to its annotation
def get_annotation(filename):
    base = filename.replace('.wav', '')
    for ann_file in annotation_files:
        if ann_file.startswith(base):
            return ann_file
    return None

df['Annotation_File'] = df['Filename'].apply(get_annotation)

# Parse annotations to get crackles/wheezes
def parse_annotation(ann_file):
    if pd.isna(ann_file) or not ann_file:
        return None, None
    try:
        with open(os.path.join(audio_dir, ann_file), 'r') as f:
            lines = f.readlines()
        has_crackle = False
        has_wheeze = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                if int(parts[2]) == 1:
                    has_crackle = True
                if int(parts[3]) == 1:
                    has_wheeze = True
        return has_crackle, has_wheeze
    except:
        return None, None

df['Has_Crackle'], df['Has_Wheeze'] = zip(*df['Annotation_File'].apply(parse_annotation))

# Create binary label: 0 = Normal (no crackles, no wheezes), 1 = Abnormal
df['Binary_Label'] = df.apply(
    lambda row: 0 if (row['Has_Crackle'] == False and row['Has_Wheeze'] == False) 
                else (1 if (row['Has_Crackle'] == True or row['Has_Wheeze'] == True) else None),
    axis=1
)

# Drop rows where we couldn't determine the label
df_clean = df.dropna(subset=['Binary_Label'])
print(f"✅ Found {len(df_clean)} samples with labels")
print(f"   Normal: {sum(df_clean['Binary_Label'] == 0)}")
print(f"   Abnormal: {sum(df_clean['Binary_Label'] == 1)}")

# Train model
X = df_clean[[f'MFCC_{i+1}' for i in range(13)]]
y = df_clean['Binary_Label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Model Accuracy (Normal vs Abnormal via crackles/wheezes): {accuracy*100:.2f}%")

# Save
joblib.dump(model, "lung_model.pkl")
label_map = {0: "Normal (No crackles/wheezes)", 1: "Abnormal (Crackles or wheezes detected)"}
joblib.dump(label_map, "label_map.pkl")

print("✅ Model saved! You can now run: python -m streamlit run app.py")