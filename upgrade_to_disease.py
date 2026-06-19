import os
import pandas as pd
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. LOCATE THE METADATA FILE
audio_dir = "data/extracted_audio"
metadata_file = None

# Look for common metadata filenames
possible_names = ['ICBHI_metadata.txt', 'ICBHI_meta.txt', 'patient_diagnosis.csv', 'diagnosis.txt', 'metadata.txt']
for root, dirs, files in os.walk(audio_dir):
    for f in files:
        if any(name.lower() in f.lower() for name in possible_names):
            metadata_file = os.path.join(root, f)
            break
    if metadata_file:
        break

if metadata_file is None:
    print("❌ Metadata file not found. Trying to load from CSV...")
    # If you already have the CSV, we'll just use it
    if os.path.exists("lung_sound_features.csv"):
        df = pd.read_csv("lung_sound_features.csv")
        print("⚠️ Using existing features CSV. Labels are chest locations, not disease.")
        print("Please find the ICBHI metadata file manually and reply to me.")
        exit()
else:
    print(f"✅ Found metadata: {metadata_file}")
    
    # 2. LOAD METADATA (Try different formats)
    try:
        if metadata_file.endswith('.csv'):
            meta = pd.read_csv(metadata_file)
        else:
            # Try tab-separated or space-separated
            meta = pd.read_csv(metadata_file, sep='\t', encoding='utf-8')
            if len(meta.columns) < 2:
                meta = pd.read_csv(metadata_file, sep=' ', encoding='utf-8')
        print("📊 Metadata preview:")
        print(meta.head())
        print(f"\n📋 Columns: {list(meta.columns)}")
        
        # 3. EXTRACT PATIENT IDs FROM YOUR FEATURES
        df = pd.read_csv("lung_sound_features.csv")
        df['Patient_ID'] = df['Filename'].str[:3]  # First 3 digits
        
        # 4. MERGE WITH METADATA (MAP DIAGNOSIS)
        # This is the tricky part—column names vary.
        # We'll try to find a column with 'diagnosis' or 'condition' or 'label'
        possible_label_cols = ['diagnosis', 'condition', 'label', 'disease', 'class', 'patient_diagnosis']
        label_col = None
        for col in possible_label_cols:
            for meta_col in meta.columns:
                if col.lower() in meta_col.lower():
                    label_col = meta_col
                    break
            if label_col:
                break
        
        if label_col is None:
            print("⚠️ Could not find diagnosis column. Here are all columns:")
            print(meta.columns.tolist())
            print("Please tell me which column has the disease diagnosis.")
            exit()
        
        print(f"✅ Using column: '{label_col}' for diagnosis")
        
        # Merge
        # Meta usually has a patient ID column
        patient_col = None
        for col in meta.columns:
            if 'patient' in col.lower() or 'id' in col.lower() or 'participant' in col.lower():
                patient_col = col
                break
        
        if patient_col is None:
            # Assume first column is patient ID
            patient_col = meta.columns[0]
            print(f"⚠️ Using '{patient_col}' as patient ID column")
        
        # Convert patient IDs to string and strip
        meta[patient_col] = meta[patient_col].astype(str).str.strip()
        df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        
        # Merge
        df_merged = df.merge(meta[[patient_col, label_col]], left_on='Patient_ID', right_on=patient_col, how='left')
        
        # Check if any missing
        missing = df_merged[label_col].isnull().sum()
        if missing > 0:
            print(f"⚠️ {missing} samples have no diagnosis. Dropping them.")
            df_merged = df_merged.dropna(subset=[label_col])
        
        # 5. BINARY CLASSIFICATION: NORMAL vs ABNORMAL
        # Common normal labels: 'normal', 'healthy', 'norm', 'healthy_control'
        def map_to_binary(diagnosis):
            diagnosis = str(diagnosis).strip().lower()
            if diagnosis in ['normal', 'healthy', 'norm', 'healthy_control', 'normal_control']:
                return 0  # Normal
            else:
                return 1  # Abnormal (pneumonia, COPD, URTI, asthma, etc.)
        
        df_merged['Binary_Label'] = df_merged[label_col].apply(map_to_binary)
        
        # 6. TRAIN THE NEW MODEL
        X = df_merged[[f'MFCC_{i+1}' for i in range(13)]]
        y = df_merged['Binary_Label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"\n🎯 NEW MODEL ACCURACY (Normal vs Abnormal): {accuracy*100:.2f}%")
        
        # Save the new model and a simple label encoder for binary
        joblib.dump(model, "lung_model.pkl")
        # We need a simple label mapping for the app
        label_map = {0: "Normal (Healthy)", 1: "Abnormal (Seek Care)"}
        joblib.dump(label_map, "label_map.pkl")
        
        print("✅ Model updated to predict 'Normal' vs 'Abnormal'")
        print(f"📊 Sample distribution: Normal={sum(y==0)}, Abnormal={sum(y==1)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please paste the error here.")