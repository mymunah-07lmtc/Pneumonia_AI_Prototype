import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # This shows a progress bar

# Path to your extracted audio files
audio_dir = "data/extracted_audio"

# Get all .wav files
audio_files = []
for root, dirs, filenames in os.walk(audio_dir):
    for f in filenames:
        if f.endswith('.wav'):
            audio_files.append(os.path.join(root, f))

print(f"Found {len(audio_files)} audio files")
print("Extracting features... This will take 2-5 minutes.")

# Store features and labels
features_list = []
labels_list = []
filenames_list = []

# Function to extract MFCCs from ONE audio file
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)  # Only take first 5 seconds
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        return None

# Loop through all files with a progress bar
for file_path in tqdm(audio_files):
    features = extract_mfcc(file_path)
    if features is not None:
        features_list.append(features)
        filenames_list.append(os.path.basename(file_path))
        
        # EXTRACT LABEL FROM FILENAME
        # ICBHI filenames: "101_1b1_Al_sc_Meditron.wav"
        # The part after the second underscore is the condition (Al, Ar, Pl, Pr, Tc)
        parts = os.path.basename(file_path).split('_')
        if len(parts) >= 3:
            condition_code = parts[2]  # This is "Al", "Ar", "Pl", "Pr", or "Tc"
            labels_list.append(condition_code)
        else:
            labels_list.append("Unknown")

# Create DataFrame
df = pd.DataFrame(features_list)
df.columns = [f'MFCC_{i+1}' for i in range(13)]
df['Label'] = labels_list
df['Filename'] = filenames_list

# Save to CSV
df.to_csv("lung_sound_features.csv", index=False)
print(f"\n✅ DONE! Saved {len(df)} samples to 'lung_sound_features.csv'")
print(f"\n📊 Label distribution:")
print(df['Label'].value_counts())
print(f"\n📁 File saved to: {os.getcwd()}\\lung_sound_features.csv")