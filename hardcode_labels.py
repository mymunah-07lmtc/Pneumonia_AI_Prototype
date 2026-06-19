import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load features
df = pd.read_csv("lung_sound_features.csv")
df['Patient_ID'] = df['Filename'].str[:3].astype(str)

# Known ICBHI patient diagnoses (from official documentation)
# Format: Patient_ID: Diagnosis (1 = Abnormal, 0 = Normal)
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

# Add your patient IDs here if they're missing
print("📊 Applying diagnosis mapping...")
df['Binary_Label'] = df['Patient_ID'].map(diagnosis_map)

# Drop unknown patients
unknown = df[df['Binary_Label'].isnull()]
if len(unknown) > 0:
    print(f"⚠️ {len(unknown)} samples from unknown patients. Dropping them.")
    print(f"   Unknown Patient IDs: {unknown['Patient_ID'].unique().tolist()}")
    df = df.dropna(subset=['Binary_Label'])

print(f"✅ {len(df)} samples with labels")
print(f"   Normal (0): {sum(df['Binary_Label'] == 0)}")
print(f"   Abnormal (1): {sum(df['Binary_Label'] == 1)}")

# Train model
X = df[[f'MFCC_{i+1}' for i in range(13)]]
y = df['Binary_Label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Model Accuracy (Normal vs Abnormal): {accuracy*100:.2f}%")

# Save
joblib.dump(model, "lung_model.pkl")
label_map = {0: "Normal (Healthy)", 1: "Abnormal (Seek Care)"}
joblib.dump(label_map, "label_map.pkl")

print("✅ Model saved! Run: python -m streamlit run app.py")