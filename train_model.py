import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your features
df = pd.read_csv("lung_sound_features.csv")

# Features (13 MFCCs)
X = df[[f'MFCC_{i+1}' for i in range(13)]]

# Labels (Chest locations - we'll upgrade this later)
y = df['Label']

# Encode labels (turn "Al", "Ar" into numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a simple Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"🎯 Model Accuracy: {accuracy:.2f}")

# Save the model and the label encoder
joblib.dump(model, "lung_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model saved as 'lung_model.pkl'")
print(f"📊 Labels: {list(le.classes_)}")