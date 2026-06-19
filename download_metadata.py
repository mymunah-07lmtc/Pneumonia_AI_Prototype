import requests
import pandas as pd
import os

# Try multiple known sources for ICBHI metadata
urls = [
    "https://raw.githubusercontent.com/ritunjaym/respiratory-sound-classification-ai/main/data/patient_diagnosis.csv",
    "https://raw.githubusercontent.com/kaen2891/adversarial_fine-tuning_using_generated_respiratory_sound/main/ICBHI_metadata.csv",
]

for url in urls:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Found metadata at: {url}")
            # Try to parse as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            print("📊 Metadata preview:")
            print(df.head())
            print(f"\n📋 Columns: {list(df.columns)}")
            # Save it
            df.to_csv("ICBHI_metadata.csv", index=False)
            print("✅ Saved to ICBHI_metadata.csv")
            break
    except Exception as e:
        print(f"❌ Failed: {url} - {e}")