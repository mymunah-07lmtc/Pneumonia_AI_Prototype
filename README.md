# 🫁 AI-Powered Lung Sound Screener



Research prototype for affordable respiratory disease screening in low-resource settings.

*Made my Maimouna Tougoutcho Coulibaly, Mali*



## Features

- Upload or record lung sounds (.wav)

- Real-time AI classification (Normal vs Abnormal)

- 36-feature extraction (MFCCs + spectral features)

- PDF report generation with patient info



## Tech Stack

- Python, Streamlit, librosa, scikit-learn

- Random Forest / SVM / XGBoost (cross-validated)



## Installation

```bash

pip install -r requirements.txt

streamlit run app.py

Dataset

ICBHI 2017 Respiratory Sound Database

```

Download the pre-trained model from https://drive.google.com/drive/folders/1FGATU8k__iJ0j6kcLZGi2JN04eYcyMMb?usp=drive_link and place it in the root folder.

## Disclaimer

For research purposes only. Not a certified medical device.
