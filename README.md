# 🫁 AI-Powered Lung Sound Screener

**Research Prototype | By Maimouna Tougoutcho Coulibaly**

An end-to-end machine learning prototype for rapid triage of respiratory abnormalities (Pneumonia, COPD, URTI) in underserved clinical environments. 

**Video demo of the Streamlit website:** https://github.com/user-attachments/assets/7987f9ed-a934-4b7b-a902-602c27b63676

- **Engineered a 36-feature audio classification pipeline** (MFCCs, spectral centroid, zero-crossing rate, RMS) using Python, librosa, and scikit-learn.
- **Achieved ~80% cross-validated accuracy** on the ICBHI 2017 Respiratory Sound Database (920 audio files) using SVM/XGBoost classifiers.
- **Developed a fully functional Streamlit web application** featuring real-time patient intake, waveform visualization, and automated PDF clinical report generation.
- **Architected a battery-powered offline deployment kit** for the Raspberry Pi, enabling field testing in rural clinics without internet access.

---

## 📦 Project Structure

```
Pneumonia_AI_Prototype/
├── app.py                     # Main Streamlit web application
├── build_real_model.py        # Script to train the model from scratch
├── extract_features.py        # Feature extraction from audio files
├── requirements.txt           # Python dependencies
├── lung_model_best.pkl        # Pre-trained SVM model (Download from Google Drive)
├── scaler.pkl                 # Feature normalizer (Download from Google Drive)
└── label_map.pkl              # Label encoder (Download from Google Drive)
```

---

## 🚀 Quick Start (Run the App in 5 Minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/mymunah-07lmtc/Pneumonia_AI_Prototype.git
cd Pneumonia_AI_Prototype
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Pre-Trained Model Files
The model files are too large for GitHub. **Download them from Google Drive (https://drive.google.com/drive/folders/1FGATU8k__iJ0j6kcLZGi2JN04eYcyMMb?usp=sharing)** and place all three files (`lung_model_best.pkl`, `scaler.pkl`, `label_map.pkl`) in the root folder of this project.

### 5. Run the App
```bash
python -m streamlit run app.py
```
Your browser will automatically open to `http://localhost:8501`.

---

## 📱 Test on Your Phone (Local Network)

1. Find your computer's local IP address:
   - **Windows:** `ipconfig` (look for `IPv4 Address`)
   - **Mac/Linux:** `ifconfig` or `ip addr`
2. Run the app with:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
3. On your phone (connected to the same WiFi), open your browser and type:
   ```
   http://[YOUR_IP_ADDRESS]:8501
   ```
   *(Example: `http://192.168.1.45:8501`)*

---

## 🛠️ Hardware Deployment (Optional)

This project is designed to run **offline** on a **Raspberry Pi** for field deployment.

**Hardware Stack:**
- Raspberry Pi 4 (2GB+)
- USB Sound Card + Lapel Microphone
- Power Bank (5V, 3A)
- MicroSD Card (16GB+)

**Quick Setup on Pi:**
```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg -y
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## 📊 Features

- **Patient Information Form** – Name, Age, Sex, Symptoms
- **Dual Audio Input** – Upload a `.wav` file OR record directly from your browser
- **Real-Time AI Prediction** – "Normal (Healthy)" vs. "Abnormal (Seek Care)"
- **Confidence Scoring** – Percentage confidence for each prediction
- **Signal Visualization** – Waveform plot and MFCC spectrogram
- **PDF Report Generation** – Download a clinical-style summary with patient data, results, and plots

---

## 📄 Disclaimer

⚠️ **For Research Purposes Only.** This tool is a proof-of-concept prototype. It is **NOT** a certified medical device. Do not make clinical decisions based solely on this output. All predictions must be verified by a qualified healthcare professional.

---

## 🧠 Model Performance

- **Dataset:** ICBHI 2017 Respiratory Sound Database (920 audio files)
- **Features:** 36 (MFCC means & standard deviations, spectral centroid, bandwidth, rolloff, zero-crossing rate, RMS)
- **Algorithm:** SVM (Support Vector Machine) with RBF kernel
- **Validation:** 5-Fold Stratified Cross-Validation
- **Accuracy:** ~80%

---

## 📬 Contact & Collaboration

**Author:** Maimouna Tougoutcho Coulibaly  
**Email:** maimounatcoul@gmail.com  
**GitHub:** [github.com/mymunah-07lmtc](https://github.com/mymunah-07lmtc)  
**LinkedIn:** [linkedin.com/in/maimouna-tougoutcho-coulibaly](https://linkedin.com/in/maimouna-tougoutcho-coulibaly)

---

**Built with ❤️ in Bamako, Mali | ICBHI 2017 Database**


---

