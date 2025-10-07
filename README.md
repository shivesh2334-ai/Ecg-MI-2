# ECG MI Detection Streamlit App

This app analyzes ECG signals for Myocardial Infarction (MI) detection using both classical ML and deep learning models, with clinical interpretation.

## Features

- Upload ECG files (`.csv`, WFDB)
- Predict MI type: Normal, Inferior MI, Anterior MI, Other
- Clinical feature extraction: ST segment, QRS, T wave, HR
- ML and DL probability outputs
- Clinical interpretation

## Quick Start

### 1. Clone the repo

```bash
git clone <repo_url>
cd <repo_folder>
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Models

- **ML Model:** Train and export a RandomForest or similar classifier as `model.joblib`.
- **Scaler:** Export fitted scaler as `scaler.joblib`.
- **DL Model:** Train and save a Keras model as `model.h5`.

> You can use the code in your pipeline to train and export these.

### 4. Run Streamlit

```bash
streamlit run app.py
```

### 5. Usage

- Upload your ML model (`.joblib`), DL model (`.h5`), and scaler (`.joblib`) in the sidebar.
- Upload your ECG file (CSV or WFDB format).
- Click **Run MI Detection**.
- View results, feature table, ECG plot, and clinical interpretation.

## ECG File Format

- **CSV:** First column should be the ECG signal (single-lead).
- **WFDB:** Standard PhysioNet format (`.dat`, `.hea`).

## Model Training

See the main pipeline for training instructions. Save models as follows:

```python
import joblib
joblib.dump(rf_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
keras_model.save('model.h5')
```

## Citation

Based on PTB Diagnostic ECG Database (PhysioNet) and clinical MI guidelines.
