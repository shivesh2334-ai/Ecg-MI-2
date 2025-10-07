import streamlit as st
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import joblib
import tensorflow as tf
import cv2
import pdf2image
from PIL import Image
import io

# Add new function to handle image preprocessing
def process_ecg_image(file):
    # Read the uploaded file
    if file.type == "application/pdf":
        # Convert PDF to image
        images = pdf2image.convert_from_bytes(file.read())
        image = np.array(images[0])  # Take first page
    else:
        # Read image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to separate ECG signal
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours to extract the ECG signal line
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by length to find the ECG signal line
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    
    # Extract y-coordinates of the ECG signal line
    if len(contours) > 0:
        signal_contour = contours[0]
        signal = signal_contour[:, 0, 1]  # Extract y-coordinates
        # Normalize signal
        signal = (signal - np.mean(signal)) / np.std(signal)
        return signal
    else:
        raise ValueError("No ECG signal line detected in the image")

# Modify the main file upload section
st.sidebar.header("Model Selection and Settings")
ml_model_file = st.sidebar.file_uploader("Upload ML model (.joblib)", type=["joblib"])
dl_model_file = st.sidebar.file_uploader("Upload DL model (.h5)", type=["h5"])
scaler_file = st.sidebar.file_uploader("Upload Scaler (.joblib)", type=["joblib"])

# Add support for multiple file types
input_type = st.radio("Select input type:", ["Signal File", "Image File"])

if input_type == "Signal File":
    ecg_file = st.file_uploader("Upload ECG file (.csv or WFDB)", type=["csv", "dat"])
else:
    ecg_file = st.file_uploader("Upload ECG image", type=["pdf", "png", "jpg", "jpeg", "ficom"])

fs = st.sidebar.number_input("Sampling Rate (Hz)", value=1000)

if st.button("Run MI Detection"):
    if not ecg_file or not ml_model_file or not dl_model_file or not scaler_file:
        st.error("Please upload all required files (ECG, ML model, DL model, Scaler).")
    else:
        try:
            # Load models and scaler
            ml_model = load_ml_model(ml_model_file)
            dl_model = load_dl_model(dl_model_file)
            scaler = load_ml_model(scaler_file)
            feature_extractor = ClinicalECGFeatures(fs=int(fs))

            # Load ECG based on input type
            if input_type == "Signal File":
                if ecg_file.name.endswith('.csv'):
                    df = pd.read_csv(ecg_file)
                    ecg_signal = df.iloc[:,0].values
                else:
                    record = wfdb.rdrecord(ecg_file.name.replace('.dat',''), pn_dir=None, channels=[0])
                    ecg_signal = record.p_signal.flatten()
            else:
                # Process image file
                ecg_signal = process_ecg_image(ecg_file)

            # Display uploaded image if it's an image file
            if input_type == "Image File":
                st.subheader("Uploaded ECG Image")
                if ecg_file.type == "application/pdf":
                    images = pdf2image.convert_from_bytes(ecg_file.read())
                    st.image(images[0], caption="Uploaded ECG (PDF)", use_column_width=True)
                else:
                    st.image(ecg_file, caption="Uploaded ECG", use_column_width=True)

            # Continue with existing analysis pipeline
            features = {}
            features.update(feature_extractor.extract_st_segment_features(ecg_signal))
            features.update(feature_extractor.extract_qrs_features(ecg_signal))
            features.update(feature_extractor.extract_t_wave_features(ecg_signal))
            features.update(feature_extractor.extract_heart_rate_features(ecg_signal))
            features['signal_mean'] = np.mean(ecg_signal)
            features['signal_std'] = np.std(ecg_signal)
            features['signal_skewness'] = skew(ecg_signal)
            features['signal_kurtosis'] = kurtosis(ecg_signal)
            
            # Rest of your existing analysis code...
            
        except Exception as e:
            st.error(f"Error processing ECG: {str(e)}")
