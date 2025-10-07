import streamlit as st
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

# ====== Load Models and Utilities ======
@st.cache_resource
def load_ml_model(path):
    return joblib.load(path)

@st.cache_resource
def load_dl_model(path):
    return tf.keras.models.load_model(path)

class ClinicalECGFeatures:
    def __init__(self, fs=1000):
        self.fs = fs
    def extract_st_segment_features(self, signal):
        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=self.fs)
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            rpeaks_idx = rpeaks['ECG_R_Peaks']
            if len(rpeaks_idx) < 2: return self._get_default_st_features()
            st_elevations, st_deviations = [], []
            for i in range(len(rpeaks_idx) - 1):
                r_peak = rpeaks_idx[i]
                st_start = r_peak + int(0.08 * self.fs)
                st_end = r_peak + int(0.12 * self.fs)
                if st_end < len(signal):
                    st_segment = signal[st_start:st_end]
                    baseline = signal[max(0, r_peak - int(0.04 * self.fs)):r_peak]
                    if len(baseline) > 0:
                        st_elevation = np.mean(st_segment) - np.mean(baseline)
                        st_elevations.append(st_elevation)
                        st_deviations.append(np.std(st_segment))
            return {
                'st_elevation_mean': np.mean(st_elevations) if st_elevations else 0,
                'st_elevation_max': np.max(st_elevations) if st_elevations else 0,
                'st_elevation_std': np.std(st_elevations) if st_elevations else 0,
                'st_deviation_mean': np.mean(st_deviations) if st_deviations else 0,
            }
        except: return self._get_default_st_features()
    def extract_qrs_features(self, signal):
        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=self.fs)
            _, waves = nk.ecg_delineate(cleaned, sampling_rate=self.fs)
            qrs_durations, q_wave_depths = [], []
            if 'ECG_Q_Peaks' in waves and 'ECG_S_Peaks' in waves:
                q_peaks = [p for p in waves['ECG_Q_Peaks'] if not np.isnan(p)]
                s_peaks = [p for p in waves['ECG_S_Peaks'] if not np.isnan(p)]
                for q, s in zip(q_peaks, s_peaks):
                    if not np.isnan(q) and not np.isnan(s):
                        duration = (s - q) / self.fs * 1000
                        qrs_durations.append(duration)
                        q_wave_depths.append(abs(signal[int(q)]))
            return {
                'qrs_duration_mean': np.mean(qrs_durations) if qrs_durations else 0,
                'qrs_duration_max': np.max(qrs_durations) if qrs_durations else 0,
                'q_wave_depth_mean': np.mean(q_wave_depths) if q_wave_depths else 0,
                'bundle_branch_block_indicator': 1 if qrs_durations and np.mean(qrs_durations) > 120 else 0,
            }
        except:
            return {
                'qrs_duration_mean': 0, 'qrs_duration_max': 0, 'q_wave_depth_mean': 0, 'bundle_branch_block_indicator': 0,
            }
    def extract_t_wave_features(self, signal):
        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=self.fs)
            _, waves = nk.ecg_delineate(cleaned, sampling_rate=self.fs)
            t_wave_amplitudes = []
            if 'ECG_T_Peaks' in waves:
                t_peaks = [p for p in waves['ECG_T_Peaks'] if not np.isnan(p)]
                for t in t_peaks:
                    if not np.isnan(t) and int(t) < len(signal):
                        t_wave_amplitudes.append(signal[int(t)])
            return {
                't_wave_amplitude_mean': np.mean(t_wave_amplitudes) if t_wave_amplitudes else 0,
                't_wave_inversion': 1 if t_wave_amplitudes and np.mean(t_wave_amplitudes) < 0 else 0,
            }
        except:
            return {'t_wave_amplitude_mean': 0, 't_wave_inversion': 0}
    def extract_heart_rate_features(self, signal):
        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=self.fs)
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            rate = nk.ecg_rate(rpeaks, sampling_rate=self.fs, desired_length=len(signal))
            rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / self.fs * 1000
            return {
                'heart_rate_mean': np.mean(rate) if len(rate) > 0 else 0,
                'heart_rate_std': np.std(rate) if len(rate) > 0 else 0,
                'rr_interval_mean': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
                'rmssd': np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0,
            }
        except:
            return {'heart_rate_mean': 0, 'heart_rate_std': 0, 'rr_interval_mean': 0, 'rmssd': 0}
    def _get_default_st_features(self):
        return {'st_elevation_mean': 0, 'st_elevation_max': 0, 'st_elevation_std': 0, 'st_deviation_mean': 0}

def preprocess_dl_ecg(ecg_signal):
    # Normalize, reshape, pad/truncate
    max_length = 5000
    signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    if len(signal) > max_length:
        signal = signal[:max_length]
    else:
        signal = np.pad(signal, (0, max_length - len(signal)), mode='edge')
    return signal.reshape(1, max_length, 1)

LABEL_NAMES = {0: "Normal/Healthy", 1: "Inferior MI", 2: "Anterior MI", 3: "Other"}

# ================= STREAMLIT APP UI =================
st.set_page_config(page_title="ECG MI Detection", layout="wide")
st.title("ECG Analysis for Myocardial Infarction Detection")
st.markdown("Upload an ECG file (WFDB, CSV) to detect MI using ML and Deep Learning models with clinical interpretation.")

st.sidebar.header("Model Selection and Settings")
ml_model_file = st.sidebar.file_uploader("Upload ML model (.joblib)", type=["joblib"])
dl_model_file = st.sidebar.file_uploader("Upload DL model (.h5)", type=["h5"])
scaler_file = st.sidebar.file_uploader("Upload Scaler (.joblib)", type=["joblib"])

ecg_file = st.file_uploader("Upload ECG file (.csv or WFDB)", type=["csv", "dat"])
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=1000)

if st.button("Run MI Detection"):
    if not ecg_file or not ml_model_file or not dl_model_file or not scaler_file:
        st.error("Please upload all required files (ECG, ML model, DL model, Scaler).")
    else:
        # ----- Load models and scaler -----
        ml_model = load_ml_model(ml_model_file)
        dl_model = load_dl_model(dl_model_file)
        scaler = load_ml_model(scaler_file)
        feature_extractor = ClinicalECGFeatures(fs=int(fs))

        # ----- Load ECG -----
        if ecg_file.name.endswith('.csv'):
            df = pd.read_csv(ecg_file)
            ecg_signal = df.iloc[:,0].values
        else:
            record = wfdb.rdrecord(ecg_file.name.replace('.dat',''), pn_dir=None, channels=[0])
            ecg_signal = record.p_signal.flatten()

        # ----- Feature Extraction -----
        features = {}
        features.update(feature_extractor.extract_st_segment_features(ecg_signal))
        features.update(feature_extractor.extract_qrs_features(ecg_signal))
        features.update(feature_extractor.extract_t_wave_features(ecg_signal))
        features.update(feature_extractor.extract_heart_rate_features(ecg_signal))
        features['signal_mean'] = np.mean(ecg_signal)
        features['signal_std'] = np.std(ecg_signal)
        features['signal_skewness'] = skew(ecg_signal)
        features['signal_kurtosis'] = kurtosis(ecg_signal)
        X_feat = np.array([list(features.values())])
        X_feat_scaled = scaler.transform(X_feat)

        # ----- ML prediction -----
        ml_pred = ml_model.predict(X_feat_scaled)[0]
        ml_proba = ml_model.predict_proba(X_feat_scaled)[0]

        # ----- DL prediction -----
        dl_input = preprocess_dl_ecg(ecg_signal)
        dl_proba = dl_model.predict(dl_input, verbose=0)[0]
        dl_pred = np.argmax(dl_proba)

        # ----- Display Results -----
        st.subheader("Results")
        st.markdown(f"**ML Diagnosis**: {LABEL_NAMES.get(ml_pred,'Unknown')}")
        st.markdown(f"**DL Diagnosis**: {LABEL_NAMES.get(dl_pred,'Unknown')}")

        st.write("### ML Probabilities")
        st.bar_chart(pd.Series(ml_proba, index=list(LABEL_NAMES.values())))

        st.write("### DL Probabilities")
        st.bar_chart(pd.Series(dl_proba, index=list(LABEL_NAMES.values())))

        st.write("### Clinical Features")
        st.dataframe(pd.DataFrame(features, index=['Value']).T)

        # ----- ECG Plot -----
        st.write("### ECG Signal")
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(np.arange(len(ecg_signal))/fs, ecg_signal, "b-", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        st.pyplot(fig)

        # ----- Clinical Interpretation -----
        st.write("### Clinical Interpretation")
        interpretation = []
        st_elevation = features.get('st_elevation_mean', 0)
        if st_elevation > 0.1:
            interpretation.append("**Significant ST-segment elevation detected (>1mm) → Suggests acute myocardial infarction.**")
        elif st_elevation < -0.1:
            interpretation.append("**ST-segment depression detected → May indicate ischemia or posterior MI.**")
        qrs_duration = features.get('qrs_duration_mean', 0)
        if qrs_duration > 120:
            interpretation.append("**Prolonged QRS duration (>120ms) → Suggests bundle branch block, associated with anterior MI.**")
        if features.get('t_wave_inversion', 0) == 1:
            interpretation.append("**T-wave inversion detected → May indicate reperfusion or normal MI evolution.**")
        hr_mean = features.get('heart_rate_mean', 0)
        if hr_mean > 100:
            interpretation.append("**Tachycardia (>100 bpm) → May indicate instability/pump failure.**")
        elif hr_mean < 60:
            interpretation.append("**Bradycardia (<60 bpm) → Common in inferior MI (vagal tone).**")
        if not interpretation:
            interpretation.append("No critical MI indicators detected in this ECG.")
        st.markdown("\n".join(interpretation))
