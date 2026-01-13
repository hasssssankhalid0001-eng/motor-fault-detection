import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

FEATURE_RANGES = {
    "RMS": (0.427, 0.847),
    "Peak": (0.605, 2.993),
    "Dominant Frequency": (47, 53),
    "Peak Magnitude": (0.203, 0.562),
    "Spectral Energy": (0.183, 0.717),
    "Sideband Energy": (0.087, 0.357),
}

def compute_feature_importance(model, scaler, input_data):
    base_pred = model.predict(scaler.transform(input_data))[0]
    base_class = np.argmax(base_pred)
    base_prob = base_pred[base_class]

    importance = {}
    for i, name in enumerate(FEATURE_RANGES.keys()):
        perturbed = input_data.copy()
        perturbed[0, i] *= 1.05
        new_pred = model.predict(scaler.transform(perturbed))[0]
        new_prob = new_pred[base_class]
        importance[name] = 100 * abs(base_prob - new_prob) / (base_prob + 1e-8)

    return importance

def check_feature_ranges(features):
    warnings = []
    for k, v in features.items():
        mn, mx = FEATURE_RANGES[k]
        if v < mn or v > mx:
            warnings.append(f"{k} = {v} (valid range: {mn} – {mx})")
    return warnings

@st.cache_resource
def load_assets():
    model = load_model("motor_fault_model-3.keras")
    scaler = joblib.load("scaler-3.pkl")
    encoder = joblib.load("label_encoder-3.pkl")
    return model, scaler, encoder

st.title("🛠️ Motor Fault Detection System")

st.write(FEATURE_RANGES)

class_map = {
    1: "Healthy",
    2: "Noisy",
    3: "Amplitude Modulation Fault",
    4: "Frequency Shift"
}

rms = st.number_input(
    "RMS Value",
    min_value=0.427,
    max_value=0.847,
    value=0.637,
    step=0.001
)

peak = st.number_input(
    "Peak Value",
    min_value=0.605,
    max_value=2.993,
    value=1.799,
    step=0.01
)

dom_freq = st.number_input(
    "Dominant Frequency (Hz)",
    min_value=47.0,
    max_value=53.0,
    value=50.0,
    step=0.1
)

peak_mag = st.number_input(
    "Peak Magnitude",
    min_value=0.203,
    max_value=0.562,
    value=0.382,
    step=0.001
)

spectral_energy = st.number_input(
    "Spectral Energy",
    min_value=0.183,
    max_value=0.717,
    value=0.450,
    step=0.001
)

sideband_energy = st.number_input(
    "Sideband Energy",
    min_value=0.087,
    max_value=0.357,
    value=0.222,
    step=0.001
)


input_data = np.array([[rms, peak, dom_freq, peak_mag, spectral_energy, sideband_energy]])

if st.button("🔍 Predict"):
    model, scaler, encoder = load_assets()

    features = {
        "RMS": rms,
        "Peak": peak,
        "Dominant Frequency": dom_freq,
        "Peak Magnitude": peak_mag,
        "Spectral Energy": spectral_energy,
        "Sideband Energy": sideband_energy,
    }

    warnings = check_feature_ranges(features)
    if warnings:
        st.warning("⚠ Input values outside trained operating range")
        for w in warnings:
            st.write("•", w)

    prediction = model.predict(scaler.transform(input_data))
    class_idx = np.argmax(prediction[0])
    class_name = class_map[class_idx + 1]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write(
        {class_map[i + 1]: f"{p * 100:.2f}%" for i, p in enumerate(prediction[0])}
    )

    importance = compute_feature_importance(model, scaler, input_data)
    st.subheader("📊 Feature Importance (%)")
    st.bar_chart(importance)

with st.expander("ℹ Model Validity Information"):
    st.write("""
    This model is trained on motor signals operating within specific feature ranges.
    Predictions are most reliable when inputs fall within these ranges.
    Inputs outside the trained distribution may lead to reduced confidence or uncertainty.
    """)
