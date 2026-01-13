# Motor Fault Detection using Artificial Neural Network

## Project Overview

This project detects faults in a **three-phase induction motor** using **machine learning**, specifically an **Artificial Neural Network (ANN)**. The system analyzes the **stator current**, extracts meaningful features, and classifies the motor condition (healthy or faulty).  

A **Streamlit-based GUI** allows users to interact with the model easily and view predictions in real time.

---

## Methodology

1. **Data Acquisition**  
   - Stator current signals were collected from a three-phase induction motor under various operational conditions.  

2. **Signal Processing**  
   - Signals were analyzed in the **time domain** and **frequency domain** using **Fourier Transform**.  
   - Feature extraction was performed to capture key characteristics of the motor signals.

3. **Feature Extraction**  
   The following features were used to train the ANN model:  
   - **RMS (Root Mean Square)**: Measures signal power.  
   - **Peak value**: Maximum amplitude of the signal.  
   - **Dominant frequency**: The frequency with the highest magnitude.  
   - **Peak magnitude**: Amplitude at the dominant frequency.  
   - **Spectral energy**: Total energy in the frequency spectrum.  
   - **Sideband energy**: Energy of sideband frequencies around the fundamental.  
   - **Labels**: Motor conditions (e.g., Healthy, NOISY, AMPLITUDE MODULATION and FRREQUENCY SHIFT )

4. **Artificial Neural Network (ANN)**  
   - The extracted features were used to train an ANN model.  
   - The ANN predicts the motor condition based on these features.  

5. **Deployment with Streamlit**  
   - The trained model is deployed via **Streamlit**, providing an intuitive web interface.  
   - Users can input motor data and get immediate predictions of motor health.

LINK-- https://motor-fault-detection-sifrh47ct8w3q9f4iz5frj.streamlit.app
