# MTSAD
Enhancing Autoncoder Models for Multivariate Time Series Anomaly Detection: The Role of Noise and Data Amount

# MTSAD: Multivariate Time Series Anomaly Detection

This repository contains the implementation of the **MTSAD** model, an advanced framework for **unsupervised anomaly detection in multivariate time series data**. The MTSAD model leverages **ConvLSTM networks** to capture spatio-temporal correlations and **Transposed Convolution layers** for effective reconstruction, optimizing anomaly detection for real-world scenarios.

---

## 🚀 Key Features

- **Robust Spatio-Temporal Modeling:** Utilizes ConvLSTM layers for capturing complex dependencies across spatial and temporal dimensions.
- **Noise Injection:** Enhances resilience to noisy and imbalanced datasets by introducing controlled noise during training.
- **Efficient Data Utilization:** Balances reconstruction accuracy and anomaly detection with an optimal use of training data.
- **High Performance:** Achieves state-of-the-art results on SWaT and WADI datasets, with superior F1 scores compared to traditional and deep learning methods.

---

## 📚 Abstract

Traditional methods for multivariate time series anomaly detection face challenges like handling large-scale data, label scarcity, and sensitivity to noise. MTSAD overcomes these limitations by employing:
- **ConvLSTM-based encoder** to extract spatio-temporal features.
- **Transposed Convolutional decoder** for reconstruction.
- Strategic **noise injection and data volume management** to prevent overfitting and optimize anomaly detection.

Our experiments demonstrate MTSAD's high efficacy on the SWaT and WADI datasets, highlighting its potential for industrial applications such as predictive maintenance, cybersecurity, and environmental monitoring.

---

## 📂 Repository Structure

# Requirements:
pandas==2.0.3
numpy==1.24.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.9.0
IPython==8.15.0
statsmodels==0.14.0
torch==2.1.1+cu118
scipy==1.11.1
tqdm==4.65.0

