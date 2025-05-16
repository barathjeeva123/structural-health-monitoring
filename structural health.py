import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.ensemble import IsolationForest

# 1. Simulate vibration sensor data (normal + abnormal)
np.random.seed(42)
time = np.linspace(0, 10, 1000)
normal_vibration = 0.5 * np.sin(2 * np.pi * 2 * time) + 0.05 * np.random.randn(1000)
anomaly = normal_vibration.copy()
anomaly[700:750] += np.random.normal(1.5, 0.2, 50)  # Inject anomaly

# 2. Feature extraction using FFT (frequency domain analysis)
def extract_fft_features(signal, sampling_rate=100):
    yf = fft(signal)
    magnitude = 2.0 / len(signal) * np.abs(yf[:len(signal)//2])
    return magnitude[:10]  # Return first 10 frequency components

features = []
window_size = 50
for i in range(0, len(anomaly) - window_size, window_size):
    window = anomaly[i:i + window_size]
    fft_feat = extract_fft_features(window)
    features.append(fft_feat)

features = np.array(features)

# 3. Anomaly detection using Isolation Forest
clf = IsolationForest(contamination=0.1)
clf.fit(features)
preds = clf.predict(features)

# 4. Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(anomaly, label="Vibration Signal")
for i, p in enumerate(preds):
    if p == -1:
        start = i * window_size
        plt.axvspan(start, start + window_size, color='red', alpha=0.3)
plt.title("Structural Vibration Monitoring with Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
