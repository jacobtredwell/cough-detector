# src/preprocessing.py
import numpy as np
import scipy.signal

class AudioPreprocessor:
    def __init__(self, sr=16000):
        self.sr = sr
        # Design Bandpass Filter (100 Hz - 6 kHz) [cite: 128]
        # 100 Hz high-pass removes rumble; 6 kHz low-pass removes HF noise [cite: 23, 24]
        nyquist = 0.5 * self.sr
        low = 100.0 / nyquist
        high = 6000.0 / nyquist
        self.b, self.a = scipy.signal.butter(5, [low, high], btype='band')

    def process(self, audio: np.ndarray) -> np.ndarray:
        # 1. Bandpass Filter
        filtered = scipy.signal.lfilter(self.b, self.a, audio)
        
        # 2. RMS Normalization [cite: 129]
        # "Avoid aggressive AGC" - preserving loudness info is useful [cite: 29]
        rms = np.sqrt(np.mean(filtered**2))
        if rms > 1e-6:
            filtered = filtered / rms * 0.1  # Normalize to arbitrary target RMS (e.g. 0.1)
            
        return filtered.astype(np.float32)
    