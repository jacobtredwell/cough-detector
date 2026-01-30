# src/features.py
import numpy as np

def featurize_last_window(audio_window, sr=16000):
    """
    Baseline feature extractor for a 1-second audio window.
    Returns a small, stable feature vector so the live loop works immediately.

    Later can replace / extend with MFCC/log-mel features.
    """
    x = np.asarray(x, dtype=np.float32)

    # remove DC offset
    x = x - float(np.mean(x))

    # simple energy features
    rms = np.sqrt(np.mean(x**2))
    peak = float(np.max(np.abs(x)) + 1e-12)  # avoid zero

    # Ratio is useful for impulsive sounds like coughs
    peak_to_rms = float(peak / (rms + 1e-12 ))  # avoid zero

    # Return as a 1D feature vector
    return np.array([rms, peak, peak_to_rms], dtype=np.float32)
    