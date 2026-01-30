# src/live_infer.py
import os
import queue
import time
import numpy as np
import sounddevice as sd
import torch

from datetime import datetime
from src.preprocessing import AudioPreprocessor
from src.features import LogMelFeatureExtractor
from src.postprocess import EventPostProcessor
from src.model import CoughDetectorCNN

# Constants from "Recommended Baseline Pipeline" [cite: 125]
SR = 16000
WINDOW_SEC = 1.0  # CNN usually trained on ~1 sec chunks
HOP_SEC = 0.1     # Classification interval

# Load settings
DEBUG = os.getenv("COUGH_DEBUG", "0") == "1"
MODEL_PATH = os.getenv("COUGH_MODEL_PATH", "models/cough_cnn.pth")

audio_q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())

def main():
    # 1. Initialize Pipeline Components [cite: 7]
    preprocessor = AudioPreprocessor(sr=SR)     # Bandpass + Norm [cite: 19, 26]
    featurizer = LogMelFeatureExtractor(sr=SR)  # Log-Mel [cite: 38]
    post = EventPostProcessor()                 # Event Logic [cite: 118]

    # 2. Load Model (CNN) 
    model = CoughDetectorCNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print(f"Loaded CNN from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}. Running in signal-only mode.")
        model = None

    # Ring buffer for audio
    ring_len = int(SR * WINDOW_SEC)
    ring = np.zeros(ring_len, dtype=np.float32)
    last_tick = time.time()

    print("Listening...")
    with sd.InputStream(samplerate=SR, channels=1, callback=callback):
        while True:
            chunk = audio_q.get()
            n = len(chunk)

            # Update ring buffer
            ring = np.roll(ring, -n)
            ring[-n:] = chunk

            now = time.time()
            if now - last_tick < HOP_SEC:
                continue

            # --- PIPELINE EXECUTION ---
            
            # Step A: Preprocessing (Bandpass 100-6k Hz) [cite: 128]
            clean_audio = preprocessor.process(ring)

            # Step B: Feature Extraction (Log-Mel) 
            # Shape: (1, 128, T)
            features = featurizer.compute(clean_audio) 

            # Step C: Inference
            p = 0.0
            if model:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(features).float().unsqueeze(0) # Batch dim
                    p = float(model(input_tensor).item())
            else:
                # Fallback: Simple energy heuristic if no model trained yet
                p = float(np.mean(features) + 80) / 40.0 # Crude mapping from dB to prob
                p = np.clip(p, 0, 1)

            # Step D: Event Post-Processing [cite: 132]
            fired = post.update(p, t=now)
            
            if DEBUG:
                print(f"Score: {p:.3f}")

            if fired:
                print(f"{datetime.now().isoformat()} - COUGH DETECTED")

            last_tick = now

if __name__ == "__main__":
    main()
    