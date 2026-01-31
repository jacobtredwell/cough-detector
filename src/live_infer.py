# src/live_infer.py
import os
import queue
import time
import json
import numpy as np
import sounddevice as sd
import torch

from datetime import datetime
from src.preprocessing import AudioPreprocessor
from src.features import LogMelFeatureExtractor
from src.postprocess import EventPostProcessor
from src.model import AudioClassifierCNN

# Constants from "Recommended Baseline Pipeline" [cite: 125]
SR = 16000
WINDOW_SEC = 1.0  # CNN usually trained on ~1 sec chunks
HOP_SEC = 0.1     # Classification interval

# Load settings
DEBUG = os.getenv("AUDIO_DEBUG", "0") == "1"
MODEL_PATH = os.getenv("AUDIO_MODEL_PATH", "models/audio_classifier_cnn.pth")
MAPPINGS_PATH = os.getenv("AUDIO_MAPPINGS_PATH", "models/class_mappings.json")

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

    # 2. Load Model and Class Mappings
    classes = []
    idx_to_class = {}
    if os.path.exists(MAPPINGS_PATH):
        with open(MAPPINGS_PATH, "r") as f:
            mappings = json.load(f)
            classes = mappings["classes"]
            idx_to_class = {int(k): v for k, v in mappings["idx_to_class"].items()}
        num_classes = len(classes)
        model = AudioClassifierCNN(num_classes=num_classes)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.eval()
            print(f"Loaded multi-class CNN from {MODEL_PATH}")
            print(f"Classes: {classes}")
        else:
            print(f"No model found at {MODEL_PATH}. Running in signal-only mode.")
            model = None
    else:
        print(f"No class mappings found at {MAPPINGS_PATH}. Running in signal-only mode.")
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
            prediction = "unknown"
            confidence = 0.0
            if model:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(features).float().unsqueeze(0) # Batch dim
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    score, idx = torch.max(probs, 1)
                    prediction = idx_to_class.get(idx.item(), "unknown")
                    confidence = score.item()
            else:
                # Fallback: Simple energy heuristic if no model trained yet
                energy = float(np.mean(features) + 80) / 40.0 # Crude mapping from dB to prob
                confidence = np.clip(energy, 0, 1)
                prediction = "high_energy" if confidence > 0.5 else "low_energy"

            # Step D: Event Post-Processing [cite: 132]
            fired = post.update(confidence, t=now)
            
            if DEBUG:
                print(f"Prediction: {prediction} ({confidence:.3f})")

            if fired:
                print(f"{datetime.now().isoformat()} - {prediction.upper()} DETECTED (conf: {confidence:.2f})")

            last_tick = now

if __name__ == "__main__":
    main()
    