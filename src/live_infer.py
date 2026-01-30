# src/live_infer.py
import os

import queue, time
from datetime import datetime
import numpy as np
import sounddevice as sd
from joblib import load

from src.features import featurize_last_window
from src.postprocess import EventPostProcessor

SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.01
audio_q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        pass
    audio_q.put(indata[:, 0].copy())  # mono

def main():
    # clf = load("models/cough_clf.joblib")

    clf = None
    model_path = "models/cough_clf.joblib"
    if os.path.exists(model_path):
        clf = load(model_path)
        print("Loaded model from", model_path)
    else:
        print("No trained model found. Using energy baseline scores")

    post = EventPostProcessor()
    ring = np.zeros(int(SR * WINDOW_SEC), dtype=np.float32)
    hop_n = int(SR * HOP_SEC)
    last_tick = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        while True:
            chunk = audio_q.get()
            # append to ring buffer
            n = len(chunk)
            ring = np.roll(ring, -n)
            ring[-n:] = chunk
            now = time.time()
            if now - last_tick >= HOP_SEC:
                # x = featurize_last_window(ring, sr=SR)  # shape (d,)
                # p = clf.predict_proba([x])[0, 1]

                ## Gives working event pipeline without necessarily a trained model
                x = featurize_last_window(ring, sr=SR)  # shape (d,)
                if clf is not None:
                    p = float(clf.predict_proba([x])[0, 1])
                else:
                    # baseline: map RMS energy to pseduo-probability
                    # Tune the scale base on your mic environment
                    rms = float(x[0])
                    p = min(1.0, rms * 30.0)
                ## Gives working event pipeline without necessarily a trained model
                
                fired = post.update(p, t=now)
                if fired:
                    print(datetime.now().isoformat(timespec="seconds"))
                last_tick = now

if __name__ == "__main__":
    main()
