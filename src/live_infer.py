# src/live_infer.py
import queue, time
from datetime import datetime
import numpy as np
import sounddevice as sd
from joblib import load

from features import featurize_last_window
from postprocess import EventPostProcessor

SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.01

audio_q = queue.Queue()

def callback(indata, frames, time_info, status):
    if status:
        pass
    audio_q.put(indata[:, 0].copy())  # mono

def main():
    clf = load("models/cough_clf.joblib")
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
                x = featurize_last_window(ring, sr=SR)  # shape (d,)
                p = clf.predict_proba([x])[0, 1]
                fired = post.update(p, t=now)
                if fired:
                    print(datetime.now().isoformat(timespec="seconds"))
                last_tick = now

if __name__ == "__main__":
    main()
