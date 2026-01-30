# src/live_infer.py
import os
import queue
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
from joblib import load

from src.features import featurize_last_window
from src.postprocess import EventPostProcessor

SR = int(os.getenv("COUGH_SR", "16000"))
WINDOW_SEC = float(os.getenv("COUGH_WINDOW_SEC", "1.0"))
HOP_SEC = float(os.getenv("COUGH_HOP_SEC", "0.01"))

# Baseline scoring options when no model is available.
# If COUGH_SCORE_MODE="fixed": p = min(1, rms * SCALE)
# If COUGH_SCORE_MODE="adaptive": p = clip((rms / noise_floor - 1) / K, 0, 1)
SCORE_MODE = os.getenv("COUGH_SCORE_MODE", "adaptive").strip().lower()
SCALE = float(os.getenv("COUGH_SCALE", "30.0"))  # used only in fixed mode
ADAPT_K = float(os.getenv("COUGH_ADAPT_K", "6.0"))  # larger = less sensitive
NOISE_ALPHA = float(os.getenv("COUGH_NOISE_ALPHA", "0.995"))  # closer to 1 = slower updates

DEBUG = os.getenv("COUGH_DEBUG", "0") == "1"

audio_q: queue.Queue[np.ndarray] = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())  # mono


def _pick_input_device():
    # Optional: force a specific input device by index.
    # Example: COUGH_INPUT_DEVICE=1 python -m src.live_infer
    dev = os.getenv("COUGH_INPUT_DEVICE")
    if dev is None or dev.strip() == "":
        return
    try:
        idx = int(dev)
    except ValueError:
        return
    sd.default.device = (idx, None)


def main():
    _pick_input_device()

    # Optional trained classifier
    clf = None
    model_path = os.getenv("COUGH_MODEL_PATH", "models/cough_clf.joblib")
    if os.path.exists(model_path):
        clf = load(model_path)
        print("Loaded model from", model_path)
    else:
        print("No trained model found. Using baseline scores (mode =", SCORE_MODE + ")")

    post = EventPostProcessor(
        start_thresh=float(os.getenv("COUGH_START_THRESH", "0.7")),
        end_thresh=float(os.getenv("COUGH_END_THRESH", "0.4")),
        min_event_sec=float(os.getenv("COUGH_MIN_EVENT_SEC", "0.10")),
        merge_gap_sec=float(os.getenv("COUGH_MERGE_GAP_SEC", "0.30")),
        refractory_sec=float(os.getenv("COUGH_REFRACTORY_SEC", "0.75")),
        smooth_len=int(os.getenv("COUGH_SMOOTH_LEN", "5")),
        fire_on=os.getenv("COUGH_FIRE_ON", "start"),
    )

    ring = np.zeros(int(SR * WINDOW_SEC), dtype=np.float32)
    last_tick = time.time()

    # Adaptive noise floor for baseline scoring
    noise_floor = 1e-4

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
            if DEBUG:
                try:
                    print("Input device:", sd.query_devices(sd.default.device[0]))
                except Exception:
                    pass

            while True:
                chunk = audio_q.get()
                n = len(chunk)

                if n >= len(ring):
                    ring[:] = chunk[-len(ring):]
                else:
                    ring = np.roll(ring, -n)
                    ring[-n:] = chunk

                now = time.time()
                if now - last_tick < HOP_SEC:
                    continue

                x = featurize_last_window(ring, sr=SR)  # [rms, peak, peak_to_rms]
                rms = float(x[0])

                if clf is not None:
                    p = float(clf.predict_proba([x])[0, 1])
                else:
                    if SCORE_MODE == "fixed":
                        p = min(1.0, rms * SCALE)
                    else:
                        # ratio = rms / (noise_floor + 1e-12)
                        # # adaptive: compare to running noise floor
                        # noise_floor = NOISE_ALPHA * noise_floor + (1.0 - NOISE_ALPHA) * rms
                        # # map ratio to [0,1], ratio ~ 1 => 0, ratio ~ (1+K) => ~1
                        # p = (ratio - 1.0) / ADAPT_K
                        # p = float(np.clip(p, 0.0, 1.0))

                        
                        ratio = rms / (noise_floor + 1e-12)
                        # Update noise floor only when we are likely in "background" audio.
                        # Two gates:
                        #  1) ratio not too high
                        #  2) peak_to_rms not too "impulsive"
                        peak = float(x[1])
                        p2r = float(x[2])
                        quiet_enough = ratio < float(os.getenv("COUGH_NOISE_RATIO_MAX", "2.0"))
                        not_impulsive = p2r < float(os.getenv("COUGH_NOISE_P2R_MAX", "6.0"))

                        if quiet_enough and not_impulsive:
                            noise_floor = NOISE_ALPHA * noise_floor + (1.0 - NOISE_ALPHA) * rms

                        # Recompute ratio after possible update
                        ratio = rms / (noise_floor + 1e-12)
                        # p = (ratio - 1.0) / ADAPT_K
                        # p = float(np.clip(p, 0.0, 1.0))

                        # Combine ratio and spike terms for better sensitivity
                        ratio_term = (ratio - 1.0) / ADAPT_K
                        spike_term = (p2r - 3.0) / 6.0   # rough scale
                        p = float(np.clip(0.7 * ratio_term + 0.3 * spike_term, 0.0, 1.0))



                fired = post.update(p, t=now)
                if DEBUG:
                    print(f"rms={rms:.6f} p={p:.3f} noise={noise_floor:.6f}")

                if fired:
                    ts = datetime.now().isoformat()
                    print(ts)

                last_tick = now

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print("Error:", repr(e))
        raise


if __name__ == "__main__":
    main()
