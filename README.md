# Real-Time Cough Detector

A modular, ML-based audio event detector that streams live microphone audio, processes it via a bioacoustics-grade pipeline, and detects cough events in real-time.

## Project Structure

```text
cough-detector/
├── data/                   # (User provided) Place training .wav files here
│   ├── coughs/
│   └── non_coughs/
├── models/                 # Stores trained 'cough_cnn.pth'
├── src/                    # Source code
│   ├── preprocessing.py    # Bandpass (100-6k Hz) & Normalization
│   ├── features.py         # Log-Mel Spectrogram extraction
│   ├── model.py            # CNN architecture
│   ├── postprocess.py      # Event hysteresis & smoothing
│   ├── train.py            # Training script
│   └── live_infer.py       # Main real-time application
├── notebooks/              # Analysis & Visualization
├── requirements.txt
└── run.sh

