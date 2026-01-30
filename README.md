# cough-detector
Real-time cough event detector for macOS. Streams live microphone audio, runs an ML classifier, and prints an ISO timestamp whenever a cough is detected. Includes training pipeline, preprocessing, and event-level post-processing.

# Cough Detector (real-time mic event detection)

This repo implements a real-time cough detector that listens to your microphone and prints the current timestamp whenever a cough is detected.

It follows a standard cough detection recipe used in speech and bioacoustics style pipelines: preprocessing (bandpass + normalization + activity gating), time-frequency features (log-mel or MFCC), a lightweight classifier, then event-level post-processing (hysteresis, min duration, refractory period). 

## What it does

- Streams live mic audio (16 kHz mono)
- Extracts features on a sliding window
- Predicts cough vs non-cough
- Converts frame predictions into cough events
- Prints an ISO timestamp once per event

## Architecture

**Audio stream**  
Mic (16 kHz mono) 

**Preprocessing**  
- Bandpass filtering around 100 Hz to 6 kHz 
- RMS or peak normalization (avoid aggressive AGC) 
- Activity detection / VAD to skip silence 

**Features and model**
Two supported feature paths:

1) Log-mel spectrogram (recommended when using CNNs) 
2) MFCCs (compact, classical ML friendly; optionally add delta and delta-delta) 

Typical log-mel parameters:
- window 20 to 40 ms
- hop 10 ms
- FFT size 512 or 1024
- mel bins 40 to 128
- log scaling 

**Event post-processing**
Coughs are events, not frames. We apply:
- median filtering or smoothing
- hysteresis thresholds
- minimum duration constraint
- merge detections within 200 to 500 ms
- reject events shorter than ~100 ms
- refractory period between coughs 

## Why these choices

- Coughs are short, impulsive events, so activity detection helps a lot. 
- Log-mel is the most common time-frequency representation for cough detection and pairs well with CNNs. 
- Event-level post-processing dramatically reduces false positives. 
- Dataset quality often matters more than model complexity (mic variability, background speech, label noise, class imbalance). 

## Project structure

