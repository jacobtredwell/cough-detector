# cough-detector
Real-time cough event detector for macOS. Streams live microphone audio, runs an ML classifier, and prints an ISO timestamp whenever a cough is detected. Includes training pipeline, preprocessing, and event-level post-processing.

# Cough Detector (real-time mic event detection)

This repo implements a real-time cough detector that listens to your microphone and prints the current timestamp whenever a cough is detected.

It follows a standard cough detection recipe used in speech and bioacoustics style pipelines: preprocessing (bandpass + normalization + activity gating), time-frequency features (log-mel or MFCC), a lightweight classifier, then event-level post-processing (hysteresis, min duration, refractory period). :contentReference[oaicite:0]{index=0}

## What it does

- Streams live mic audio (16 kHz mono)
- Extracts features on a sliding window
- Predicts cough vs non-cough
- Converts frame predictions into cough events
- Prints an ISO timestamp once per event

## Architecture

**Audio stream**  
Mic (16 kHz mono) :contentReference[oaicite:1]{index=1}

**Preprocessing**  
- Bandpass filtering around 100 Hz to 6 kHz :contentReference[oaicite:2]{index=2}  
- RMS or peak normalization (avoid aggressive AGC) :contentReference[oaicite:3]{index=3}  
- Activity detection / VAD to skip silence :contentReference[oaicite:4]{index=4}  

**Features and model**
Two supported feature paths:

1) Log-mel spectrogram (recommended when using CNNs) :contentReference[oaicite:5]{index=5}  
2) MFCCs (compact, classical ML friendly; optionally add delta and delta-delta) :contentReference[oaicite:6]{index=6}  

Typical log-mel parameters:
- window 20 to 40 ms
- hop 10 ms
- FFT size 512 or 1024
- mel bins 40 to 128
- log scaling :contentReference[oaicite:7]{index=7}  

**Event post-processing**
Coughs are events, not frames. We apply:
- median filtering or smoothing
- hysteresis thresholds
- minimum duration constraint
- merge detections within 200 to 500 ms
- reject events shorter than ~100 ms
- refractory period between coughs :contentReference[oaicite:8]{index=8}  

## Why these choices

- Coughs are short, impulsive events, so activity detection helps a lot. :contentReference[oaicite:9]{index=9}  
- Log-mel is the most common time-frequency representation for cough detection and pairs well with CNNs. :contentReference[oaicite:10]{index=10}  
- Event-level post-processing dramatically reduces false positives. :contentReference[oaicite:11]{index=11}  
- Dataset quality often matters more than model complexity (mic variability, background speech, label noise, class imbalance). :contentReference[oaicite:12]{index=12}  

## Project structure

