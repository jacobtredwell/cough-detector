#!/bin/bash
# run.sh

# 1. Check if model exists, if not, warn user (or train optionally)
if [ ! -f "models/cough_cnn.pth" ]; then
    echo "WARNING: Pre-trained model not found at models/cough_cnn.pth."
    echo "Please run 'python -m src.train' first or place a model file."
fi

# 2. Run the live inference
# Note: For Docker usage, this requires --device flags to access the mic.
echo "Starting Live Cough Detector..."
python -m src.live_infer