# Use slim python image
FROM python:3.9-slim

# Install system dependencies for audio (libsndfile is needed for librosa)
# portaudio19-dev is needed for sounddevice
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    libasound-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["./run.sh"]
