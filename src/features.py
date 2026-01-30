# src/features.py
import numpy as np
import librosa

class LogMelFeatureExtractor:
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=160):
        """
        Configuration matching "Recommended Baseline Pipeline":
        - sr: 16 kHz [cite: 127]
        - n_mels: 128 bins (Typical: 40-128) [cite: 45]
        - n_fft: 1024 (approx 64ms window) [cite: 43]
        - hop_length: 160 (10ms at 16k sr) [cite: 42]
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def compute(self, audio: np.ndarray) -> np.ndarray:
        # Ensure correct length or pad if necessary
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Log scaling (Log-Mel) is standard for CNNs [cite: 46, 131]
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Return shape: (1, n_mels, time_steps) for PyTorch 2D CNN
        return log_mel[np.newaxis, ...]
    