# src/model.py
import torch
import torch.nn as nn

class AudioClassifierCNN(nn.Module):
    """
    Multi-class 2D CNN for Log-Mel Spectrograms.
    Can classify any number of audio classes (e.g., cough, dog, sneezing, etc.).
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces 128x100 -> 64x50
            
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces 64x50 -> 32x25

            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduces 32x25 -> 16x12
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits  # Return raw logits for multi-class (use softmax/cross-entropy)
    