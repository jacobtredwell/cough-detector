# src/train.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

# Import our modular components
from src.preprocessing import AudioPreprocessor
from src.features import LogMelFeatureExtractor
from src.model import CoughDetectorCNN

# Configuration based on PDF Section 2 & 3
SR = 16000
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

class CoughDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = AudioPreprocessor(sr=SR)
        self.featurizer = LogMelFeatureExtractor(sr=SR)
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 1. Load & Preprocess (Bandpass + Norm) [cite: 127-129]
        # Load exactly 1 second for consistency with the CNN input
        y, _ = librosa.load(path, sr=SR, duration=1.0, mono=True)
        
        # Pad if shorter than 1 sec
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))
        # Truncate if longer (take first second)
        elif len(y) > SR:
            y = y[:SR]
            
        y_clean = self.preprocessor.process(y)
        
        # 2. Extract Features (Log-Mel)
        # Returns shape (1, 128, 101)
        features = self.featurizer.compute(y_clean)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train():
    # 1. Prepare Data
    # Assumes data/coughs and data/non_coughs exist in the root
    cough_files = glob.glob("data/coughs/*.wav")
    non_cough_files = glob.glob("data/non_coughs/*.wav")
    
    files = cough_files + non_cough_files
    # Label 1 for Cough, 0 for Non-Cough
    labels = [1]*len(cough_files) + [0]*len(non_cough_files)
    
    if not files:
        print("ERROR: No data found in data/coughs/*.wav or data/non_coughs/*.wav")
        return

    print(f"Found {len(cough_files)} coughs and {len(non_cough_files)} non-coughs.")

    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.2, random_state=42)

    train_ds = CoughDataset(X_train, y_train)
    val_ds = CoughDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = CoughDetectorCNN().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                predicted = (outputs > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")

    # 4. Save Artifacts
    os.makedirs("models", exist_ok=True)
    save_path = "models/cough_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
    