# src/train.py
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(__file__))

# Import our modular components
from preprocessing import AudioPreprocessor
from features import LogMelFeatureExtractor
from model import AudioClassifierCNN

# Configuration based on PDF Section 2 & 3
SR = 16000
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

class AudioDataset(Dataset):
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
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train():
    # 1. Prepare Data - Scan all subfolders in data/ as classes
    data_dir = 'data'
    
    # Get all subfolders (classes) avoiding hidden files
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    classes.sort()  # Ensure consistent order
    
    if not classes:
        print("ERROR: No class folders found in data/")
        return
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Collect all files and labels
    all_files = []
    all_labels = []
    
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        files = glob.glob(os.path.join(cls_folder, "*.wav"))
        all_files.extend(files)
        all_labels.extend([class_to_idx[cls_name]] * len(files))
    
    if not all_files:
        print("ERROR: No .wav files found in data/ subfolders")
        return
    
    print(f"Total audio files: {len(all_files)}")
    
    X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    train_ds = AudioDataset(X_train, y_train)
    val_ds = AudioDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    num_classes = len(classes)
    model = AudioClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Multi-class loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
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
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {100 * correct / total:.2f}%")

    # 4. Save Artifacts
    os.makedirs("models", exist_ok=True)
    save_path = "models/audio_classifier_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save class mappings for inference
    import json
    with open("models/class_mappings.json", "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)
    print("Class mappings saved to models/class_mappings.json")

if __name__ == "__main__":
    train()
    