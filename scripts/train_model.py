"""
STEP 5 - Model Training Script
--------------------------------
Loads the processed dataset, trains a Conv1D + LSTM neural network,
and saves the best model.

Architecture:
  Input:      (batch, 60 frames, 3 features)
  Conv1D      detects local patterns (rising / falling signal)
  MaxPool1D   downsamples
  LSTM        learns patterns over time
  Dense       outputs 4 scores, one per breathing class
  Softmax     converts to probabilities

Run with:
  py -3.12 scripts/train_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001
VAL_SPLIT  = 0.2

LABEL_NAMES = ['inhale', 'exhale', 'hold_in', 'hold_out']

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir  = os.path.dirname(__file__)
dataset_path = os.path.normpath(os.path.join(scripts_dir, '..', 'data', 'processed', 'dataset.npz'))
models_dir   = os.path.normpath(os.path.join(scripts_dir, '..', 'models'))
model_path   = os.path.join(models_dir, 'breathing_model.pt')

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
data = np.load(dataset_path)
X = data['X']   # (num_windows, 60, 3)
y = data['y']   # (num_windows,)

num_windows, window_size, num_features = X.shape
num_classes = len(LABEL_NAMES)

print(f"  Windows:  {num_windows}")
print(f"  Shape:    {window_size} frames x {num_features} features")
print(f"  Classes:  {num_classes}")
print()

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class BreathingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset    = BreathingDataset(X, y)
val_size   = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {train_size}  |  Validation samples: {val_size}")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
class BreathingModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm       = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)       # (batch, features, frames)
        x = self.conv(x)              # (batch, 32, frames/2)
        x = x.permute(0, 2, 1)       # (batch, frames/2, 32)
        _, (hidden, _) = self.lstm(x)
        x = hidden.squeeze(0)         # (batch, 64)
        x = self.dropout(x)
        return self.classifier(x)     # (batch, 4)

model = BreathingModel(num_features=num_features, num_classes=num_classes)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ── Training ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            preds   = outputs.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total

print(f"Training for {EPOCHS} epochs...")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("-" * 52)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(train_loader)
    val_loss,   val_acc   = evaluate(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        saved = " <-- saved"
    else:
        saved = ""

    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.1%}  {val_loss:>8.4f}  {val_acc:>6.1%}{saved}")

# ── Results ───────────────────────────────────────────────────────────────────
print()
print(f"Best validation accuracy: {best_val_acc:.1%}")
print(f"Model saved to: {model_path}")
print()

model.load_state_dict(torch.load(model_path))
model.eval()

class_correct = [0] * num_classes
class_total   = [0] * num_classes

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        preds = model(X_batch).argmax(dim=1)
        for label, pred in zip(y_batch, preds):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

print("Per-class accuracy on validation set:")
for i, name in enumerate(LABEL_NAMES):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"  {name:<12}  {acc:.1%}  ({class_correct[i]}/{class_total[i]})")
