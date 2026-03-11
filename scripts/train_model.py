"""
STEP 5 - Model Training Script
--------------------------------
Loads the processed dataset, builds a Conv1D + LSTM neural network,
trains it, and saves the trained model.

Architecture:
  Input:      (batch, 30 frames, 26 features)
  Conv1D      learns local patterns within the window (e.g. "rising signal")
  MaxPool1D   downsamples, keeps the strongest patterns
  LSTM        learns patterns over time across the sequence
  Dense       outputs 4 scores, one per breathing class
  Softmax     converts scores to probabilities (0-1, sum to 1)

Output:
  models/breathing_model.pt   the trained model weights
  models/label_map.txt         label index -> name mapping

Run with:
  py -3.12 scripts/train_model.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 100        # how many times to train through the full dataset
BATCH_SIZE  = 64        # how many windows to process at once
LR          = 0.001     # learning rate — how fast the model updates its weights
VAL_SPLIT   = 0.2       # 20% of data held out for validation (not used in training)

LABEL_NAMES = ['inhale', 'exhale', 'hold_in', 'hold_out']

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir  = os.path.dirname(__file__)
dataset_path = os.path.normpath(os.path.join(scripts_dir, '..', 'data', 'processed', 'dataset.npz'))
models_dir   = os.path.normpath(os.path.join(scripts_dir, '..', 'models'))
model_path   = os.path.join(models_dir, 'breathing_model.pt')

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
data = np.load(dataset_path)
X = data['X']   # shape: (9098, 30, 26)
y = data['y']   # shape: (9098,)

num_windows, window_size, num_features = X.shape
num_classes = len(LABEL_NAMES)

print(f"  Windows:  {num_windows}")
print(f"  Shape:    {window_size} frames x {num_features} features")
print(f"  Classes:  {num_classes}")
print()

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class BreathingDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset so DataLoader can use them."""
    def __init__(self, X, y):
        # X: (N, 30, 26) float32
        # y: (N,)        int64
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = BreathingDataset(X, y)

# Split into training and validation sets
val_size   = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {train_size}  |  Validation samples: {val_size}")
print()

# ── Model Definition ──────────────────────────────────────────────────────────
class BreathingModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Conv1D block — detects local patterns in the time window
        # Input shape to conv: (batch, channels=num_features, length=30)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 30 -> 15 time steps
        )

        # LSTM — learns patterns across the sequence of conv outputs
        # Input to LSTM: (batch, seq_len=15, features=32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

        # Dropout — randomly zeros some values during training to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Final classifier — maps 64 LSTM outputs to 4 class scores
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, 30, 26)

        # Conv1D expects (batch, channels, length) — swap last two dims
        x = x.permute(0, 2, 1)          # -> (batch, 26, 30)
        x = self.conv(x)                 # -> (batch, 32, 15)

        # LSTM expects (batch, seq, features) — swap back
        x = x.permute(0, 2, 1)          # -> (batch, 15, 32)
        _, (hidden, _) = self.lstm(x)   # hidden: (1, batch, 64)
        x = hidden.squeeze(0)           # -> (batch, 64)

        x = self.dropout(x)
        x = self.classifier(x)          # -> (batch, 4)
        return x

model = BreathingModel(num_features=num_features, num_classes=num_classes)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print()

# ── Training Setup ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()           # measures how wrong the predictions are
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # updates weights to reduce loss

def evaluate(loader):
    """Run model on a data loader, return average loss and accuracy."""
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

# ── Training Loop ─────────────────────────────────────────────────────────────
print(f"Training for {EPOCHS} epochs...")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("-" * 52)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()           # clear gradients from last step
        outputs = model(X_batch)        # forward pass
        loss    = criterion(outputs, y_batch)
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights

    train_loss, train_acc = evaluate(train_loader)
    val_loss,   val_acc   = evaluate(val_loader)

    # Save the best model seen so far
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

# Per-class accuracy on validation set
print("Per-class accuracy on validation set:")
model.load_state_dict(torch.load(model_path))
model.eval()

class_correct = [0] * num_classes
class_total   = [0] * num_classes

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        preds   = outputs.argmax(dim=1)
        for label, pred in zip(y_batch, preds):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

for i, name in enumerate(LABEL_NAMES):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"  {name:<12}  {acc:.1%}  ({class_correct[i]}/{class_total[i]})")
