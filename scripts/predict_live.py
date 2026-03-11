"""
STEP 6 - Live Prediction Script
---------------------------------
Opens your webcam, detects your body with MediaPipe, and uses the trained
model to predict your breathing phase in real time.

How it works:
  - Keeps a rolling buffer of the last 60 frames
  - Every frame: extracts landmarks, adds to buffer
  - Once buffer is full: smooths + normalizes the window, runs the model
  - Displays the predicted breathing phase live on screen

Run with:
  py -3.12 scripts/predict_live.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
from scipy.signal import savgol_filter

# ── Config — must match process_data.py and train_model.py ───────────────────
WINDOW_SIZE   = 60     # frames in the rolling buffer
SMOOTH_WINDOW = 15     # savgol smoothing window
SMOOTH_ORDER  = 3      # savgol polynomial order
NUM_FEATURES  = 26     # 13 landmarks x 2 (x, y)
NUM_CLASSES   = 4

LABEL_NAMES = ['inhale', 'exhale', 'hold_in', 'hold_out']

# Label display colors (BGR)
LABEL_COLORS = {
    'inhale':   (0,   255, 100),
    'exhale':   (0,   100, 255),
    'hold_in':  (255, 200, 0  ),
    'hold_out': (200, 0,   255),
}

# Landmarks to extract — must be same order as in collect_data.py
LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Skeleton lines to draw
POSE_CONNECTIONS = [
    (11, 12),
    (0, 7), (0, 8),
]

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir = os.path.dirname(__file__)
model_path  = os.path.normpath(os.path.join(scripts_dir, '..', 'models', 'breathing_model.pt'))
mp_model    = os.path.normpath(os.path.join(scripts_dir, '..', 'models', 'pose_landmarker_lite.task'))

# ── Model Definition — must match train_model.py exactly ─────────────────────
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
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        x = hidden.squeeze(0)
        x = self.dropout(x)
        return self.classifier(x)

# ── Load trained model ────────────────────────────────────────────────────────
print("Loading model...")
model = BreathingModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("Model loaded.\n")

# ── Setup MediaPipe ───────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=mp_model)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ── Rolling buffer — holds last 60 frames of landmark data ───────────────────
buffer = deque(maxlen=WINDOW_SIZE)

current_label      = None   # latest prediction
current_confidence = 0.0    # confidence of latest prediction

print("Running — press Q to quit.\n")

def predict(buffer):
    """Smooth, normalize and run the model on the current buffer."""
    window = np.array(buffer, dtype=np.float32)   # (60, 26)

    # Smooth each feature column
    for i in range(window.shape[1]):
        window[:, i] = savgol_filter(window[:, i], SMOOTH_WINDOW, SMOOTH_ORDER)

    # Normalize each feature to 0-1 within this window
    col_min   = window.min(axis=0)
    col_max   = window.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1           # avoid division by zero
    window = (window - col_min) / col_range

    # Run model
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 60, 26)
    with torch.no_grad():
        logits = model(x)                            # (1, 4)
        probs  = torch.softmax(logits, dim=1)[0]     # (4,)
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    return LABEL_NAMES[pred], conf, probs.numpy()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    key  = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Run MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Extract features in the same order as collect_data.py
        frame_features = []
        for idx in LANDMARK_INDICES:
            frame_features.append(landmarks[idx].x)
            frame_features.append(landmarks[idx].y)

        buffer.append(frame_features)

        # Draw skeleton
        for (a, b) in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for idx in LANDMARK_INDICES:
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # Predict once buffer is full
        if len(buffer) == WINDOW_SIZE:
            current_label, current_confidence, all_probs = predict(buffer)

            # Draw confidence bars for all 4 classes on the left side
            bar_x     = 10
            bar_width = 150
            for i, name in enumerate(LABEL_NAMES):
                bar_y   = 120 + i * 40
                bar_len = int(all_probs[i] * bar_width)
                color   = LABEL_COLORS[name]
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len,   bar_y + 20), color, -1)
                cv2.putText(frame, f'{name} {all_probs[i]:.0%}', (bar_x + bar_width + 10, bar_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show buffer fill progress until model starts predicting
        else:
            fill_pct = len(buffer) / WINDOW_SIZE
            cv2.putText(frame, f'Warming up... {len(buffer)}/{WINDOW_SIZE}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.rectangle(frame, (10, 70), (10 + int(fill_pct * 200), 85), (100, 200, 100), -1)

    else:
        buffer.clear()   # reset buffer if body is lost
        cv2.putText(frame, 'No body detected — step back', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show current prediction top-right
    if current_label:
        color     = LABEL_COLORS[current_label]
        label_txt = f'{current_label}  {current_confidence:.0%}'
        txt_size  = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, label_txt, (w - txt_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, 'Q = quit', (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Breathing AI — Live', frame)

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Done.")
