"""
STEP 6 - Live Prediction Script
---------------------------------
Opens your webcam, tracks your breathing using optical flow boxes,
and predicts your breathing phase in real time.

How it works:
  - YOLO finds your shoulder + hip to position 3 boxes (shoulder, chest, belly)
  - Each frame: optical flow measures x-direction movement inside each box
  - 60-frame rolling buffer of those 3 signals feeds the trained model
  - Predicted phase shown top-right with confidence bars on the left

Run with:
  py -3.12 scripts/predict_live.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
from scipy.signal import savgol_filter

# ── Config — must match process_data.py ──────────────────────────────────────
WINDOW_SIZE   = 60
SMOOTH_WINDOW = 11
SMOOTH_ORDER  = 3
NUM_FEATURES  = 3
NUM_CLASSES   = 4

LABEL_NAMES = ['inhale', 'exhale', 'hold_in', 'hold_out']

LABEL_COLORS = {
    'inhale':   (0,   255, 100),
    'exhale':   (0,   100, 255),
    'hold_in':  (255, 200, 0  ),
    'hold_out': (200, 0,   255),
}

BOX_COLORS = {
    'shoulder': (255, 100, 0  ),
    'chest':    (0,   200, 255),
    'belly':    (0,   255, 100),
}

# Box positioning — must match collect_data.py
STEP     = 80
BOX_HALF = 80

# COCO keypoint indices
IDX_LS, IDX_RS, IDX_LH, IDX_RH = 5, 6, 11, 12

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir = os.path.dirname(__file__)
model_path  = os.path.normpath(os.path.join(scripts_dir, '..', 'models', 'breathing_model.pt'))

# ── Model definition — must match train_model.py exactly ─────────────────────
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

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = BreathingModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("Model loaded.\n")

# ── Load YOLO ─────────────────────────────────────────────────────────────────
yolo_model = YOLO('yolov8n-pose.pt')

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ── Rolling buffer ────────────────────────────────────────────────────────────
buffer    = deque(maxlen=WINDOW_SIZE)
prev_gray = None

# Smoothing: keep last N raw predictions, display the majority vote
PRED_HISTORY     = 10     # number of recent predictions to vote over
CONF_THRESHOLD   = 0.60   # ignore predictions below this confidence

pred_history       = deque(maxlen=PRED_HISTORY)
current_label      = None
current_confidence = 0.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def box_flow(flow_x, y1, y2, x1, x2):
    region = flow_x[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.mean())

def hbox(cx, w):
    return (max(0, cx - BOX_HALF), min(w - 1, cx + BOX_HALF))

def predict(buffer):
    window = np.array(buffer, dtype=np.float32)   # (60, 3)
    for i in range(window.shape[1]):
        window[:, i] = savgol_filter(window[:, i], SMOOTH_WINDOW, SMOOTH_ORDER)
    col_min   = window.min(axis=0)
    col_max   = window.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    window = (window - col_min) / col_range
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()
    return LABEL_NAMES[pred], conf, probs.numpy()

print("Running — press Q to quit.\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    results  = yolo_model(frame, verbose=False)
    detected = False

    if (results[0].keypoints is not None
            and len(results[0].keypoints) > 0
            and results[0].keypoints.xy.shape[0] > 0):

        kp = results[0].keypoints.xy[0]

        sh = kp[IDX_LS] if float(kp[IDX_LS][0]) > 0 else kp[IDX_RS]
        hp = kp[IDX_LH] if float(kp[IDX_LH][0]) > 0 else kp[IDX_RH]

        sh_px, sh_py = int(float(sh[0])), int(float(sh[1]))
        hp_px, hp_py = int(float(hp[0])), int(float(hp[1]))

        if sh_px > 0 and hp_px > 0 and hp_py > sh_py:
            detected = True

            torso_h = hp_py - sh_py
            third   = torso_h // 3

            boxes = {
                'shoulder': (max(0, sh_py - third//2), sh_py + third//2,         *hbox(sh_px,               w)),
                'chest':    (sh_py + third//2,          sh_py + third + third//2, *hbox(sh_px - STEP,        w)),
                'belly':    (sh_py + third + third//2,  min(h-1, hp_py),          *hbox(sh_px - int(STEP*1.5), w)),
            }

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                flow_x = flow[:, :, 0]
                buffer.append([
                    box_flow(flow_x, *boxes['shoulder']),
                    box_flow(flow_x, *boxes['chest']),
                    box_flow(flow_x, *boxes['belly']),
                ])

            # Draw boxes
            for name, (by1, by2, bx1, bx2) in boxes.items():
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), BOX_COLORS[name], 2)

            cv2.circle(frame, (sh_px, sh_py), 6, (0, 255, 255), -1)
            cv2.circle(frame, (hp_px, hp_py), 6, (0, 255, 255), -1)

            if len(buffer) == WINDOW_SIZE:
                raw_label, raw_conf, all_probs = predict(buffer)

                # Only count this prediction if confident enough
                if raw_conf >= CONF_THRESHOLD:
                    pred_history.append(raw_label)

                # Displayed label = majority vote over recent history
                if pred_history:
                    from collections import Counter
                    current_label      = Counter(pred_history).most_common(1)[0][0]
                    current_confidence = raw_conf

                bar_x     = 10
                bar_width = 150
                for i, name in enumerate(LABEL_NAMES):
                    bar_y   = 120 + i * 40
                    bar_len = int(all_probs[i] * bar_width)
                    color   = LABEL_COLORS[name]
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len,   bar_y + 20), color, -1)
                    cv2.putText(frame, f'{name} {all_probs[i]:.0%}',
                                (bar_x + bar_width + 10, bar_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                fill_pct = len(buffer) / WINDOW_SIZE
                cv2.putText(frame, f'Warming up... {len(buffer)}/{WINDOW_SIZE}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.rectangle(frame, (10, 70), (10 + int(fill_pct * 200), 85), (100, 200, 100), -1)

    if not detected:
        prev_gray = None
        buffer.clear()
        pred_history.clear()
        cv2.putText(frame, 'No body detected - sit sideways, step back',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if current_label:
        color    = LABEL_COLORS[current_label]
        txt      = f'{current_label}  {current_confidence:.0%}'
        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, txt, (w - txt_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, 'Q = quit', (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Breathing AI - Live', frame)
    prev_gray = curr_gray

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Done.")
