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
import time
import csv
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
NUM_FEATURES  = 12  # 3 signed-norm flow + 3 std + 3 mean_abs + 3 mean_signed
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
logs_dir    = os.path.normpath(os.path.join(scripts_dir, '..', 'logs'))
os.makedirs(logs_dir, exist_ok=True)

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

# Box position locking — set once on first detection, X to reposition
anchor_sh_px = anchor_sh_py = anchor_hp_px = anchor_hp_py = None
reposition_requested = False

# Smoothing: rolling average of probabilities, label = highest averaged bar
SMOOTH_PROBS_LEN = 15   # number of frames to average probabilities over

prob_history          = deque(maxlen=SMOOTH_PROBS_LEN)
current_label         = None
session_log           = []   # rows: [timestamp, label, inhale%, exhale%, hold_in%, hold_out%]
session_start         = time.time()
locked_hold_type      = None   # set at hold onset from last active label
last_active_label     = None   # last non-hold label (inhale/exhale)
active_frames_since_hold = 0   # how many consecutive non-hold frames since lock set
HOLD_RELEASE_COUNT    = 8      # require this many consecutive active frames to release lock


# ── Helpers ───────────────────────────────────────────────────────────────────
def box_flow(flow_x, y1, y2, x1, x2):
    region = flow_x[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.mean())

def hbox(cx, w):
    return (max(0, cx - BOX_HALF), min(w - 1, cx + BOX_HALF))

HOLD_THRESHOLD = 0.020   # mean abs chest_flow below this → force hold

def predict(buffer):
    window = np.array(buffer, dtype=np.float32)   # (60, 3)
    for i in range(window.shape[1]):
        window[:, i] = savgol_filter(window[:, i], SMOOTH_WINDOW, SMOOTH_ORDER)

    # Std per box BEFORE normalization
    window_std       = window.std(axis=0)                          # (3,)
    std_feature      = np.tile(window_std, (WINDOW_SIZE, 1))      # (60, 3)

    # Mean absolute value — separates holds from active breathing
    mean_abs         = np.abs(window).mean(axis=0)                 # (3,)
    mean_abs_feature = np.tile(mean_abs, (WINDOW_SIZE, 1))         # (60, 3)

    # Signed mean — net direction: positive = expanding (inhale), negative = contracting (exhale)
    mean_signed         = window.mean(axis=0)                      # (3,)
    mean_signed_feature = np.tile(mean_signed, (WINDOW_SIZE, 1))   # (60, 3)

    # Normalize to [-1, 1] — preserves sign unlike 0-1 normalization
    abs_max = np.abs(window).max(axis=0)
    abs_max[abs_max == 0] = 1
    window = window / abs_max                                      # (60, 3), range [-1, 1]

    # Combine: 3 signed-norm + 3 std + 3 mean_abs + 3 mean_signed = 12 features
    window = np.concatenate([window, std_feature, mean_abs_feature, mean_signed_feature], axis=1)  # (60, 12)

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    # Hard override: very low movement → force hold
    mean_abs_cf = mean_abs[1]   # chest_flow column
    if mean_abs_cf <= HOLD_THRESHOLD:
        conf = 0.5 + (1.0 - mean_abs_cf / HOLD_THRESHOLD) * 0.4
        return None, conf, probs.numpy(), mean_abs_cf   # None = let state machine pick hold type

    return LABEL_NAMES[pred], conf, probs.numpy(), mean_abs_cf

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
    elif key == ord('x'):
        reposition_requested = True

    results  = yolo_model(frame, verbose=False)
    detected = False

    if (results[0].keypoints is not None
            and len(results[0].keypoints) > 0
            and results[0].keypoints.xy.shape[0] > 0):

        kp = results[0].keypoints.xy[0]

        sh = kp[IDX_LS] if float(kp[IDX_LS][0]) > 0 else kp[IDX_RS]
        hp = kp[IDX_LH] if float(kp[IDX_LH][0]) > 0 else kp[IDX_RH]

        raw_sh_px, raw_sh_py = float(sh[0]), float(sh[1])
        raw_hp_px, raw_hp_py = float(hp[0]), float(hp[1])

        if raw_sh_px > 0 and raw_hp_px > 0 and raw_hp_py > raw_sh_py:
            if anchor_sh_px is None or reposition_requested:
                anchor_sh_px, anchor_sh_py = raw_sh_px, raw_sh_py
                anchor_hp_px, anchor_hp_py = raw_hp_px, raw_hp_py
                reposition_requested = False

        sh_px, sh_py = int(anchor_sh_px or raw_sh_px), int(anchor_sh_py or raw_sh_py)
        hp_px, hp_py = int(anchor_hp_px or raw_hp_px), int(anchor_hp_py or raw_hp_py)

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
                _, _, all_probs, _ = predict(buffer)

                # Smooth probabilities by averaging over recent frames
                prob_history.append(all_probs.copy())
                avg_probs = np.mean(prob_history, axis=0)
                top_idx   = int(np.argmax(avg_probs))
                top_label = LABEL_NAMES[top_idx]

                # Hold type: lock at onset using last active label, release after sustained active
                if top_label in ('hold_in', 'hold_out'):
                    active_frames_since_hold = 0
                    if locked_hold_type is None:
                        # Use last known breathing direction to pick hold type
                        if last_active_label == 'inhale':
                            locked_hold_type = 'hold_in'
                        elif last_active_label == 'exhale':
                            locked_hold_type = 'hold_out'
                        else:
                            locked_hold_type = top_label  # fallback: trust model
                    top_label = locked_hold_type
                else:
                    last_active_label = top_label  # track last inhale/exhale
                    if locked_hold_type is not None:
                        active_frames_since_hold += 1
                        if active_frames_since_hold >= HOLD_RELEASE_COUNT:
                            locked_hold_type = None
                            active_frames_since_hold = 0

                current_label = top_label

                # Sync bars with lock: merge both hold probs into the locked hold type
                display_probs = avg_probs.copy()
                if locked_hold_type is not None:
                    hi_idx = LABEL_NAMES.index('hold_in')
                    ho_idx = LABEL_NAMES.index('hold_out')
                    locked_idx = LABEL_NAMES.index(locked_hold_type)
                    other_idx  = ho_idx if locked_hold_type == 'hold_in' else hi_idx
                    display_probs[locked_idx] += display_probs[other_idx]
                    display_probs[other_idx]   = 0.0

                session_log.append([
                    round(time.time() - session_start, 2),
                    current_label,
                    round(float(display_probs[0]), 4),
                    round(float(display_probs[1]), 4),
                    round(float(display_probs[2]), 4),
                    round(float(display_probs[3]), 4),
                ])

                bar_x     = 10
                bar_width = 150
                for i, name in enumerate(LABEL_NAMES):
                    bar_y   = 120 + i * 40
                    bar_len = int(display_probs[i] * bar_width)
                    color   = LABEL_COLORS[name]
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len,   bar_y + 20), color, -1)
                    cv2.putText(frame, f'{name} {display_probs[i]:.0%}',
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
        prob_history.clear()
        current_label = None
        last_active_label = None
        locked_hold_type = None
        active_frames_since_hold = 0
        cv2.putText(frame, 'No body detected - sit sideways, step back',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if current_label and len(prob_history) > 0:
        avg_p    = float(np.mean(prob_history, axis=0)[LABEL_NAMES.index(current_label)])
        color    = LABEL_COLORS[current_label]
        txt      = f'{current_label}  {avg_p:.0%}'
        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, txt, (w - txt_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, 'X = reposition boxes  |  Q = quit', (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Breathing AI - Live', frame)
    prev_gray = curr_gray

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

if session_log:
    ts  = time.strftime('%Y%m%d_%H%M%S')

    # Full frame-by-frame log
    csv_path = os.path.join(logs_dir, f'session_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'label', 'inhale', 'exhale', 'hold_in', 'hold_out'])
        w.writerows(session_log)

    # Phase summary: one row per phase with duration in seconds
    phases = []
    cur_label = session_log[0][1]
    cur_start = session_log[0][0]
    for row in session_log[1:]:
        if row[1] != cur_label:
            phases.append([cur_label, cur_start, row[0], round(row[0] - cur_start, 2)])
            cur_label, cur_start = row[1], row[0]
    phases.append([cur_label, cur_start, session_log[-1][0],
                   round(session_log[-1][0] - cur_start, 2)])

    summary_path = os.path.join(logs_dir, f'summary_{ts}.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'start_s', 'end_s', 'duration_s'])
        w.writerows(phases)

    print(f"Session saved  → {csv_path}  ({len(session_log)} frames)")
    print(f"Summary saved  → {summary_path}  ({len(phases)} phases)")
    print()
    print(f"{'label':<12} {'duration':>8}")
    print("-" * 22)
    for p in phases:
        print(f"{p[0]:<12} {p[3]:>7.1f}s")

print("Done.")
