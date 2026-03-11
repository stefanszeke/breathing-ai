"""
STEP 2 - Data Collection Script
--------------------------------
Opens your webcam, detects your body with MediaPipe,
and saves chest Y-position + breathing label to a CSV file every frame.

Controls:
  W = inhale
  S = exhale
  E = hold_in  (holding breath after inhale)
  D = hold_out (holding breath after exhale)
  Q = stop and save

How to use:
  1. Run this script
  2. Press the matching key as you breathe — hold it down for the whole phase
  3. Press Q to stop - the CSV file is saved automatically

Output: data/session_001.csv (or next available number)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import time
import os
import urllib.request

# ── Download pose model on first run ─────────────────────────────────────────
model_dir  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')

if not os.path.exists(model_path):
    print("Downloading pose model (one-time, ~3MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!\n")

# ── Find the next available session filename ──────────────────────────────────
data_folder = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
session_num = 1
while os.path.exists(os.path.join(data_folder, f'session_{session_num:03d}.csv')):
    session_num += 1
output_file = os.path.join(data_folder, f'session_{session_num:03d}.csv')

# ── Setup MediaPipe Pose Landmarker ───────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ── Label key mapping ─────────────────────────────────────────────────────────
KEY_LABELS = {
    ord('w'): 'inhale',
    ord('s'): 'exhale',
    ord('e'): 'hold_in',
    ord('d'): 'hold_out',
}

# Label display colors (BGR)
LABEL_COLORS = {
    'inhale':   (0,   255, 100),   # green
    'exhale':   (0,   100, 255),   # orange
    'hold_in':  (255, 200, 0  ),   # cyan-ish
    'hold_out': (200, 0,   255),   # purple
    None:       (100, 100, 100),   # grey — no key pressed
}

def draw_skeleton(frame, landmarks, h, w):
    for (a, b) in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Check it is connected.")
    exit()

print(f"Recording to: {output_file}")
print("Hold a key to label each breathing phase:")
print("  W = inhale  |  S = exhale  |  E = hold_in  |  D = hold_out")
print("  Q = stop and save\n")

# ── Prepare CSV ───────────────────────────────────────────────────────────────
csv_file = open(output_file, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['frame', 'chest_y', 'shoulder_width', 'timestamp', 'label'])

frame_num    = 0
start_time   = time.time()
current_label = None   # label currently held down by user

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost camera feed.")
        break

    h, w = frame.shape[:2]

    # Check which key is being pressed this frame
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in KEY_LABELS:
        current_label = KEY_LABELS[key]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        left_shoulder  = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        chest_y       = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        timestamp      = round(time.time() - start_time, 3)

        # Only save frames that have a label (key is being held)
        if current_label is not None:
            writer.writerow([frame_num, round(chest_y, 5), round(shoulder_width, 5), timestamp, current_label])

        draw_skeleton(frame, landmarks, h, w)

        cv2.putText(frame, f'chest_y: {chest_y:.4f}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'frame: {frame_num}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, 'No body detected - step back', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show current label in top-right corner with its color
    label_text  = current_label if current_label else 'no label'
    label_color = LABEL_COLORS[current_label]
    text_size   = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    cv2.putText(frame, label_text, (w - text_size[0] - 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

    # Controls reminder at the bottom
    cv2.putText(frame, 'W=inhale  S=exhale  E=hold_in  D=hold_out  Q=quit', (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Breathing Data Collection', frame)
    frame_num += 1

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
csv_file.close()
cv2.destroyAllWindows()
landmarker.close()

print(f"\nDone! Saved to: {output_file}")
