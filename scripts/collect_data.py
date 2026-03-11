"""
STEP 2 - Data Collection Script
--------------------------------
Opens your webcam, detects your body with MediaPipe,
and saves chest Y-position to a CSV file every frame.

How to use:
  1. Run this script (it downloads the pose model on first run, ~3MB)
  2. Breathe deliberately (inhale, exhale, hold...)
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
# MediaPipe 0.10+ needs a model file to detect poses
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

# ── Setup MediaPipe Pose Landmarker (new Tasks API) ───────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,  # process each frame independently
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# ── Landmark indices we care about ────────────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

# Pairs of landmarks to draw as lines (the skeleton)
POSE_CONNECTIONS = [
    (11, 12),           # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso sides
    (23, 24),           # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28), # right leg
]

def draw_skeleton(frame, landmarks, h, w):
    """Draw pose skeleton lines and dots on the frame."""
    for (a, b) in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

# ── Open the webcam ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Check it is connected.")
    exit()

print(f"Recording to: {output_file}")
print("Press Q to stop recording.\n")

# ── Prepare the CSV file ──────────────────────────────────────────────────────
csv_file = open(output_file, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['frame', 'chest_y', 'shoulder_width', 'timestamp'])

frame_num  = 0
start_time = time.time()

# ── Main loop - runs once per camera frame ────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Lost camera feed.")
        break

    h, w = frame.shape[:2]

    # Convert frame to MediaPipe image format (needs RGB, not BGR)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    # Run pose detection on this frame
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # [0] = first person in frame

        left_shoulder  = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        # chest_y = average Y of both shoulders (0.0 = top, 1.0 = bottom of frame)
        # it rises when you inhale and falls when you exhale
        chest_y = (left_shoulder.y + right_shoulder.y) / 2

        # shoulder_width helps normalize for distance from camera
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        timestamp = round(time.time() - start_time, 3)
        writer.writerow([frame_num, round(chest_y, 5), round(shoulder_width, 5), timestamp])

        draw_skeleton(frame, landmarks, h, w)

        cv2.putText(frame, f'chest_y: {chest_y:.4f}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'frame: {frame_num}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, 'No body detected - step back', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, 'Press Q to stop', (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Breathing Data Collection', frame)
    frame_num += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
csv_file.close()
cv2.destroyAllWindows()
landmarker.close()

print(f"\nDone! Saved {frame_num} frames to: {output_file}")
