"""
STEP 2 - Data Collection Script
--------------------------------
Opens your webcam, detects your body, and saves landmark positions +
breathing label to a CSV file every frame.

VIEW MODE — set VIEW = 'side' or VIEW = 'front' below.

  front: face + shoulders visible, tracks Y movement (chest rising/falling)
         Landmarks: nose, eyes, ears, mouth, shoulders
         Uses: MediaPipe Pose

  side:  body profile visible, tracks X movement (chest/belly expanding forward)
         Landmarks: nose, shoulders, hips
         Much stronger breathing signal — recommended
         Uses: YOLOv8 Pose (better side-view detection than MediaPipe)

Controls:
  W = inhale
  S = exhale
  E = hold_in  (holding breath after inhale)
  D = hold_out (holding breath after exhale)
  Q = stop and save

How to use:
  1. Run this script
  2. Press the matching key when a phase starts — label stays until you press the next key
  3. Press Q to stop - the CSV file is saved automatically

Output: data/side/session_001.csv  or  data/front/session_001.csv
"""

import cv2
import csv
import time
import os
import urllib.request

# ── View mode ────────────────────────────────────────────────────────────────
# Set to 'side' or 'front'
VIEW = 'side'

# ── Find the next available session filename ──────────────────────────────────
data_root   = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
data_folder = os.path.join(data_root, VIEW)
os.makedirs(data_folder, exist_ok=True)

session_num = 1
while os.path.exists(os.path.join(data_folder, f'session_{session_num:03d}.csv')):
    session_num += 1
output_file = os.path.join(data_folder, f'session_{session_num:03d}.csv')

# ── Label key mapping ─────────────────────────────────────────────────────────
KEY_LABELS = {
    ord('w'): 'inhale',
    ord('s'): 'exhale',
    ord('e'): 'hold_in',
    ord('d'): 'hold_out',
}

# Label display colors (BGR)
LABEL_COLORS = {
    'inhale':   (0,   255, 100),
    'exhale':   (0,   100, 255),
    'hold_in':  (255, 200, 0  ),
    'hold_out': (200, 0,   255),
    None:       (100, 100, 100),
}

# =============================================================================
# SIDE VIEW — YOLOv8 Pose
# =============================================================================
if VIEW == 'side':
    from ultralytics import YOLO

    yolo_model = YOLO('yolov8n-pose.pt')

    # COCO keypoint indices: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    # We use the visible-side shoulder + hip to position the 3 boxes
    IDX_LS, IDX_RS, IDX_LH, IDX_RH = 5, 6, 11, 12

    # Box colors (BGR)
    BOX_COLORS = {
        'shoulder': (255, 100, 0  ),
        'chest':    (0,   200, 255),
        'belly':    (0,   255, 100),
    }

    # CSV header — 3 optical-flow signals, one per box
    header = ['frame', 'shoulder_flow', 'chest_flow', 'belly_flow', 'timestamp', 'label']

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print(f"View mode:    SIDE  (YOLOv8 + optical flow boxes)")
    print(f"Recording to: {output_file}")
    print("Press a key when each phase starts:")
    print("  W = inhale  |  S = exhale  |  E = hold_in  |  D = hold_out")
    print("  Q = stop and save\n")
    print("Side view tips:")
    print("  - Sit sideways — shoulder AND hip must be visible")
    print("  - Camera at waist/chest height")
    print("  - Breathe deliberately so the chest/belly clearly expands\n")
    print("  Orange box = shoulder region")
    print("  Cyan  box  = chest region")
    print("  Green box  = belly region\n")

    csv_file = open(output_file, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(header)

    frame_num     = 0
    start_time    = time.time()
    current_label = None
    prev_gray     = None   # previous frame for optical flow

    # Box position locking — set once on first detection, X to reposition
    anchor_sh_px = anchor_sh_py = anchor_hp_px = anchor_hp_py = None
    reposition_requested = False

    def box_flow(flow_x, y1, y2, x1, x2):
        """Mean x-direction optical flow in a pixel box. + = expanding, - = contracting."""
        region = flow_x[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float(region.mean())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost camera feed.")
            break

        h, w = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('x'):
            reposition_requested = True
        elif key in KEY_LABELS:
            current_label = KEY_LABELS[key]

        results    = yolo_model(frame, verbose=False)
        detected   = False

        if (results[0].keypoints is not None
                and len(results[0].keypoints) > 0
                and results[0].keypoints.xy.shape[0] > 0):

            kp = results[0].keypoints.xy[0]   # (17, 2) pixel coords

            # Use whichever shoulder/hip are more visible (larger x = facing right)
            ls = kp[IDX_LS]; rs = kp[IDX_RS]
            lh = kp[IDX_LH]; rh = kp[IDX_RH]

            # Pick the shoulder that has a valid (non-zero) position
            sh = ls if float(ls[0]) > 0 else rs
            hp = lh if float(lh[0]) > 0 else rh

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

                # Torso height in pixels
                torso_h = hp_py - sh_py
                third   = torso_h // 3

                # Horizontal step: each box shifts this many pixels to the left
                # Adjust STEP if the boxes don't land on the right body area
                STEP      = 80
                BOX_HALF  = 80   # half-width of each box

                # shoulder: centred on sh_px
                # chest:    shifted left by 1 step  (right edge = shoulder centre)
                # belly:    shifted left by 2 steps (double the chest offset)
                def hbox(cx):
                    return (max(0, cx - BOX_HALF), min(w - 1, cx + BOX_HALF))

                boxes = {
                    'shoulder': (max(0, sh_py - third//2),         sh_py + third//2,          *hbox(sh_px)),
                    'chest':    (sh_py + third//2,                  sh_py + third + third//2,  *hbox(sh_px - STEP)),
                    'belly':    (sh_py + third + third//2,          min(h-1, hp_py),            *hbox(sh_px - int(STEP * 1.5))),
                }

                # Compute optical flow if we have a previous frame
                shoulder_flow = chest_flow = belly_flow = 0.0
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    flow_x = flow[:, :, 0]   # horizontal motion component
                    shoulder_flow = box_flow(flow_x, *boxes['shoulder'])
                    chest_flow    = box_flow(flow_x, *boxes['chest'])
                    belly_flow    = box_flow(flow_x, *boxes['belly'])

                # Draw the 3 boxes
                for name, (by1, by2, bx1, bx2) in boxes.items():
                    color = BOX_COLORS[name]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(frame, name, (bx2 + 5, (by1 + by2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw shoulder + hip dots
                cv2.circle(frame, (sh_px, sh_py), 6, (0, 255, 255), -1)
                cv2.circle(frame, (hp_px, hp_py), 6, (0, 255, 255), -1)

                # Show flow values on screen
                cv2.putText(frame, f'shoulder: {shoulder_flow:+.3f}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLORS['shoulder'], 2)
                cv2.putText(frame, f'chest:    {chest_flow:+.3f}',    (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLORS['chest'],    2)
                cv2.putText(frame, f'belly:    {belly_flow:+.3f}',    (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLORS['belly'],    2)

                timestamp = round(time.time() - start_time, 3)
                if current_label is not None and prev_gray is not None:
                    writer.writerow([
                        frame_num,
                        round(shoulder_flow, 5),
                        round(chest_flow,    5),
                        round(belly_flow,    5),
                        timestamp,
                        current_label,
                    ])

        if not detected:
            prev_gray = None   # reset flow if body lost
            cv2.putText(frame, 'No body detected - sit sideways, step back',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        label_text  = current_label if current_label else 'no label'
        label_color = LABEL_COLORS[current_label]
        text_size   = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, label_text, (w - text_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        cv2.putText(frame, 'W=inhale  S=exhale  E=hold_in  D=hold_out  X=repos  Q=quit',
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Breathing Data Collection — SIDE (boxes)', frame)
        prev_gray = curr_gray
        frame_num += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nDone! Saved to: {output_file}")


# =============================================================================
# FRONT VIEW — MediaPipe Pose (unchanged)
# =============================================================================
else:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_dir  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')

    if not os.path.exists(model_path):
        print("Downloading pose model (one-time, ~3MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded!\n")

    LANDMARKS_TO_RECORD = [
        (0,  'nose'),
        (1,  'left_eye_inner'),
        (2,  'left_eye'),
        (3,  'left_eye_outer'),
        (4,  'right_eye_inner'),
        (5,  'right_eye'),
        (6,  'right_eye_outer'),
        (7,  'left_ear'),
        (8,  'right_ear'),
        (9,  'mouth_left'),
        (10, 'mouth_right'),
        (11, 'left_shoulder'),
        (12, 'right_shoulder'),
    ]
    POSE_CONNECTIONS = [(11, 12), (0, 7), (0, 8)]

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

    header = ['frame']
    for _, name in LANDMARKS_TO_RECORD:
        header += [f'{name}_x', f'{name}_y']
    header += ['shoulder_width', 'timestamp', 'label']

    def draw_skeleton(frame, landmarks, h, w):
        for (a, b) in POSE_CONNECTIONS:
            if a < len(landmarks) and b < len(landmarks):
                x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
                x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for idx, _ in LANDMARKS_TO_RECORD:
            if idx < len(landmarks):
                cx = int(landmarks[idx].x * w)
                cy = int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    print(f"View mode:    FRONT  (MediaPipe)")
    print(f"Recording to: {output_file}")
    print("Press a key when each phase starts:")
    print("  W = inhale  |  S = exhale  |  E = hold_in  |  D = hold_out")
    print("  Q = stop and save\n")

    csv_file = open(output_file, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(header)

    frame_num     = 0
    start_time    = time.time()
    current_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost camera feed.")
            break

        h, w = frame.shape[:2]

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

            left_shoulder  = landmarks[11]
            right_shoulder = landmarks[12]
            chest_y        = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            timestamp      = round(time.time() - start_time, 3)

            if current_label is not None:
                row = [frame_num]
                for idx, _ in LANDMARKS_TO_RECORD:
                    lm = landmarks[idx]
                    row += [round(lm.x, 5), round(lm.y, 5)]
                row += [round(shoulder_width, 5), timestamp, current_label]
                writer.writerow(row)

            draw_skeleton(frame, landmarks, h, w)

            cv2.putText(frame, f'chest_y: {chest_y:.4f}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'frame: {frame_num}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, 'No body detected - step back', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        label_text  = current_label if current_label else 'no label'
        label_color = LABEL_COLORS[current_label]
        text_size   = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, label_text, (w - text_size[0] - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        cv2.putText(frame, 'W=inhale  S=exhale  E=hold_in  D=hold_out  Q=quit',
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Breathing Data Collection — FRONT (MediaPipe)', frame)
        frame_num += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nDone! Saved to: {output_file}")
