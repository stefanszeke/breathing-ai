"""
Quick test — opens webcam with YOLOv8 pose detection.
Sit sideways and check if your shoulder and hip landmarks are detected.
Press Q to quit.
"""

import cv2
from ultralytics import YOLO

# Downloads yolov8n-pose model automatically on first run (~6MB)
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Running YOLOv8 pose test — sit sideways and check detection.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()   # draws skeleton + keypoints automatically

    # Print detected keypoint count to terminal
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kpts = results[0].keypoints.xy[0]   # (17, 2) — 17 COCO keypoints
        # COCO keypoint 5=left shoulder, 6=right shoulder, 11=left hip, 12=right hip
        ls  = kpts[5]
        rs  = kpts[6]
        lh  = kpts[11]
        rh  = kpts[12]
        print(f"L.shoulder={ls[0]:.0f},{ls[1]:.0f}  R.shoulder={rs[0]:.0f},{rs[1]:.0f}  "
              f"L.hip={lh[0]:.0f},{lh[1]:.0f}  R.hip={rh[0]:.0f},{rh[1]:.0f}", end='\r')

    cv2.imshow('YOLOv8 Pose Test', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
