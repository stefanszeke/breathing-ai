# Breathing AI

A computer vision project that uses your webcam to detect and classify breathing phases in real time.

The camera tracks your chest movement using body landmark detection. That movement signal is recorded, labeled, and used to train a neural network that can recognize four breathing states: **inhale**, **exhale**, **hold in**, and **hold out**.

---

## How It Works

```
Webcam → MediaPipe detects body → chest Y-position recorded
       → data cleaned & labeled
       → neural network trained
       → live predictions from webcam
```

The key insight: when you inhale, your chest rises (Y goes down in image coordinates). When you exhale, it falls. The AI learns the shape of those movements over time.

---

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment setup | ✅ Done |
| 2 | Data collection | ✅ Done |
| 3 | Data processing & smoothing | ⬜ Next |
| 4 | Labeling | ⬜ Pending |
| 5 | Train model | ⬜ Pending |
| 6 | Test & improve | ⬜ Pending |
| 7 | Live integration | ⬜ Pending |

---

## Folder Structure

```
breathing-ai/
│
├── data/                        # Recorded CSV sessions
│   ├── session_001.csv
│   ├── session_002.csv
│   └── ...
│
├── models/                      # AI model files
│   └── pose_landmarker_lite.task   # MediaPipe body detection model (auto-downloaded)
│
├── scripts/                     # Python scripts
│   └── collect_data.py          # Step 2 — webcam recording script
│
└── readme.md
```

### CSV Data Format

Each session file contains one row per camera frame:

| Column | Description |
|--------|-------------|
| `frame` | Frame number (0, 1, 2, ...) |
| `chest_y` | Average Y position of both shoulders (0.0 = top, 1.0 = bottom) |
| `shoulder_width` | Distance between shoulders — helps normalize for camera distance |
| `timestamp` | Seconds elapsed since recording started |

---

## Libraries

| Library | Purpose |
|---------|---------|
| `opencv-python` | Reads webcam, draws skeleton and text on screen, shows the window |
| `mediapipe` | Google's pre-trained body detection AI — finds 33 body landmarks |
| `numpy` | Fast math on large arrays of numbers — backbone of all data processing |
| `pandas` | Load and inspect CSV files as tables (like Excel in code) |
| `scipy` | Signal smoothing — cleans up the noisy chest_y data |
| `matplotlib` | Plot charts to visualize the breathing signal |
| `torch` | PyTorch — the AI framework used to build and train the neural network |
| `jupyter` | Interactive notebook environment for running code step by step |

---

## Requirements

- **Python 3.12** (3.13+ is too new — some libraries like MediaPipe are not compatible yet)
- A webcam
- Windows / macOS / Linux

---

## Installation

**1. Clone or download this project**

**2. Install all dependencies using Python 3.12 specifically**

```bash
py -3.12 -m pip install opencv-python mediapipe scipy numpy pandas matplotlib torch jupyter
```

**3. Verify everything installed correctly**

```bash
py -3.12 -c "import cv2, mediapipe, torch; print('All good! PyTorch version:', torch.__version__)"
```

**4. That's it.** The MediaPipe pose model (~3MB) is downloaded automatically on first run.

> **Note:** Always use `py -3.12` to run scripts in this project, not just `python` or `py`, to make sure you're using the correct Python version.

---

## Running the Data Collection

```bash
py -3.12 scripts/collect_data.py
```

- A window opens showing your webcam with a skeleton overlay
- The `chest_y` value is shown live on screen
- Breathe deliberately — inhale, exhale, hold
- Press **Q** to stop — data is saved automatically to `data/session_XXX.csv`

---

## Planned Model Architecture

- **Input:** 30-frame sliding window of `chest_y` values
- **Architecture:** Conv1D → MaxPooling → LSTM → Dense (softmax)
- **Output:** One of 4 classes — `inhale`, `exhale`, `hold_in`, `hold_out`