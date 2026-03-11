"""
STEP 3 - Data Processing Script
---------------------------------
Loads all labeled session CSVs, smooths the signals, normalizes values,
then slices them into 30-frame windows ready for model training.

What this script does, step by step:
  1. Load all session_XXX.csv files from the data/ folder
  2. Smooth every feature column with a Savitzky-Golay filter (removes noise)
  3. Normalize each feature to 0-1 range (so all landmarks are on the same scale)
  4. Slide a 30-frame window across the data — each window = one training sample
  5. Label each window by majority vote (whatever phase most frames in it are)
  6. Save the result to data/processed/dataset.npz (a compressed numpy file)

Output:
  data/processed/dataset.npz  contains:
    X  — shape (num_windows, 30, num_features)   the input windows
    y  — shape (num_windows,)                     the label for each window

Run with:
  py -3.12 scripts/process_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 60      # frames per training sample (~1 second at 30fps)
STEP_SIZE     = 5       # how many frames to slide the window each step
                        # smaller = more training samples (more overlap)

SMOOTH_WINDOW = 15      # savgol filter window length (must be odd)
SMOOTH_ORDER  = 3       # savgol polynomial order

# Features to use for training — all x,y landmark columns
# (timestamp, frame, shoulder_width, label are excluded)
EXCLUDE_COLS = {'frame', 'timestamp', 'shoulder_width', 'label'}

# Label encoding — maps string label to number for the model
LABEL_MAP = {
    'inhale':   0,
    'exhale':   1,
    'hold_in':  2,
    'hold_out': 3,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir   = os.path.dirname(__file__)
data_dir      = os.path.normpath(os.path.join(scripts_dir, '..', 'data'))
processed_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_dir, exist_ok=True)
output_path   = os.path.join(processed_dir, 'dataset.npz')

# ── Load all session CSV files ────────────────────────────────────────────────
csv_files = sorted(
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.startswith('session_') and f.endswith('.csv')
)

if not csv_files:
    print("ERROR: No session CSV files found in data/")
    exit()

print(f"Found {len(csv_files)} session files:")
for f in csv_files:
    print(f"  {os.path.basename(f)}")
print()

# ── Process each session ──────────────────────────────────────────────────────
all_X = []   # will collect all windows from all sessions
all_y = []   # will collect all labels from all sessions

for csv_path in csv_files:
    session_name = os.path.basename(csv_path)
    df = pd.read_csv(csv_path)

    # Skip sessions with no labels or too few rows
    if 'label' not in df.columns:
        print(f"  SKIP {session_name} — no label column")
        continue
    if len(df) < WINDOW_SIZE:
        print(f"  SKIP {session_name} — too short ({len(df)} frames)")
        continue

    # Get feature columns (everything except excluded ones)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # ── Step 1: Smooth each feature column ───────────────────────────────────
    smoothed = df[feature_cols].copy()
    for col in feature_cols:
        smoothed[col] = savgol_filter(
            df[col].values,
            window_length=SMOOTH_WINDOW,
            polyorder=SMOOTH_ORDER
        )

    # ── Step 2: Normalize each feature to 0-1 range ──────────────────────────
    col_min = smoothed.min()
    col_max = smoothed.max()
    col_range = col_max - col_min
    # Avoid division by zero for constant columns
    col_range[col_range == 0] = 1
    normalized = (smoothed - col_min) / col_range

    features = normalized.values          # shape: (num_frames, num_features)
    labels   = df['label'].values         # shape: (num_frames,)

    # ── Step 3: Slide window across the session ───────────────────────────────
    session_windows = 0
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        window_features = features[start:end]   # shape: (30, num_features)
        window_labels   = labels[start:end]      # 30 label strings

        # Label the window by majority vote
        label_counts  = Counter(window_labels)
        majority_label = label_counts.most_common(1)[0][0]

        # Skip window if majority label is not one of our 4 classes
        if majority_label not in LABEL_MAP:
            continue

        all_X.append(window_features)
        all_y.append(LABEL_MAP[majority_label])
        session_windows += 1

    print(f"  {session_name}  {len(df):>5} frames  ->  {session_windows} windows")

# ── Combine all sessions ──────────────────────────────────────────────────────
X = np.array(all_X, dtype=np.float32)   # shape: (total_windows, 30, num_features)
y = np.array(all_y, dtype=np.int64)     # shape: (total_windows,)

print()
print(f"Dataset shape:    X={X.shape}  y={y.shape}")
print(f"Features per frame: {X.shape[2]}")
print()

# Show label distribution
label_names = {v: k for k, v in LABEL_MAP.items()}
print("Label distribution:")
for label_id, count in sorted(Counter(y.tolist()).items()):
    pct = count / len(y) * 100
    print(f"  {label_names[label_id]:<12}  {count:>5} windows  ({pct:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(output_path, X=X, y=y)
print()
print(f"Saved to: {output_path}")
