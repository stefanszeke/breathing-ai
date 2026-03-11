"""
STEP 3 - Data Processing Script
---------------------------------
Loads all labeled session CSVs from data/side/, smooths the optical-flow
signals, normalizes values, then slices into windows ready for training.

What this script does:
  1. Load all session_XXX.csv files from data/side/
  2. Smooth every feature column with a Savitzky-Golay filter (removes noise)
  3. Normalize each feature to 0-1 range within each session
  4. Slide a window across the data — each window = one training sample
  5. Label each window by majority vote
  6. Save to data/processed/dataset.npz

Features used: shoulder_flow, chest_flow, belly_flow  (3 total)

Run with:
  py -3.12 scripts/process_data.py
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 60      # frames per training sample (~3 seconds at 20fps)
STEP_SIZE     = 5       # slide step — smaller = more training samples
SMOOTH_WINDOW = 11      # savgol filter window (must be odd, <= session length)
SMOOTH_ORDER  = 3       # savgol polynomial order

FEATURE_COLS = ['shoulder_flow', 'chest_flow', 'belly_flow']

LABEL_MAP = {
    'inhale':   0,
    'exhale':   1,
    'hold_in':  2,
    'hold_out': 3,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
scripts_dir   = os.path.dirname(__file__)
data_dir      = os.path.normpath(os.path.join(scripts_dir, '..', 'data', 'side'))
processed_dir = os.path.normpath(os.path.join(scripts_dir, '..', 'data', 'processed'))
os.makedirs(processed_dir, exist_ok=True)
output_path   = os.path.join(processed_dir, 'dataset.npz')

# ── Load all session CSV files ────────────────────────────────────────────────
csv_files = sorted(
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.startswith('session_') and f.endswith('.csv')
)

if not csv_files:
    print(f"ERROR: No session CSV files found in {data_dir}")
    exit()

print(f"Found {len(csv_files)} session files:")
for f in csv_files:
    print(f"  {os.path.basename(f)}")
print()

# ── Process each session ──────────────────────────────────────────────────────
all_X = []
all_y = []

for csv_path in csv_files:
    session_name = os.path.basename(csv_path)
    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        print(f"  SKIP {session_name} — no label column")
        continue
    if len(df) < WINDOW_SIZE:
        print(f"  SKIP {session_name} — too short ({len(df)} frames)")
        continue

    # ── Smooth each feature column ────────────────────────────────────────────
    smoothed = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        smoothed[col] = savgol_filter(
            df[col].values,
            window_length=SMOOTH_WINDOW,
            polyorder=SMOOTH_ORDER
        )

    # ── Normalize each feature to 0-1 within this session ────────────────────
    col_min   = smoothed.min()
    col_max   = smoothed.max()
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normalized = (smoothed - col_min) / col_range

    features = normalized.values    # (num_frames, 3)
    labels   = df['label'].values   # (num_frames,)

    # ── Slide window ──────────────────────────────────────────────────────────
    session_windows = 0
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        window_features = features[start:end]
        window_labels   = labels[start:end]

        majority_label = Counter(window_labels).most_common(1)[0][0]
        if majority_label not in LABEL_MAP:
            continue

        all_X.append(window_features)
        all_y.append(LABEL_MAP[majority_label])
        session_windows += 1

    print(f"  {session_name}  {len(df):>5} frames  ->  {session_windows} windows")

# ── Combine and save ──────────────────────────────────────────────────────────
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.int64)

print()
print(f"Dataset shape:  X={X.shape}  y={y.shape}")
print(f"Features:       {X.shape[2]}  (shoulder_flow, chest_flow, belly_flow)")
print()

label_names = {v: k for k, v in LABEL_MAP.items()}
print("Label distribution:")
for label_id, count in sorted(Counter(y.tolist()).items()):
    pct = count / len(y) * 100
    print(f"  {label_names[label_id]:<12}  {count:>5} windows  ({pct:.1f}%)")

np.savez_compressed(output_path, X=X, y=y)
print()
print(f"Saved to: {output_path}")
