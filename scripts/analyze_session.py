"""
Analyze a session CSV from predict_live.py

Run with:
  py -3.12 scripts/analyze_session.py
  py -3.12 scripts/analyze_session.py logs/session_20260311_191913.csv
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Find CSV ──────────────────────────────────────────────────────────────────
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    files = sorted(glob.glob(os.path.join(logs_dir, 'session_*.csv')))
    if not files:
        print("No session CSVs found in logs/")
        sys.exit(1)
    csv_path = files[-1]   # most recent
    print(f"Loading: {csv_path}\n")

df = pd.read_csv(csv_path)

# ── Summary ───────────────────────────────────────────────────────────────────
total_s  = df['time_s'].max() - df['time_s'].min()
fps      = len(df) / total_s if total_s > 0 else 0
counts   = df['label'].value_counts()
durations = df.groupby('label')['time_s'].count() / fps

print(f"Duration : {total_s:.1f}s   |   Frames: {len(df)}   |   ~{fps:.0f} fps")
print()
print("Label breakdown:")
for label in ['inhale', 'exhale', 'hold_in', 'hold_out']:
    n   = counts.get(label, 0)
    sec = n / fps if fps > 0 else 0
    pct = 100 * n / len(df) if len(df) > 0 else 0
    print(f"  {label:<12}  {sec:5.1f}s  ({pct:.0f}%)")

# Detect phase transitions
transitions = df[df['label'] != df['label'].shift()]
print(f"\nPhase transitions: {len(transitions)}")
print(transitions[['time_s', 'label']].to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    'inhale':   '#00ff64',
    'exhale':   '#0064ff',
    'hold_in':  '#ffc800',
    'hold_out': '#c800ff',
}

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(os.path.basename(csv_path), fontsize=11)

# Top: probability lines
ax = axes[0]
for col in ['inhale', 'exhale', 'hold_in', 'hold_out']:
    ax.plot(df['time_s'], df[col], label=col, color=COLORS[col], linewidth=1.2)
ax.set_ylabel('Probability')
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.2)

# Bottom: label as colored background bands
ax2 = axes[1]
ax2.set_ylabel('Predicted Label')
ax2.set_yticks([])
ax2.set_xlabel('Time (s)')

prev_t     = df['time_s'].iloc[0]
prev_label = df['label'].iloc[0]
for _, row in df.iterrows():
    if row['label'] != prev_label:
        ax2.axvspan(prev_t, row['time_s'], color=COLORS.get(prev_label, '#888888'), alpha=0.6)
        prev_t, prev_label = row['time_s'], row['label']
ax2.axvspan(prev_t, df['time_s'].iloc[-1], color=COLORS.get(prev_label, '#888888'), alpha=0.6)

patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax2.legend(handles=patches, loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
