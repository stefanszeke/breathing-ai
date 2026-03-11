import sys
import pandas as pd
import numpy as np
import os

# Force UTF-8 output on Windows consoles that default to cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SESSION_FILES = [
    r"c:\Users\sekes\Desktop\breathing-ai\data\side\session_001.csv",
    r"c:\Users\sekes\Desktop\breathing-ai\data\side\session_002.csv",
    r"c:\Users\sekes\Desktop\breathing-ai\data\side\session_003.csv",
    r"c:\Users\sekes\Desktop\breathing-ai\data\side\session_004.csv",
    r"c:\Users\sekes\Desktop\breathing-ai\data\side\session_005.csv",
]

WINDOW = 60
STEP = 5

# ---- Load sessions -----------------------------------------------------------
all_dfs = []
for path in SESSION_FILES:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue
    df = pd.read_csv(path)
    if df.shape[0] < 2:
        print(f"  SKIP (empty/header-only): {path}")
        continue
    required = {"frame", "chest_flow", "label"}
    if not required.issubset(df.columns):
        print(f"  SKIP (missing columns): {path}")
        continue
    name = os.path.basename(path)
    print(f"  Loaded {name}  ({len(df)} rows)  labels: {df['label'].value_counts().to_dict()}")
    all_dfs.append(df)

if not all_dfs:
    raise SystemExit("No usable session files found.")

# ---- Frame counts per label across all sessions ------------------------------
combined = pd.concat(all_dfs, ignore_index=True)
print("\n=== Total frame count per label (all sessions) ===")
label_counts = combined["label"].value_counts().sort_index()
for lbl, cnt in label_counts.items():
    print(f"  {lbl:12s}: {cnt:6d} frames")
print(f"  {'TOTAL':12s}: {len(combined):6d} frames")

# ---- Rolling-window feature extraction ---------------------------------------
from collections import Counter

def majority_label(labels):
    return Counter(labels).most_common(1)[0][0]

rows = []
for df in all_dfs:
    cf = df["chest_flow"].values
    labels = df["label"].values
    n = len(df)
    for start in range(0, n - WINDOW + 1, STEP):
        end = start + WINDOW
        window_cf = cf[start:end]
        window_lbl = labels[start:end]
        std_cf = np.std(window_cf)
        mean_abs_cf = np.mean(np.abs(window_cf))
        maj_lbl = majority_label(window_lbl)
        rows.append({"std_cf": std_cf, "mean_abs_cf": mean_abs_cf, "label": maj_lbl})

windows = pd.DataFrame(rows)
print(f"\n=== Windows extracted: {len(windows)}  (window={WINDOW}, step={STEP}) ===")
print(f"  Window label counts: {windows['label'].value_counts().to_dict()}")

# ---- Per-label percentile stats ----------------------------------------------
PERCS = [5, 25, 50, 75, 95]
LABELS_ORDER = ["inhale", "exhale", "hold_in", "hold_out"]

def pct_row(series):
    return {
        "mean": series.mean(),
        **{f"p{p}": np.percentile(series, p) for p in PERCS},
    }

header = f"  {'label':12s}  {'mean':8s}  {'p5':8s}  {'p25':8s}  {'p50':8s}  {'p75':8s}  {'p95':8s}  {'n':>6s}"

print("\n=== chest_flow STD per label ===")
print(header)
for lbl in LABELS_ORDER:
    sub = windows[windows["label"] == lbl]["std_cf"]
    if len(sub) == 0:
        print(f"  {lbl:12s}  (no data)")
        continue
    s = pct_row(sub)
    print(f"  {lbl:12s}  {s['mean']:8.5f}  {s['p5']:8.5f}  {s['p25']:8.5f}  {s['p50']:8.5f}  {s['p75']:8.5f}  {s['p95']:8.5f}  {len(sub):>6d}")

s = pct_row(windows["std_cf"])
print(f"  {'ALL':12s}  {s['mean']:8.5f}  {s['p5']:8.5f}  {s['p25']:8.5f}  {s['p50']:8.5f}  {s['p75']:8.5f}  {s['p95']:8.5f}  {len(windows):>6d}")

print("\n=== mean abs chest_flow per label ===")
print(header)
for lbl in LABELS_ORDER:
    sub = windows[windows["label"] == lbl]["mean_abs_cf"]
    if len(sub) == 0:
        print(f"  {lbl:12s}  (no data)")
        continue
    s = pct_row(sub)
    print(f"  {lbl:12s}  {s['mean']:8.5f}  {s['p5']:8.5f}  {s['p25']:8.5f}  {s['p50']:8.5f}  {s['p75']:8.5f}  {s['p95']:8.5f}  {len(sub):>6d}")

s = pct_row(windows["mean_abs_cf"])
print(f"  {'ALL':12s}  {s['mean']:8.5f}  {s['p5']:8.5f}  {s['p25']:8.5f}  {s['p50']:8.5f}  {s['p75']:8.5f}  {s['p95']:8.5f}  {len(windows):>6d}")

# ---- Best threshold on mean_abs_cf: hold vs active --------------------------
print("\n=== Threshold sweep: mean_abs_cf  (hold_in/hold_out vs inhale/exhale) ===")

windows["is_hold"] = windows["label"].isin(["hold_in", "hold_out"])
y_true = windows["is_hold"].values
mac_vals = windows["mean_abs_cf"].values

candidates = np.unique(mac_vals)
midpoints = (candidates[:-1] + candidates[1:]) / 2
thresholds = np.sort(np.concatenate([candidates, midpoints]))

results = []
for t in thresholds:
    y_pred_hold = mac_vals <= t
    tp = int(np.sum(y_true & y_pred_hold))
    tn = int(np.sum(~y_true & ~y_pred_hold))
    fp = int(np.sum(~y_true & y_pred_hold))
    fn = int(np.sum(y_true & ~y_pred_hold))
    acc = (tp + tn) / len(y_true)
    rec_hold  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_hold = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_hold   = 2 * prec_hold * rec_hold / (prec_hold + rec_hold) if (prec_hold + rec_hold) > 0 else 0.0
    results.append((t, acc, rec_hold, prec_hold, f1_hold, tp, tn, fp, fn))

results_arr = np.array([(r[1], r[4]) for r in results])
best_idx_acc = int(np.argmax(results_arr[:, 0]))
best_idx_f1  = int(np.argmax(results_arr[:, 1]))

def print_result(title, r):
    t, acc, rec, prec, f1, tp, tn, fp, fn = r
    print(f"  {title}")
    print(f"    threshold (mean_abs_cf <= T -> hold): {t:.6f}")
    print(f"    accuracy       : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    hold recall    : {rec:.4f}  ({rec*100:.1f}%)")
    print(f"    hold precision : {prec:.4f}  ({prec*100:.1f}%)")
    print(f"    hold F1        : {f1:.4f}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

print_result("Best by Accuracy", results[best_idx_acc])
print()
print_result("Best by Hold F1", results[best_idx_f1])

# ---- IQR overlap check -------------------------------------------------------
hold_mac   = windows[windows["is_hold"]]["mean_abs_cf"]
active_mac = windows[~windows["is_hold"]]["mean_abs_cf"]
print("\n=== Overlap check ===")
print(f"  Hold    median={hold_mac.median():.5f}  IQR=[{hold_mac.quantile(.25):.5f}, {hold_mac.quantile(.75):.5f}]")
print(f"  Active  median={active_mac.median():.5f}  IQR=[{active_mac.quantile(.25):.5f}, {active_mac.quantile(.75):.5f}]")

overlap_lo = max(hold_mac.quantile(.25), active_mac.quantile(.25))
overlap_hi = min(hold_mac.quantile(.75), active_mac.quantile(.75))
if overlap_lo < overlap_hi:
    print(f"  IQR overlap region: [{overlap_lo:.5f}, {overlap_hi:.5f}]  (some overlap exists)")
else:
    print("  IQR boxes do NOT overlap  (clean separation in IQR range)")

print("\nDone.")
