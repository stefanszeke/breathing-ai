import pandas as pd
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────
data_dir = Path("c:/Users/sekes/Desktop/breathing-ai/data/side")
sessions = {}
for i in range(1, 5):
    path = data_dir / f"session_{i:03d}.csv"
    df = pd.read_csv(path)
    if len(df) > 0:
        df["session"] = i
        sessions[i] = df
        print(f"session_{i:03d}: {len(df)} rows")
    else:
        print(f"session_{i:03d}: EMPTY (skipped)")

if not sessions:
    print("No data found in any session.")
    raise SystemExit(1)

all_data = pd.concat(sessions.values(), ignore_index=True)

# ── Frame counts per label per session ────────────────────────────────────────
print("\n=== FRAME COUNTS PER LABEL PER SESSION ===")
pivot = all_data.groupby(["session", "label"]).size().unstack(fill_value=0)
print(pivot.to_string())
print(f"\nTotal across all sessions:")
print(all_data["label"].value_counts().sort_index().to_string())

# ── Rolling-window feature extraction ─────────────────────────────────────────
WINDOW = 60
STEP   = 5

records = []
for sid, df in sessions.items():
    df = df.reset_index(drop=True)
    chest = df["chest_flow"].values
    labels = df["label"].values
    n = len(df)
    for start in range(0, n - WINDOW + 1, STEP):
        end = start + WINDOW
        window_chest  = chest[start:end]
        window_labels = labels[start:end]

        std_cf  = np.std(window_chest)
        mean_abs_cf = np.mean(np.abs(window_chest))

        # majority label in window
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        records.append({
            "session":      sid,
            "start":        start,
            "std_cf":       std_cf,
            "mean_abs_cf":  mean_abs_cf,
            "label":        majority_label,
        })

windows = pd.DataFrame(records)

print(f"\n=== ROLLING WINDOWS GENERATED ===")
print(f"Total windows: {len(windows)}")
print(windows["label"].value_counts().sort_index().to_string())

# ── Per-label statistics ───────────────────────────────────────────────────────
PERCS = [5, 25, 50, 75, 95]

def label_stats(df, col):
    rows = []
    for lbl in sorted(df["label"].unique()):
        sub = df.loc[df["label"] == lbl, col]
        row = {"label": lbl, "n": len(sub), "mean": sub.mean()}
        for p in PERCS:
            row[f"p{p}"] = np.percentile(sub, p)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")

print("\n=== CHEST_FLOW STD — per-label percentiles ===")
print(label_stats(windows, "std_cf").round(5).to_string())

print("\n=== MEAN ABS CHEST_FLOW — per-label percentiles ===")
print(label_stats(windows, "mean_abs_cf").round(5).to_string())

# ── Overlap analysis ──────────────────────────────────────────────────────────
hold_labels   = {"hold_in", "hold_out"}
active_labels = {"inhale", "exhale"}

holds   = windows[windows["label"].isin(hold_labels)]
actives = windows[windows["label"].isin(active_labels)]

print("\n=== OVERLAP BETWEEN HOLD vs ACTIVE (chest_flow STD) ===")
for col, name in [("std_cf", "std"), ("mean_abs_cf", "mean_abs")]:
    hold_vals   = holds[col].values
    active_vals = actives[col].values
    print(f"\n  [{name}]")
    print(f"    Hold   range: [{hold_vals.min():.5f}, {hold_vals.max():.5f}]  "
          f"median={np.median(hold_vals):.5f}")
    print(f"    Active range: [{active_vals.min():.5f}, {active_vals.max():.5f}]  "
          f"median={np.median(active_vals):.5f}")
    overlap_lo = max(hold_vals.min(), active_vals.min())
    overlap_hi = min(hold_vals.max(), active_vals.max())
    if overlap_lo < overlap_hi:
        print(f"    Overlap region: [{overlap_lo:.5f}, {overlap_hi:.5f}]")
    else:
        print(f"    No overlap! (clean separation)")

# ── Threshold sweep on std_cf ─────────────────────────────────────────────────
print("\n=== THRESHOLD SWEEP on chest_flow STD ===")
print("(Predict HOLD if std_cf <= threshold, else ACTIVE)")

is_hold_true = windows["label"].isin(hold_labels).values
std_vals     = windows["std_cf"].values

thresholds = np.linspace(std_vals.min(), std_vals.max(), 200)
best_acc   = -1
best_thresh = None
best_tp = best_tn = best_fp = best_fn = 0

results = []
for t in thresholds:
    pred_hold = std_vals <= t
    tp = int(np.sum( pred_hold &  is_hold_true))
    tn = int(np.sum(~pred_hold & ~is_hold_true))
    fp = int(np.sum( pred_hold & ~is_hold_true))
    fn = int(np.sum(~pred_hold &  is_hold_true))
    acc = (tp + tn) / len(windows)
    results.append((t, tp, tn, fp, fn, acc))
    if acc > best_acc:
        best_acc   = acc
        best_thresh = t
        best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

# Print a sparse table (every 10th entry) around the best threshold
results_df = pd.DataFrame(results, columns=["threshold", "TP", "TN", "FP", "FN", "accuracy"])
# Show rows near best threshold
mask = (results_df["threshold"] >= best_thresh - 0.01) & \
       (results_df["threshold"] <= best_thresh + 0.01)
print("\nRows near best threshold:")
print(results_df[mask].round(5).to_string(index=False))

print(f"\n--- BEST THRESHOLD (std_cf) ---")
print(f"  Threshold : {best_thresh:.5f}")
print(f"  Accuracy  : {best_acc:.4f}  ({best_acc*100:.1f}%)")
print(f"  TP (hold correctly detected)  : {best_tp}")
print(f"  TN (active correctly detected): {best_tn}")
print(f"  FP (active wrongly called hold): {best_fp}")
print(f"  FN (hold wrongly called active): {best_fn}")
n_holds  = int(is_hold_true.sum())
n_active = int((~is_hold_true).sum())
print(f"  Hold recall   : {best_tp}/{n_holds}  = {best_tp/max(n_holds,1):.3f}")
print(f"  Active recall : {best_tn}/{n_active} = {best_tn/max(n_active,1):.3f}")

# ── Same sweep on mean_abs_cf ──────────────────────────────────────────────────
print("\n=== THRESHOLD SWEEP on mean_abs chest_flow ===")
abs_vals = windows["mean_abs_cf"].values
thresholds2 = np.linspace(abs_vals.min(), abs_vals.max(), 200)
best_acc2 = -1; best_thresh2 = None
best_tp2 = best_tn2 = best_fp2 = best_fn2 = 0

for t in thresholds2:
    pred_hold = abs_vals <= t
    tp = int(np.sum( pred_hold &  is_hold_true))
    tn = int(np.sum(~pred_hold & ~is_hold_true))
    fp = int(np.sum( pred_hold & ~is_hold_true))
    fn = int(np.sum(~pred_hold &  is_hold_true))
    acc = (tp + tn) / len(windows)
    if acc > best_acc2:
        best_acc2 = acc; best_thresh2 = t
        best_tp2, best_tn2, best_fp2, best_fn2 = tp, tn, fp, fn

print(f"\n--- BEST THRESHOLD (mean_abs_cf) ---")
print(f"  Threshold : {best_thresh2:.5f}")
print(f"  Accuracy  : {best_acc2:.4f}  ({best_acc2*100:.1f}%)")
print(f"  TP : {best_tp2}  TN : {best_tn2}  FP : {best_fp2}  FN : {best_fn2}")
print(f"  Hold recall   : {best_tp2}/{n_holds}  = {best_tp2/max(n_holds,1):.3f}")
print(f"  Active recall : {best_tn2}/{n_active} = {best_tn2/max(n_active,1):.3f}")

# ── Summary recommendation ─────────────────────────────────────────────────────
print("\n=== SUMMARY RECOMMENDATION ===")
better = "std_cf" if best_acc >= best_acc2 else "mean_abs_cf"
bt = best_thresh if best_acc >= best_acc2 else best_thresh2
ba = max(best_acc, best_acc2)
print(f"  Best feature  : {better}")
print(f"  Best threshold: {bt:.5f}")
print(f"  Best accuracy : {ba*100:.1f}%")
feasibility = "YES — rule-based hold detection looks feasible." if ba >= 0.80 \
              else "MARGINAL — rule-based threshold may be insufficient." if ba >= 0.65 \
              else "NO — too much overlap; a learned model is needed."
print(f"  Feasibility   : {feasibility}")
