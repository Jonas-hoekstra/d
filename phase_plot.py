"""
Phase-folded light curve of SZ Lyncis.
Reads all lightcurve_*.csv files in the current directory — no pipeline rerun needed.

Usage:
    python phase_plot.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── settings ──────────────────────────────────────────────────────────────────
PERIOD   = 0.120534896   # days — SZ Lyncis pulsation period
T0       = 2461118.405   # HJD of maximum light (20260316, SZLYN-2 frame 8)
MIN_PTS  = 3
OUT_FILE = "phase_plot.png"
# ──────────────────────────────────────────────────────────────────────────────

FILTER_COLORS = {"r": "tomato", "g": "steelblue"}

HERE     = Path(__file__).parent
CSV_DIR  = HERE / "csv"
ZOOM_DIR = HERE / "csv" / "zoomed_pictures"

root_files = sorted(CSV_DIR.glob("lightcurve_*.csv"))
if not root_files:
    raise FileNotFoundError(f"No lightcurve_*.csv files found in '{CSV_DIR}'.")

csv_files = []
for root in root_files:
    if ZOOM_DIR.exists():
        zoomed = sorted(ZOOM_DIR.glob(f"{root.stem}_zoom*.csv"))
    else:
        zoomed = []

    if zoomed:
        best_zoom = zoomed[-1]
        csv_files.append(best_zoom)
        print(f"  {root.name} → using zoomed: {best_zoom.name}")
    else:
        csv_files.append(root)

fig, ax = plt.subplots(figsize=(14, 6))
plotted    = 0
all_phases = []

for path in csv_files:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < MIN_PTS:
        print(f"  {path.name}: {len(rows)} points — skipped")
        continue

    hjds = np.array([float(row["HJD"])      for row in rows])
    mags = np.array([float(row["diff_mag"]) for row in rows])
    # per-point Poisson error on ref-check, also used for diff_mag
    if rows[0].get("mag_err", "") != "":
        errs = np.array([float(row["mag_err"]) for row in rows])
    else:
        ref_mags = np.array([float(row["diff_mag_ref"]) for row in rows])
        errs = np.full(len(mags), float(np.std(ref_mags)))

    phases = ((hjds - T0) / PERIOD) % 1

    # Duplicate data shifted by 1 to show 0–2 phase range
    phases_ext = np.concatenate([phases, phases + 1])
    mags_ext   = np.concatenate([mags,   mags])
    errs_ext   = np.concatenate([errs,   errs])
    sort_idx   = np.argsort(phases_ext)
    phases_ext = phases_ext[sort_idx]
    mags_ext   = mags_ext[sort_idx]
    errs_ext   = errs_ext[sort_idx]

    # Build phases and mags for the line, inserting NaN at phase wraps
    # (jumps > 0.5) so the connecting line never shoots horizontally across the plot.
    ph_line  = []
    mag_line = []
    for i in range(len(phases_ext)):
        ph_line.append(phases_ext[i])
        mag_line.append(mags_ext[i])
        if i < len(phases_ext) - 1 and abs(phases_ext[i + 1] - phases_ext[i]) > 0.5:
            ph_line.append(np.nan)
            mag_line.append(np.nan)
    ph_line  = np.array(ph_line)
    mag_line = np.array(mag_line)

    # Extract filter letter and label from the filename
    stem_clean = path.stem.split("_zoom")[0]
    parts = stem_clean.split("_")

    last_part = parts[-1]
    if last_part in ("r", "g"):
        filt = last_part
    else:
        filt = "?"

    label = "_".join(parts[1:])

    if filt in FILTER_COLORS:
        color = FILTER_COLORS[filt]
    else:
        color = "gray"

    if filt == "r":
        marker = "o"
    else:
        marker = "s"

    ax.errorbar(phases_ext, mags_ext, yerr=errs_ext, fmt='none',
                ecolor=color, elinewidth=0.8, capsize=2, zorder=2)
    ax.scatter(phases_ext, mags_ext, s=25, color=color, marker=marker, zorder=3, label=label)
    ax.plot(ph_line, mag_line, color=color, linewidth=1.0, alpha=0.7)
    all_phases.extend(phases_ext.tolist())
    plotted += 1
    print(f"  {path.name}: {len(rows)} pts, phase {phases.min():.4f}–{phases.max():.4f}")

ax.set_xlim(0, 2)
ax.invert_yaxis()
ax.set_xlabel(f"Phase  (P = {PERIOD} d,  T₀ = {T0})")
ax.set_ylabel("diff_mag  (SZ Lyn Ref)")
ax.set_title("SZ Lyncis — phase-folded light curve")
ax.legend(fontsize=7, loc="lower center", ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_FILE}  ({plotted} runs plotted)")
