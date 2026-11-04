#!/usr/bin/env python3
"""Plot the final tuning Pareto fronts (lap time vs. worst gate-crossing offset).

Selected (deployed) configurations are highlighted; the dashed line marks
R_safe = 0.30 m. Usage: python3 scripts/reproduce_pareto_figure.py [output.pdf]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tuning_fronts.pdf")
R_SAFE = 0.30

rows = list(csv.DictReader(open(ROOT / "tuning" / "pareto_fronts.csv")))
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for ctrl, color in (("MPCC", "tab:blue"), ("DQ-MPCC", "tab:red")):
    pts = sorted((float(r["lap_time_min_s"]), 100 * float(r["worst_gate_offset_m"]), r["selected"] == "true")
                 for r in rows if r["controller"] == ctrl)
    ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=color, lw=1, alpha=0.6)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, color=color, label=ctrl)
    for t, d, sel in pts:
        if sel:
            ax.scatter([t], [d], s=70, facecolors="none", edgecolors="k", lw=1.2)
ax.axhline(100 * R_SAFE, ls="--", color="gray", lw=0.8)
ax.set_xlabel("minimum lap time [s]")
ax.set_ylabel("worst gate offset [cm]")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
