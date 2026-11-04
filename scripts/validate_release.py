#!/usr/bin/env python3
"""Consistency checks for the DQ-MPCC release data.

Checks that the Pareto fronts are non-dominated, that exactly one configuration
per controller is selected, that the selected rows match the deployed JSON
configurations, and that the controller copies of those JSON files are identical.
Exit code 0 means all checks passed.
"""
import csv
import filecmp
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-4  # relative tolerance for 6-significant-digit CSV values

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


front = list(csv.DictReader(open(ROOT / "tuning" / "pareto_fronts.csv")))
selected = list(csv.DictReader(open(ROOT / "tuning" / "selected_gains.csv")))

counts = {c: sum(r["controller"] == c for r in front) for c in ("MPCC", "DQ-MPCC")}
check(counts == {"MPCC": 12, "DQ-MPCC": 11}, f"front sizes {counts} == 12 MPCC / 11 DQ-MPCC")

for c in counts:
    pts = [(float(r["lap_time_min_s"]), float(r["worst_gate_offset_m"])) for r in front if r["controller"] == c]
    dominated = [p for p in pts if any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)]
    check(not dominated, f"{c}: no dominated point in the front")
    sel = [r for r in front if r["controller"] == c and r["selected"] == "true"]
    check(len(sel) == 1, f"{c}: exactly one selected configuration")

check({(r["controller"], r["trial"]) for r in selected} == {("MPCC", "76"), ("DQ-MPCC", "46")},
      "selected_gains.csv lists MPCC trial 76 and DQ-MPCC trial 46")
for r in selected:
    match = [f for f in front if f["controller"] == r["controller"] and f["trial"] == r["trial"]]
    check(len(match) == 1 and match[0] == r, f"{r['controller']} trial {r['trial']}: identical row in pareto_fronts.csv")
    check(float(r["worst_gate_offset_m"]) <= 0.30, f"{r['controller']}: worst tuning offset within R_safe = 0.30 m")
    check(float(r["w_thrust_rate"]) == 1e-3, f"{r['controller']}: W_df = 1e-3")

PAIRS = [("q_contour", "Q_ec"), ("q_lag", "Q_el"), ("w_body_rate", "U_w"),
         ("w_progress", "Q_s"), ("v_progress_max_mps", "vtheta_max")]
for ctrl, fname, att in (("MPCC", "mpcc_deployed.json", "Q_q"), ("DQ-MPCC", "dq_mpcc_deployed.json", "Q_phi")):
    entry = json.load(open(ROOT / "config" / fname))["front"][0]
    p, w = entry["params"], entry["weights"]
    r = next(x for x in selected if x["controller"] == ctrl)
    check(str(entry["trial"]) == r["trial"], f"{fname}: trial {entry['trial']} matches CSV")
    ok = close(float(r["q_attitude"]), p[att]) and all(close(float(r[c]), p[k]) for c, k in PAIRS)
    check(ok, f"{fname}: parameters match selected_gains.csv")
    check(all(close(v, p[att]) for v in w[att]) and all(close(v, p["U_w"]) for v in w["U_mat"][1:]),
          f"{fname}: isotropic weight vectors consistent with parameters")
    check(close(w["W_df"], 1e-3), f"{fname}: W_df = 1e-3")

for src, dst in (("mpcc_deployed.json", "pareto_mpcc_manual.json"),
                 ("dq_mpcc_deployed.json", "pareto_dq_mpcc_ablation_manual.json")):
    check(filecmp.cmp(ROOT / "config" / src, ROOT / "controller" / "tuning" / "sil" / "manual" / dst, shallow=False),
          f"config/{src} identical to controller/tuning/sil/manual/{dst}")

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
