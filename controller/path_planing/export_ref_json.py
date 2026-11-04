"""
export_ref_json.py — Convert ref_flat_waypoints.npz to a compact JSON
readable by the C++ side (nlohmann_json).

Schema:
    {
      "s_max": float,
      "n": int,
      "s":    [N],
      "pos":  [N, 3],
      "tang": [N, 3],
      "quat": [N, 4]   (wxyz Hamilton convention)
    }
"""
import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_NPZ    = os.path.join(SCRIPT_DIR, "ref_flat_waypoints.npz")
IN_META   = os.path.join(SCRIPT_DIR, "multilap_metadata.npz")
IN_GATES  = os.path.join(SCRIPT_DIR, "gates.npz")
OUT_JSON  = os.path.join(SCRIPT_DIR, "ref_flat_waypoints.json")

d  = np.load(IN_NPZ,   allow_pickle=True)
m  = np.load(IN_META,  allow_pickle=True)
gd = np.load(IN_GATES, allow_pickle=True)

s    = d["s_wp"].astype(float)        # (N,)
pos  = d["pos_wp"].astype(float).T    # (N,3)
tang = d["tang_wp"].astype(float).T   # (N,3)
quat = d["quat_wp"].astype(float).T   # (N,4) wxyz

# Ensure unit quaternions and continuity (no sign flip)
quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
for i in range(1, len(quat)):
    if np.dot(quat[i-1], quat[i]) < 0.0:
        quat[i] = -quat[i]

# Derived metadata from the periodic analysis window
s_t   = d["s_t"]                     # (T,) arc-length along time-axis
ka    = int(m["k_analysis_start"])
kb    = int(m["k_analysis_end"])
N_lap = int(m["N_analysis_laps"])
s_ana_start = float(s_t[ka])
s_ana_end   = float(s_t[kb])
s_lap       = (s_ana_end - s_ana_start) / N_lap
v_lap_avg   = s_lap / float(m["T_lap"])
v_peak_G0   = float(np.linalg.norm(m["v_steady_G0"]))

# Gates: positions, normals, and θ of first passage
gate_pos    = gd["gate_positions"].astype(float)      # (Ng, 3)
gate_normal = gd["gate_normals"].astype(float)        # (Ng, 3)
gate_radius = float(gd["gate_radius"][0])             # 0.75 m
drone_r = float(gd["safety_margin"][0])           # 0.40 m
Ng          = gate_pos.shape[0]

# θ_gate_first_i = arc-length where path passes nearest to gate_i,
# searching only up to just after the end of lap 1 so later-lap hits
# don't confuse the first-passage index.
theta_gate_first = np.zeros(Ng)
dmin_gate_first  = np.zeros(Ng)
s_search_end_idx = int(np.searchsorted(s, s_ana_start + 1.1 * s_lap))
for i in range(Ng):
    dist = np.linalg.norm(pos[:s_search_end_idx] - gate_pos[i], axis=1)
    k_best = int(np.argmin(dist))
    theta_gate_first[i] = float(s[k_best])
    dmin_gate_first[i]  = float(dist[k_best])

out = {
    "s_max":            float(d["s_max"][0]),
    "n":                int(s.size),
    "s":                s.tolist(),
    "pos":              pos.tolist(),
    "tang":             tang.tolist(),
    "quat":             quat.tolist(),
    # Metadata for MiL/SiL/HW runs:
    "hover_pos":        [float(x) for x in m["hover_pos"]],
    "s_analysis_start": s_ana_start,   # start of lap-1 [m]
    "s_analysis_end":   s_ana_end,     # end   of lap-3 [m]
    "N_analysis_laps":  N_lap,
    "s_lap":            s_lap,
    "T_lap":            float(m["T_lap"]),
    "v_lap_avg":        v_lap_avg,     # average speed in a lap [m/s]
    "v_peak_G0":        v_peak_G0,     # peak speed at G0 [m/s]  ← v_θ_max target
    "v_steady_G0":      [float(x) for x in m["v_steady_G0"]],
    # Gates (for tube-width design and lap-time metrics):
    "n_gates":          Ng,
    "gate_pos":         gate_pos.tolist(),                 # (Ng, 3)
    "gate_normal":      gate_normal.tolist(),              # (Ng, 3)
    "gate_radius":      gate_radius,                       # [m]
    "gate_safety_margin": drone_r,                     # [m]
    "gate_safe_radius": gate_radius - drone_r,       # [m]
    "theta_gate_first": theta_gate_first.tolist(),         # (Ng,) arc-length of 1st passage
}

with open(OUT_JSON, "w") as f:
    json.dump(out, f)

print(f"Wrote {OUT_JSON}")
print(f"  n={out['n']}  s_max={out['s_max']:.2f} m")
print(f"  hover_pos        = {out['hover_pos']}")
print(f"  s_analysis       = [{s_ana_start:.2f}, {s_ana_end:.2f}] m  ({N_lap} laps)")
print(f"  s_lap            = {s_lap:.2f} m")
print(f"  T_lap            = {out['T_lap']:.3f} s")
print(f"  v_lap_avg        = {v_lap_avg:.3f} m/s")
print(f"  v_peak_G0        = {v_peak_G0:.3f} m/s   ← v_θ_max suggestion")
print(f"\n  {Ng} gates  R={gate_radius:.2f}m  drone={drone_r:.2f}m  safe={gate_radius-drone_r:.2f}m")
order = np.argsort(theta_gate_first)
print(f"  First-passage order by θ:  " +
      " → ".join(f"G{order[i]}({theta_gate_first[order[i]]:.1f}m)" for i in range(Ng)))
