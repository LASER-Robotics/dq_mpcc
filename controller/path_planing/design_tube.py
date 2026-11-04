"""
design_tube.py — Visualise and save the variable tube profile d_tube(θ)
for Romero-style MPCC with gate-aware contouring constraint.

Design:
  d_tube(θ) = d_gate  if min_i |θ - θ_gate_i_k| < eps_near
             d_wide  if min_i |θ - θ_gate_i_k| > eps_far
             smoothstep between

Here θ_gate_i_k = theta_gate_first[i] + k·s_lap for k = 0..N_lap (and partially
for entry/exit too, using the same pattern).

Produces:
  - figure           path_planing/tube_profile.png
  - numpy array      path_planing/d_tube_wp.npy  (N_WP,)
  - JSON augment     appends "d_tube" to ref_flat_waypoints.json
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_JSON   = os.path.join(SCRIPT_DIR, "ref_flat_waypoints.json")

# Tube design parameters (two-zone smoothstep)
# Validated in MiL + SiL. The plateau at d_pinch around each gate gives
# the solver a "quiet" region that MuJoCo's first-order actuator can
# handle; a pure Gaussian alternative was tested but destabilised the
# closed loop under model mismatch with the inner rate loop.
D_PINCH      = 0.40   # [m] at each gate (plateau)  — racing exploration room
D_WIDE       = 0.70   # [m] between gates (plateau)
R_PINCH      = 0.4    # [m] within → fully pinched
R_WIDE_START = 0.7    # [m] beyond → fully wide (C¹ smoothstep ramp)

# Wall-aware clipping (HW cage interior, walls 0.05m discounted)
WS_X = (-2.95, 7.95)
WS_Y = (-1.95, 1.95)
WS_Z = ( 0.00, 3.00)
R_DRONE = 0.20        # [m] drone tip-radius (6" frame)

with open(REF_JSON) as f:
    ref = json.load(f)

s_wp       = np.asarray(ref["s"])                   # (N,)
s_ana_s    = float(ref["s_analysis_start"])
s_ana_e    = float(ref["s_analysis_end"])
ref_pos_wp = np.asarray(ref["pos"])                 # (N, 3)
gate_pos   = np.asarray(ref["gate_pos"])            # (Ng, 3)
Ng         = gate_pos.shape[0]

# Geometric detection: for each gate, find ALL local minima of |path(s) - gate_i|.
# Robust to multilap topology (entry + N×periodic + warmup_out + exit), no
# dependency on theta_gate_first or s_lap (which fail to capture entry/exit/warmup).
# Local minimum = waypoint where d_path_to_gate is smaller than both neighbors AND
# below threshold (0.5 m, well within R_gate=50cm + path-near-gate band).
NEAR_THRESH = 0.5  # [m] only count s where path approaches within 0.5m of gate
theta_gates_all = []
for i in range(Ng):
    d = np.linalg.norm(ref_pos_wp - gate_pos[i], axis=1)
    # local min: d[k-1] > d[k] < d[k+1]
    is_min = (d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < NEAR_THRESH)
    idxs = np.where(is_min)[0] + 1
    theta_gates_all.extend(s_wp[idxs].tolist())
theta_gates_all = np.asarray(sorted(theta_gates_all))
print(f"Geometric detection: {len(theta_gates_all)} gate crossings found "
      f"({len(theta_gates_all)/Ng:.1f} per gate avg, expected ≈ 5-6)")

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x * x * (3.0 - 2.0 * x)

def d_tube_fn(θ_vec):
    """Two-zone smoothstep: plateau at d_pinch near each gate, plateau at
    d_wide far from gates, C¹ smooth ramp in between."""
    g = np.min(np.abs(θ_vec[:, None] - theta_gates_all[None, :]), axis=1)
    out = np.full_like(g, D_WIDE)
    mA = g < R_PINCH
    out[mA] = D_PINCH
    mAB = (g >= R_PINCH) & (g < R_WIDE_START)
    u = (g[mAB] - R_PINCH) / (R_WIDE_START - R_PINCH)
    out[mAB] = D_PINCH + (D_WIDE - D_PINCH) * smoothstep(u)
    return out

d_tube_design = d_tube_fn(s_wp)

# Wall-aware clipping
# At each waypoint, max admissible tube radius so that
# drone tip (path + d_tube + r_drone) stays inside cage interior.
pos_for_clip = np.asarray(ref["pos"]).T   # (3, N)
margin_x = np.minimum(WS_X[1] - pos_for_clip[0], pos_for_clip[0] - WS_X[0])
margin_y = np.minimum(WS_Y[1] - pos_for_clip[1], pos_for_clip[1] - WS_Y[0])
margin_z = np.minimum(WS_Z[1] - pos_for_clip[2], pos_for_clip[2] - WS_Z[0])
wall_margin = np.minimum(np.minimum(margin_x, margin_y), margin_z)
d_tube_max = np.maximum(wall_margin - R_DRONE, D_PINCH)   # never below D_PINCH
d_tube_wp = np.minimum(d_tube_design, d_tube_max)
n_clipped = int(np.sum(d_tube_design > d_tube_max))
print(f"Wall-aware clip: {n_clipped}/{len(s_wp)} waypoints clipped  "
      f"(min wall_margin={wall_margin.min():.2f}m at p={pos_for_clip[:, wall_margin.argmin()].round(2)})")

# Plot
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(s_wp, d_tube_wp * 100, color="C0", lw=1.5, label="d_tube(θ)")
ax.axhline(D_PINCH    * 100, color="C1", ls="--", lw=0.8, label=f"d_pinch={D_PINCH*100:.0f}cm")
ax.axhline(D_WIDE     * 100, color="C2", ls="--", lw=0.8, label=f"d_wide={D_WIDE*100:.0f}cm")
ax.axhline(30,               color="C3", ls=":",  lw=0.8, label="R_safe=30cm")
ax.axhline(50,               color="k",  ls=":",  lw=0.8, label="R_gate=50cm")
for θg in theta_gates_all:
    ax.axvline(θg, color="r", lw=0.3, alpha=0.6)
ax.axvspan(s_ana_s, s_ana_e, color="yellow", alpha=0.12, label="analysis (lap 2+3)")
ax.set_xlabel("θ [m]")
ax.set_ylabel("d_tube [cm]")
ax.set_title(f"Two-zone tube  pinch={D_PINCH*100:.0f}cm  wide={D_WIDE*100:.0f}cm   "
             f"R_pinch={R_PINCH:.1f} R_wide_start={R_WIDE_START:.1f}")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, s_wp[-1])
fig.tight_layout()
_fig_dir = os.path.join(SCRIPT_DIR, "figures", "tube")
os.makedirs(_fig_dir, exist_ok=True)
out_png = os.path.join(_fig_dir, "tube_profile.png")
fig.savefig(out_png, dpi=130)
plt.close(fig)

# Envelope plot: tube along path (XY + XZ)
import matplotlib.patches as mpatches
pos_wp = np.asarray(ref["pos"]).T       # (3, N)
tang_wp = np.asarray(ref["tang"]).T     # (3, N) unit tangent

# PMM trajectories (raw npy) for visual overlay (centered + free)
try:
    pmm_cen = np.load(os.path.join(SCRIPT_DIR, "xref_optimo_3D_PMM_periodic.npy"))[:3]
except FileNotFoundError:
    pmm_cen = None
try:
    pmm_fr  = np.load(os.path.join(SCRIPT_DIR, "xref_optimo_3D_PMM_periodic_free.npy"))[:3]
except FileNotFoundError:
    pmm_fr = None
gp = np.asarray(ref["gate_pos"])        # (Ng, 3)
gn = np.asarray(ref["gate_normal"])
Ng_real = gp.shape[0]
R_gate = float(ref["gate_radius"])
R_safe = float(ref["gate_safe_radius"])

# Cage walls (HW interior)
WS_X = (-2.95, 7.95); WS_Y = (-1.95, 1.95); WS_Z = (0.0, 3.0)

# Build tube envelope vertices in XY/XZ planes:
# at each waypoint, plot path point ± d_tube perpendicular to path tangent
def envelope_2d(pos, tang, d, axis_a, axis_b):
    """Return (x_outer, y_outer, x_inner, y_inner) for filled band in axes a,b."""
    pa, pb = pos[axis_a], pos[axis_b]
    ta, tb = tang[axis_a], tang[axis_b]
    norm_t = np.sqrt(ta**2 + tb**2) + 1e-9
    # Perpendicular in 2D plane (rotate 90°): (−tb, ta) / norm
    perp_a = -tb / norm_t
    perp_b = ta  / norm_t
    return pa + d*perp_a, pb + d*perp_b, pa - d*perp_a, pb - d*perp_b

theta_circle = np.linspace(0, 2*np.pi, 60)
def gate_disk_2d(center, normal, r, axis_a, axis_b):
    ref_v = np.array([0.,0.,1.]) if abs(normal[2])<0.9 else np.array([1.,0.,0.])
    t1 = np.cross(normal, ref_v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    pts = np.array([center + r*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta_circle])
    return pts[:, axis_a], pts[:, axis_b]

fig, (axXY, axXZ) = plt.subplots(1, 2, figsize=(18, 7))
for view, ax, (a, b), (wx, wy) in [
        ('XY', axXY, (0,1), (WS_X, WS_Y)),
        ('XZ', axXZ, (0,2), (WS_X, WS_Z))]:
    # Cage walls
    ax.add_patch(mpatches.Rectangle((wx[0], wy[0]), wx[1]-wx[0], wy[1]-wy[0],
                  fill=False, edgecolor='gray', lw=1.4, alpha=0.8))
    # Tube band only inside analysis window (laps), not entry/exit
    s_a_start = float(ref["s_analysis_start"])
    s_a_end   = float(ref["s_analysis_end"])
    mask = (s_wp >= s_a_start) & (s_wp <= s_a_end)
    pos_an  = pos_wp[:,  mask]
    tang_an = tang_wp[:, mask]
    d_an    = d_tube_wp[mask]
    xo, yo, xi, yi = envelope_2d(pos_an, tang_an, d_an, a, b)
    band_x = np.concatenate([xo, xi[::-1]])
    band_y = np.concatenate([yo, yi[::-1]])
    ax.fill(band_x, band_y, color='C0', alpha=0.20, label='tube envelope')
    # PMM trajectories (overlay) — centered (used as OCP ref) + free (theoretical)
    if pmm_cen is not None:
        ax.plot(pmm_cen[a], pmm_cen[b], color='black', lw=1.4, alpha=0.85,
                label='PMM centered (OCP ref)' if view == 'XY' else None)
    if pmm_fr is not None:
        ax.plot(pmm_fr[a], pmm_fr[b], color='red', lw=1.0, ls='--', alpha=0.75,
                label='PMM free (racing line bound)' if view == 'XY' else None)
    # Multilap path centerline (entry+laps+exit)
    ax.plot(pos_wp[a], pos_wp[b], color='C0', lw=0.9, alpha=0.6,
            label='multilap ref' if view == 'XY' else None)
    # Gates
    for gi in range(Ng_real):
        xg, yg = gate_disk_2d(gp[gi], gn[gi], R_gate, a, b)
        ax.plot(xg, yg, color='limegreen', lw=1.6, alpha=0.9)
        xs, ys = gate_disk_2d(gp[gi], gn[gi], R_safe, a, b)
        ax.plot(xs, ys, color='darkorange', lw=0.8, ls='--', alpha=0.7)
        ax.scatter(gp[gi, a], gp[gi, b], color='red', s=22, zorder=5)
        ax.text(gp[gi, a], gp[gi, b]+0.15, f'G{gi}', fontsize=9, ha='center', fontweight='bold')
    ax.set_xlabel(f"{view[0].lower()} [m]"); ax.set_ylabel(f"{view[1].lower()} [m]")
    ax.set_title(f"{view} — tube envelope (d_tube ∈ [{D_PINCH*100:.0f},{D_WIDE*100:.0f}]cm)")
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlim(wx[0]-0.3, wx[1]+0.3); ax.set_ylim(wy[0]-0.3, wy[1]+0.3)
fig.suptitle('Tube envelope along multilap path  (band: ‖e_c‖ ≤ d_tube(θ))',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
handles, labels = axXY.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=len(labels), fontsize=9, frameon=True)
fig.subplots_adjust(top=0.88)
out_env = os.path.join(_fig_dir, "tube_envelope.png")
fig.savefig(out_env, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"Wrote {out_env}")
print(f"Wrote {out_png}")

# Save LUT
np.save(os.path.join(SCRIPT_DIR, "d_tube_wp.npy"), d_tube_wp)
print(f"d_tube_wp: shape={d_tube_wp.shape}  min={d_tube_wp.min()*100:.1f}cm  "
      f"max={d_tube_wp.max()*100:.1f}cm  mean={d_tube_wp.mean()*100:.1f}cm")

# Also append into the JSON so the MiL/SiL loader can access it if needed
ref["d_tube"]      = d_tube_wp.tolist()
ref["d_gate"]      = D_PINCH
ref["d_pinch"]     = D_PINCH
ref["d_wide"]      = D_WIDE
ref["R_pinch"]     = R_PINCH
ref["R_wide_start"]= R_WIDE_START
with open(REF_JSON, "w") as f:
    json.dump(ref, f)
print(f"Updated {REF_JSON} with d_tube array")
