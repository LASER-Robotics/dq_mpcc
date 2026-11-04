"""
path_lissajous.py — Analytical Lissajous reference trajectory for gate navigation.

Trajectory (3D Lissajous figure):
    x(t) = 3.70 * sin(1*w*t) + 2.5
    y(t) = 1.25 * sin(2*w*t)
    z(t) = 0.35 * sin(3*w*t) + 1.5

Peak speed at t=0 (all cosines = 1 simultaneously):
    |v|_max = w * sqrt(5^2 + 2.5^2 + 2.25^2) = w * 6.026

Usage: set V_MAX_TARGET [m/s], script computes w automatically.
Gates are placed at N_GATES equally-spaced arc-length samples; each gate
normal is the normalised velocity at that instant.

Outputs (saved to this directory):
    xref_optimo_3D_PMM.npy   (6, N+1) — [p; v]
    uref_optimo_3D_PMM.npy   (3, N)   — [ax; ay; az]
    tref_optimo_3D_PMM.npy   (N+1,)   — time vector
    gate_config.npz           — gate_positions, gate_normals, gate_radius, …
"""

import os
import sys
import numpy as np
from scipy.io import savemat

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import sys as _sys
for _k in list(_sys.modules.keys()):
    if "mpl_toolkits" in _k:
        del _sys.modules[_k]
import mpl_toolkits
mpl_toolkits.__path__ = [
    p for p in [os.path.expanduser("~/.local/lib/python3.10/site-packages/mpl_toolkits")]
    if os.path.isdir(p)
] or mpl_toolkits.__path__
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETERS — adjust here
# =============================================================================
V_MAX_TARGET = 10.0     # [m/s]  desired peak speed (theoretical max of Lissajous)
N_CANDIDATES = 9        # curvature-weighted candidates; closest to each exclusion point is removed
KAPPA_ALPHA  = 2.0      # curvature weighting strength: w(s) = 1 + α·κ_normalised(s)
CROSSING_PT  = np.array([2.5, 0.0, 1.5])  # Lissajous self-intersection to exclude
# Additional candidate positions to remove (thins out gate clusters in corners).
# Each row is a 3D position; the candidate closest to it is dropped.
# Here we remove the former G0 [4.56, 1.50, 2.22] and G6 [-1.86, -1.71, 1.53]
# so each corner has at most one gate → 6 gates total, with more room for the
# drone to execute racing line between them.
EXTRA_EXCLUDE_PTS = np.zeros((0, 3))   # empty → no extra exclusions (8 gates)
N_LAPS       = 1        # number of complete periods to simulate
DT_REF       = 0.01     # [s]  reference time step (= 1/f_c, 100 Hz)
GATE_RADIUS  = 0.50     # [m]  physical gate radius — 1.0 m aperture, IROS ADR standard
SAFETY_MARGIN = 0.20    # [m]  = r_drone (6-inch frame, tip-to-tip 0.40 m)
USABLE_RADIUS = GATE_RADIUS - SAFETY_MARGIN   # 0.30 m — body-safe deviation (R_safe)

# =============================================================================
# WORKSPACE (HW cage interior, walls 0.05 m thick already discounted)
# =============================================================================
WS_X = (-2.95, 7.95)
WS_Y = (-1.95, 1.95)
WS_Z = ( 0.00, 3.00)

# =============================================================================
# COMPUTE w
# =============================================================================
_speed_scale = np.sqrt(3.70**2 + (2*1.30)**2 + (3*0.35)**2)   # amplitudes × freqs
w = V_MAX_TARGET / _speed_scale
T_period = 2.0 * np.pi / w
T_total  = N_LAPS * T_period

print("=" * 60)
print("LISSAJOUS PATH GENERATOR")
print("=" * 60)
print(f"V_max target  = {V_MAX_TARGET:.2f} m/s")
print(f"w             = {w:.4f} rad/s")
print(f"T_period      = {T_period:.3f} s")
print(f"T_total       = {T_total:.3f} s")
print(f"N_candidates  = {N_CANDIDATES} → N_gates after crossing-exclusion (over {N_LAPS} lap)")

# =============================================================================
# TRAJECTORY FUNCTIONS (analytical)
# =============================================================================
def pos(t):
    return np.array([
        3.70 * np.sin(1.0 * w * t) + 2.5,
        1.30 * np.sin(2.0 * w * t),
        0.35 * np.sin(3.0 * w * t) + 1.5,
    ])

def vel(t):
    return np.array([
        3.70 * 1.0 * w * np.cos(1.0 * w * t),
        1.30 * 2.0 * w * np.cos(2.0 * w * t),
        0.35 * 3.0 * w * np.cos(3.0 * w * t),
    ])

def acc(t):
    return np.array([
        -3.70 * (1.0 * w)**2 * np.sin(1.0 * w * t),
        -1.30 * (2.0 * w)**2 * np.sin(2.0 * w * t),
        -0.35 * (3.0 * w)**2 * np.sin(3.0 * w * t),
    ])

# =============================================================================
# BUILD REFERENCE ARRAYS (100 Hz)
# =============================================================================
N_steps = int(round(T_total / DT_REF))
t_vec   = np.linspace(0.0, T_total, N_steps + 1)

p_arr = np.array([pos(t) for t in t_vec]).T   # (3, N+1)
v_arr = np.array([vel(t) for t in t_vec]).T   # (3, N+1)
a_arr = np.array([acc(t) for t in t_vec[:N_steps]]).T  # (3, N)

X_opt = np.vstack([p_arr, v_arr])   # (6, N+1)
U_opt = a_arr                        # (3, N)

v_mag = np.linalg.norm(v_arr, axis=0)
print(f"\nTrajectory: {N_steps+1} points  |  v_max={v_mag.max():.3f} m/s  |  v_mean={v_mag.mean():.3f} m/s")

# =============================================================================
# GATE PLACEMENT — curvature-weighted, crossing-exclusion
# =============================================================================
# Dense sampling for arc-length and curvature
N_arc = 10000
t_arc  = np.linspace(0.0, T_period, N_arc)
v_arc  = np.array([vel(t) for t in t_arc])          # (N_arc, 3)
a_arc  = np.array([acc(t) for t in t_arc])          # (N_arc, 3)
v_mag_arc = np.linalg.norm(v_arc, axis=1)
ds     = v_mag_arc * (T_period / N_arc)
s_cum  = np.concatenate([[0.0], np.cumsum(ds)])
arc_total = s_cum[-1]

# Curvature κ(t) = |v × a| / |v|³
cross_va = np.cross(v_arc, a_arc)
kappa    = np.linalg.norm(cross_va, axis=1) / (v_mag_arc**3 + 1e-9)
kappa_n  = kappa / kappa.max()

# Weighted density: w(s) = 1 + α·κ_normalised  → denser at turns
w_density = 1.0 + KAPPA_ALPHA * kappa_n
cumw      = np.concatenate([[0.0], np.cumsum(w_density[:-1] * ds[:-1])])
w_total   = cumw[-1]

# Place N_CANDIDATES at equal cumulative weight, half-interval offset
cand_step   = w_total / N_CANDIDATES
cand_offset = cand_step / 2.0
w_targets   = np.arange(N_CANDIDATES) * cand_step + cand_offset
t_cand      = np.interp(w_targets, cumw, np.concatenate([[0.0], t_arc[:-1]]))
pos_cand    = np.array([pos(t) for t in t_cand])

# Exclude the candidate closest to the self-intersection
keep_mask  = np.ones(N_CANDIDATES, dtype=bool)
dist_cross = np.linalg.norm(pos_cand - CROSSING_PT, axis=1)
excl_cross = int(np.argmin(dist_cross))
keep_mask[excl_cross] = False

# Additionally exclude the candidate closest to each extra point
excl_extra = []
for p_ex in EXTRA_EXCLUDE_PTS:
    d = np.linalg.norm(pos_cand - p_ex, axis=1)
    # mask out already-excluded so we don't drop the same candidate twice
    d_masked = np.where(keep_mask, d, np.inf)
    idx = int(np.argmin(d_masked))
    keep_mask[idx] = False
    excl_extra.append((idx, p_ex, float(d_masked[idx])))

t_gates    = t_cand[keep_mask]

print(f"\nArc length (1 lap) = {arc_total:.2f} m")
print(f"Candidates placed: {N_CANDIDATES} (α={KAPPA_ALPHA} curvature weighting)")
print(f"Excluded (crossing) candidate {excl_cross} at pos={pos_cand[excl_cross].round(2)} "
      f"(dist to crossing {dist_cross[excl_cross]:.2f} m)")
for idx, p_ex, d in excl_extra:
    print(f"Excluded (extra)    candidate {idx} at pos={pos_cand[idx].round(2)} "
          f"(dist to target {p_ex.round(2)} = {d:.2f} m)")

N_GATES = len(t_gates)

# Build gate attributes
gate_positions  = np.array([pos(t) for t in t_gates])
gate_velocities = np.array([vel(t) for t in t_gates])
gate_normals    = gate_velocities / np.linalg.norm(gate_velocities, axis=1, keepdims=True)

# Spacing stats (to show non-uniformity)
s_gates = np.interp(t_gates, np.concatenate([[0.0], t_arc[:-1]]), s_cum[:-1])
s_gaps  = np.diff(np.concatenate([s_gates, [s_gates[0] + arc_total]]))
print(f"Arc-length gaps between consecutive gates: min={s_gaps.min():.2f}  "
      f"max={s_gaps.max():.2f}  ratio={s_gaps.max()/s_gaps.min():.2f}×")

t_gates_full = t_gates
k_gate       = np.round(t_gates_full / DT_REF).astype(int)

print(f"\nGate layout ({len(gate_positions)} gates):")
for i, (gp, gn, tg) in enumerate(zip(gate_positions, gate_normals, t_gates_full)):
    v_at = np.linalg.norm(vel(tg))
    print(f"  G{i}: p={np.round(gp,2)}  n={np.round(gn,3)}  t={tg:.3f}s  |v|={v_at:.2f} m/s")

# =============================================================================
# VERIFY CROSSINGS (post-hoc check)
# =============================================================================
print("\nVerification — gate crossings in reference trajectory:")
all_ok = True
for i in range(len(gate_positions)):
    k_i   = k_gate[i]
    p_ref = p_arr[:, k_i]
    gpos  = gate_positions[i]
    gnorm = gate_normals[i]
    d     = p_ref - gpos
    perp  = np.dot(d, gnorm)
    radial= np.linalg.norm(d - perp * gnorm)
    ok    = radial < GATE_RADIUS
    print(f"  G{i}: r={radial:.4f}m  perp={perp:.4f}m  {'✓' if ok else '✗ MISS'}")
    if not ok:
        all_ok = False
print("  All OK ✓" if all_ok else "  Some gates missed — increase N_arc resolution")

# =============================================================================
# SAVE
# =============================================================================
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only save gate geometry — PMM will generate xref/uref/tref from these gates
np.savez(os.path.join(script_dir, 'gates.npz'),
    gate_positions = gate_positions,
    gate_normals   = gate_normals,
    gate_radius    = np.array([GATE_RADIUS]),
    safety_margin  = np.array([SAFETY_MARGIN]),
)
print(f"\nSaved: gates.npz  →  {script_dir}/")
print("Next: run path_time_3D_PMM_periodic.py to compute the time-optimal PMM reference.")

# =============================================================================
# PLOTS
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# 3D trajectory
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
vel_norm = v_mag / v_mag.max()
for k in range(len(vel_norm) - 1):
    c = plt.cm.plasma(vel_norm[k])
    ax1.plot(p_arr[0, k:k+2], p_arr[1, k:k+2], p_arr[2, k:k+2],
             color=c, linewidth=2.5, alpha=0.85)

theta_c = np.linspace(0, 2*np.pi, 60)

def _gate_circle(gp, gn, r):
    ref_v = np.array([0.,0.,1.]) if abs(gn[2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gn, ref_v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gn, t1)
    return np.array([gp + r*(np.cos(a)*t1 + np.sin(a)*t2) for a in theta_c])

for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    outer = _gate_circle(gp, gn, GATE_RADIUS)
    inner = _gate_circle(gp, gn, USABLE_RADIUS)
    ax1.plot(outer[:,0], outer[:,1], outer[:,2], color='limegreen', lw=2)
    ax1.plot(inner[:,0], inner[:,1], inner[:,2], color='darkorange', lw=1.0, ls='--', alpha=0.8)
    ax1.scatter(*gp, color='red', s=50, zorder=5)
    ax1.text(gp[0], gp[1], gp[2]+0.25, f'G{gi}', fontsize=7, ha='center')

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, v_mag.max()))
sm.set_array([])
fig.colorbar(sm, ax=ax1, shrink=0.5, label='Speed [m/s]', pad=0.1)

# 3D workspace box (HW cage walls)
def _box_edges(xr, yr, zr):
    x0,x1=xr; y0,y1=yr; z0,z1=zr
    pts = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                    [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(pts[a],pts[b]) for a,b in edges]
for a,b in _box_edges(WS_X, WS_Y, WS_Z):
    ax1.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], color='gray', lw=0.8, alpha=0.5)

ax1.set_xlabel('X [m]'); ax1.set_ylabel('Y [m]'); ax1.set_zlabel('Z [m]')
ax1.set_title(f'Lissajous 3D  (v_max={v_mag.max():.1f} m/s, cage 10.9×3.9×3 m)', fontsize=9)
ax1.set_xlim(WS_X); ax1.set_ylim(WS_Y); ax1.set_zlim(WS_Z)
ax1.view_init(elev=25, azim=-55)

# Speed profile
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_vec, v_mag, 'k-', lw=1.5)
for i, tg in enumerate(t_gates_full):
    ax2.axvline(tg, color='#2ECC71', lw=0.8, ls=':', alpha=0.8)
    ax2.text(tg, v_mag.max()*0.92, f'G{i}', fontsize=7, ha='center', color='green')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Speed [m/s]')
ax2.set_title('Speed profile')
ax2.set_ylim(0, None)
ax2.grid(True, alpha=0.4)

# XY top view
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(p_arr[0], p_arr[1], 'b-', lw=1.5, alpha=0.7)
for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    outer = _gate_circle(gp, gn, GATE_RADIUS)
    inner = _gate_circle(gp, gn, USABLE_RADIUS)
    ax3.plot(outer[:,0], outer[:,1], color='limegreen', lw=2)
    ax3.plot(inner[:,0], inner[:,1], color='darkorange', lw=1.0, ls='--', alpha=0.8)
    ax3.scatter(gp[0], gp[1], color='red', s=20, zorder=5)
    ax3.text(gp[0], gp[1]+0.18, f'G{gi}', fontsize=7, ha='center')
# XY walls
import matplotlib.patches as mpatches
ax3.add_patch(mpatches.Rectangle((WS_X[0], WS_Y[0]), WS_X[1]-WS_X[0], WS_Y[1]-WS_Y[0],
              fill=False, edgecolor='gray', lw=1.2, ls='-', alpha=0.7))
ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]')
ax3.set_title('Top view (XY) — gray: cage walls')
ax3.set_aspect('equal')
ax3.set_xlim(WS_X[0]-0.3, WS_X[1]+0.3); ax3.set_ylim(WS_Y[0]-0.3, WS_Y[1]+0.3)
ax3.grid(True, alpha=0.3)

# XZ side view
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(p_arr[0], p_arr[2], 'b-', lw=1.5, alpha=0.7)
for gi, (gp, gn) in enumerate(zip(gate_positions, gate_normals)):
    outer = _gate_circle(gp, gn, GATE_RADIUS)
    inner = _gate_circle(gp, gn, USABLE_RADIUS)
    ax4.plot(outer[:,0], outer[:,2], color='limegreen', lw=2)
    ax4.plot(inner[:,0], inner[:,2], color='darkorange', lw=1.0, ls='--', alpha=0.8)
    ax4.scatter(gp[0], gp[2], color='red', s=20, zorder=5)
    ax4.text(gp[0], gp[2]+0.1, f'G{gi}', fontsize=7, ha='center')
# XZ walls
ax4.add_patch(mpatches.Rectangle((WS_X[0], WS_Z[0]), WS_X[1]-WS_X[0], WS_Z[1]-WS_Z[0],
              fill=False, edgecolor='gray', lw=1.2, ls='-', alpha=0.7))
ax4.set_xlabel('X [m]'); ax4.set_ylabel('Z [m]')
ax4.set_title('Side view (XZ) — gray: cage walls')
ax4.set_aspect('equal')
ax4.set_xlim(WS_X[0]-0.3, WS_X[1]+0.3); ax4.set_ylim(WS_Z[0]-0.3, WS_Z[1]+0.3)
ax4.grid(True, alpha=0.3)

# Workspace clearance check
print()
print("=" * 60)
print("WORKSPACE CLEARANCE CHECK")
print("=" * 60)
print(f"Cage interior: X{WS_X}, Y{WS_Y}, Z{WS_Z}")
print(f"Required clearance from walls: r_drone + R_gate = {SAFETY_MARGIN + GATE_RADIUS:.2f} m")
print()
violations = 0
for gi, gp in enumerate(gate_positions):
    margins = {
        'x_min': gp[0] - WS_X[0],   'x_max': WS_X[1] - gp[0],
        'y_min': gp[1] - WS_Y[0],   'y_max': WS_Y[1] - gp[1],
        'z_min': gp[2] - WS_Z[0],   'z_max': WS_Z[1] - gp[2],
    }
    min_m = min(margins.values())
    tag = "OK" if min_m >= GATE_RADIUS else ("TIGHT" if min_m >= SAFETY_MARGIN else "FAIL")
    if tag == "FAIL": violations += 1
    print(f"  G{gi} @ ({gp[0]:+.2f},{gp[1]:+.2f},{gp[2]:.2f})  min_margin={min_m:.2f}m  [{tag}]")
print()
print(f"FAIL count: {violations}/{len(gate_positions)}")
print(f"Trajectory bounds: X[{p_arr[0].min():+.2f},{p_arr[0].max():+.2f}]"
      f" Y[{p_arr[1].min():+.2f},{p_arr[1].max():+.2f}]"
      f" Z[{p_arr[2].min():.2f},{p_arr[2].max():.2f}]")

plt.suptitle(f'Lissajous Reference  |  w={w:.3f} rad/s  |  v_max={v_mag.max():.1f} m/s  |  T={T_period:.2f}s', fontsize=10)
plt.tight_layout()
fig_dir = os.path.join(script_dir, 'figures', 'path_lissajous')
os.makedirs(fig_dir, exist_ok=True)
out_png = os.path.join(fig_dir, 'path_lissajous.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {out_png}")
