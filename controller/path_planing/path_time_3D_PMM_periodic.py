"""
path_time_3D_PMM_periodic.py — PMM time-optimal PERIODIC lap.

Periodic lap instead of a hover-to-hover problem:
  - periodic boundary conditions x(0) == x(T_lap) (position and velocity)
  - v(0) = v(T) = 0 is not imposed
  - the solver chooses the optimal crossing velocity at every gate
  - the first gate is not appended at the end; periodicity closes the lap

Used to build:
    Entry ⊕ 5× Periodic ⊕ Exit   (complete multi-lap reference)

Outputs (suffix _periodic):
    xref_optimo_3D_PMM_periodic.npy   (6, N+1)  = [p(0..N); v(0..N)]
    uref_optimo_3D_PMM_periodic.npy   (3, N)    = [a_x, a_y, a_z]
    tref_optimo_3D_PMM_periodic.npy   (N+1,)
"""
import os, sys, argparse
import numpy as np
import casadi as ca

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CLI
ap = argparse.ArgumentParser()
ap.add_argument('--gates', default='gates.npz')
ap.add_argument('--suffix', default=None,
                help='suffix appended to xref/uref/tref filenames; '
                     'if unset → _periodic (centered) / _periodic_free (free)')
ap.add_argument('--mode', choices=['free', 'centered'], default='centered',
                help='free: PMM may pass within R_usable from gate center (diagnostic); '
                     'centered: PMM forced through gate centers (main MPCC reference)')
ap.add_argument('--laps-preview', type=int, default=3,
                help='number of concatenated laps in validation plot')
ap.add_argument('--max-r', type=float, default=None,
                help='override max radial deviation at gates [m]. '
                     'Default: centered=0.05, free=R_usable. '
                     'Relax (e.g. 0.20) for aggressive circuits where '
                     'strict centering is infeasible.')
args = ap.parse_args()

gate_file = args.gates if os.path.isabs(args.gates) else os.path.join(SCRIPT_DIR, args.gates)
if args.suffix is not None:
    SUFFIX = args.suffix
elif args.mode == 'centered':
    SUFFIX = '_periodic'           # main reference (MPCC + tube consume this)
else:
    SUFFIX = '_periodic_free'      # diagnostic only

# GATES
g_cfg = np.load(gate_file)
gate_positions = g_cfg['gate_positions'].astype(float)   # (7, 3)
gate_normals   = g_cfg['gate_normals'].astype(float)
n_gates        = len(gate_positions)
gate_radius   = float(g_cfg['gate_radius'])
safety_margin = float(g_cfg['safety_margin'])

print("=" * 60)
print("PMM PERIODIC LAP — time-optimal through gates")
print("=" * 60)
for i, (pos, n) in enumerate(zip(gate_positions, gate_normals)):
    print(f"  Gate {i}: pos={np.round(pos,2)}, n={np.round(n,3)}")
print(f"  gate_radius={gate_radius:.2f} m  safety_margin={safety_margin:.2f} m")

# DISCRETIZATION
n_states   = 6
n_controls = 3
N_per_segment = 20
N_segments    = n_gates          # 7 segments for 7 gates (seam = gate[0] ↔ gate[0])
N             = N_per_segment * N_segments      # 140
k_gate        = [i * N_per_segment for i in range(n_gates)]
delta_k       = N_per_segment // 2

total_dist = (sum(np.linalg.norm(gate_positions[(i+1) % n_gates] - gate_positions[i])
                  for i in range(n_gates)))
print(f"\nN={N}  k_gate={k_gate}  total_dist={total_dist:.2f} m")

# OPT VARIABLES
X = ca.MX.sym('X', n_states, N + 1)
U = ca.MX.sym('U', n_controls, N)
T = ca.MX.sym('T')
dt = T / N

# COST
w_T      = 500.0
w_a      = 5e-4
w_center = 0.5

cost = w_T * T
for k in range(N):
    cost += w_a * ca.sumsqr(U[:, k]) * dt

# centering at each gate
for i in range(n_gates):
    cost += w_center * ca.sumsqr(X[0:3, k_gate[i]] - gate_positions[i])

# CONSTRAINTS
g_list, lbg, ubg = [], [], []

# 1) Dynamics (RK4)
def pmm(x, u):
    return ca.vertcat(x[3:6], u)

for k in range(N):
    xk = X[:, k]; uk = U[:, k]
    k1 = pmm(xk, uk)
    k2 = pmm(xk + dt/2*k1, uk)
    k3 = pmm(xk + dt/2*k2, uk)
    k4 = pmm(xk + dt*k3,   uk)
    x_next = xk + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    g_list.append(X[:, k+1] - x_next)
    lbg.extend([0]*n_states); ubg.extend([0]*n_states)

# 2) PERIODICITY — x(0) == x(N)  (no hard initial condition)
g_list.append(X[:, 0] - X[:, N])
lbg.extend([0]*n_states); ubg.extend([0]*n_states)

# 3) Gate passage
#    mode=centered: force PMM through gate CENTER (main MPCC reference).
#                   Guarantees the contouring tube stays inside R_gate at gates.
#    mode=free:     allow deviation within R_usable = R_gate - safety_margin.
#                   Produces a more-aggressive theoretical lower bound (diagnostic).
max_perp = 0.05
if args.mode == 'centered':
    max_r = args.max_r if args.max_r is not None else 0.05
    print(f"Gate constraint: CENTERED  (max_r = {max_r:.3f} m)")
else:
    max_r = args.max_r if args.max_r is not None else (gate_radius - safety_margin)
    print(f"Gate constraint: FREE      (max_r = {max_r:.3f} m)")

for i in range(n_gates):
    k_i   = k_gate[i]
    gpos  = gate_positions[i]
    gnorm = gate_normals[i]
    d     = X[0:3, k_i] - gpos
    dist_perp = ca.dot(d, gnorm)
    g_list.append(dist_perp)
    lbg.append(-max_perp); ubg.append(max_perp)
    d_plane = d - dist_perp * gnorm
    g_list.append(ca.sumsqr(d_plane))
    lbg.append(0); ubg.append(max_r**2)

# 4) Crossing direction — v aligned with gate normal at each gate
v_cross_min = 0.5
for i in range(n_gates):
    k_i   = k_gate[i]
    g_list.append(ca.dot(X[3:6, k_i], gate_normals[i]))
    lbg.append(v_cross_min); ubg.append(1000)

# 5) Accel magnitude ‖a‖ ≤ 4g
a_max_mag = 5.0 * 9.81
for k in range(N):
    g_list.append(ca.sumsqr(U[:, k]))
    lbg.append(0.0); ubg.append(a_max_mag**2)
print(f"Accel constraint: ‖a‖ ≤ {a_max_mag:.2f} m/s² = 5.00 g (L2)")

# NLP
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), T)
n_vars = opt_vars.shape[0]
g_vec = ca.vertcat(*g_list)
nlp = {'f': cost, 'x': opt_vars, 'g': g_vec}
opts = {
    'ipopt.max_iter': 20000,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 20,
    'ipopt.print_level': 3,
    'print_time': 1,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.mu_init': 1.0,
    'ipopt.bound_push': 1e-4,
    'ipopt.bound_frac': 1e-4,
    'ipopt.nlp_scaling_method': 'gradient-based',
    'ipopt.hessian_approximation': 'limited-memory',
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# VAR BOUNDS
v_max = 14.0
z_min, z_max = 0.3, 5.5
# Structural feasibility: thrust must point up (quadrotor can't push down).
# a_z + g ≥ 0.1 g  → a_z ≥ -0.9 g  → max tilt ≈ 84°.
# Keeps dynamic infeasibility (4g L2 budget) while removing unphysical
# inversion moments that would poison the diff-flat attitude reference.
a_z_min = -0.9 * 9.81
lbx = np.full(n_vars, -np.inf)
ubx = np.full(n_vars,  np.inf)
for k in range(N + 1):
    idx = k * n_states
    lbx[idx+2] = z_min; ubx[idx+2] = z_max
    for j in range(3, 6):
        lbx[idx+j] = -v_max; ubx[idx+j] = v_max
# a_z lower bound on control variables (indices n_states*(N+1) .. end-1)
ctrl_offset = n_states * (N + 1)
for k in range(N):
    lbx[ctrl_offset + k * n_controls + 2] = a_z_min      # a_z ≥ -0.9g
lbx[-1] = 1.0; ubx[-1] = 60.0

# WARM START (multi-seed)
def build_x0(T_init, speed_frac=0.75):
    x0 = np.zeros(n_vars)
    for k in range(N + 1):
        seg = min(k // N_per_segment, N_segments - 1)
        t_local = (k - seg * N_per_segment) / N_per_segment
        p_a = gate_positions[seg]
        p_b = gate_positions[(seg + 1) % n_gates]
        pos = (1 - t_local) * p_a + t_local * p_b
        vel_dir = p_b - p_a
        vmag = np.linalg.norm(vel_dir)
        spd = v_max * speed_frac * (0.6 if (t_local < 0.15 or t_local > 0.85) else 1.0)
        vel = (vel_dir / vmag) * spd if vmag > 0 else np.zeros(3)
        x0[k*n_states:k*n_states+3]   = pos
        x0[k*n_states+3:k*n_states+6] = vel
    x0[-1] = T_init
    return x0

T_seeds     = [total_dist / (0.70 * v_max),
               total_dist / (0.55 * v_max),
               total_dist / (0.85 * v_max),
               total_dist / (0.40 * v_max)]
speed_fracs = [0.75, 0.60, 0.85, 0.50]

print("\n" + "=" * 60)
print("MULTI-START")
print("=" * 60)
best_sol, best_T = None, np.inf
for i, (T_s, sf) in enumerate(zip(T_seeds, speed_fracs)):
    print(f"--- seed {i+1}/{len(T_seeds)}  T_init={T_s:.2f}s  spd_frac={sf:.2f}")
    try:
        sol = solver(x0=build_x0(T_s, sf), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        T_s_opt = float(sol['x'].full().flatten()[-1])
        ok = solver.stats().get('return_status', '') in (
            'Solve_Succeeded', 'Solved_To_Acceptable_Level')
        print(f"    T_opt={T_s_opt:.3f}  ok={ok}")
        if ok and T_s_opt < best_T:
            best_T, best_sol = T_s_opt, sol
    except Exception as e:
        print(f"    FAILED: {e}")

if best_sol is None:
    raise RuntimeError("All multi-start seeds failed for periodic OCP")

# Phase-2 refine
sol2 = solver(x0=best_sol['x'], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
T2 = float(sol2['x'].full().flatten()[-1])
if solver.stats().get('return_status', '') in (
    'Solve_Succeeded', 'Solved_To_Acceptable_Level') and T2 < best_T:
    best_T, best_sol = T2, sol2

print(f"\n{'='*60}\nBEST T_lap = {best_T:.3f} s\n{'='*60}")

# EXTRACT
sol_x = best_sol['x'].full().flatten()
X_opt = sol_x[:n_states*(N+1)].reshape((N+1, n_states)).T   # (6, N+1)
U_opt = sol_x[n_states*(N+1):n_states*(N+1)+n_controls*N].reshape((N, n_controls)).T
T_lap = float(sol_x[-1])
t_traj = np.linspace(0.0, T_lap, N+1)

v_all   = X_opt[3:6]
v_mag   = np.linalg.norm(v_all, axis=0)
v_G0    = v_all[:, 0]
a_all   = U_opt
a_mag   = np.linalg.norm(a_all, axis=0)

print(f"\nSeam state at gate[0]:")
print(f"  p(0) = {X_opt[0:3, 0].round(3)}     p(T) = {X_opt[0:3, -1].round(3)}")
print(f"  v(0) = {v_G0.round(3)} m/s   |v(0)| = {np.linalg.norm(v_G0):.2f}")
print(f"  v(T) = {v_all[:, -1].round(3)}")
print(f"\nLap stats:")
print(f"  T_lap    = {T_lap:.3f} s")
print(f"  |v|_max  = {v_mag.max():.3f} m/s   |v|_mean = {v_mag.mean():.3f}")
print(f"  |a|_max  = {a_mag.max():.3f} m/s² = {a_mag.max()/9.81:.2f} g")

# Periodicity sanity
seam_err_p = np.linalg.norm(X_opt[0:3, 0] - X_opt[0:3, -1])
seam_err_v = np.linalg.norm(X_opt[3:6, 0] - X_opt[3:6, -1])
print(f"\nPeriodicity residuals:  Δp = {seam_err_p:.2e} m   Δv = {seam_err_v:.2e} m/s")

# GATE VERIFICATION
print("\nGate verification:")
all_ok = True
for i in range(n_gates):
    k_i = k_gate[i]
    d = X_opt[0:3, k_i] - gate_positions[i]
    dp = float(np.dot(d, gate_normals[i]))
    dr = float(np.linalg.norm(d - dp * gate_normals[i]))
    v_dot_n = float(np.dot(X_opt[3:6, k_i], gate_normals[i]))
    ok = dr < gate_radius and abs(dp) < 0.1 and v_dot_n > 0
    all_ok &= ok
    print(f"  G{i}: r={dr:.3f}m  perp={dp:+.3f}m  v·n={v_dot_n:+.2f} m/s  {'✓' if ok else '✗'}")
print(f"  → {'ALL GATES CROSSED ✓' if all_ok else 'SOME GATES MISSED ✗'}")

# SAVE
np.save(os.path.join(SCRIPT_DIR, f"xref_optimo_3D_PMM{SUFFIX}.npy"), X_opt)
np.save(os.path.join(SCRIPT_DIR, f"uref_optimo_3D_PMM{SUFFIX}.npy"), U_opt)
np.save(os.path.join(SCRIPT_DIR, f"tref_optimo_3D_PMM{SUFFIX}.npy"), t_traj)
print(f"\n✓ Saved:  xref_optimo_3D_PMM{SUFFIX}.npy  uref...  tref...")

# VALIDATION PLOT — concatenate N_LAPS laps and visualize seams
N_LAPS = args.laps_preview
# Drop last sample of each lap except the very last, to avoid duplicated seam points
p_cat = np.concatenate([X_opt[0:3, :-1]] * (N_LAPS-1) + [X_opt[0:3]], axis=1)
v_cat = np.concatenate([X_opt[3:6, :-1]] * (N_LAPS-1) + [X_opt[3:6]], axis=1)
t_cat = np.concatenate([t_traj[:-1] + l*T_lap for l in range(N_LAPS-1)] + [t_traj + (N_LAPS-1)*T_lap])
a_cat = np.concatenate([U_opt] * N_LAPS, axis=1)
t_u_cat = np.concatenate([t_traj[:-1] + l*T_lap for l in range(N_LAPS)])

v_cat_mag = np.linalg.norm(v_cat, axis=0)
a_cat_mag = np.linalg.norm(a_cat, axis=0)

fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
vn = v_cat_mag / max(v_cat_mag.max(), 1e-9)
for k in range(len(vn)-1):
    ax1.plot(p_cat[0,k:k+2], p_cat[1,k:k+2], p_cat[2,k:k+2],
             color=plt.cm.plasma(vn[k]), lw=1.8, alpha=0.9)
for i in range(n_gates):
    theta = np.linspace(0, 2*np.pi, 60)
    ref = np.array([0.,0.,1.]) if abs(gate_normals[i,2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gate_normals[i], ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gate_normals[i], t1)
    circ = np.array([gate_positions[i] + gate_radius*(np.cos(a)*t1+np.sin(a)*t2) for a in theta])
    ax1.plot(circ[:,0], circ[:,1], circ[:,2], color='limegreen', lw=1.5)
    ax1.text(*gate_positions[i], f'G{i}', fontsize=7)
ax1.set_title(f'{N_LAPS} laps concatenated  (T_lap={T_lap:.2f}s)', fontsize=10)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_cat, v_cat_mag, 'k-', lw=1.3)
for l in range(1, N_LAPS):
    ax2.axvline(l*T_lap, color='red', ls=':', alpha=0.6, lw=1.0)
ax2.set_xlabel('t [s]'); ax2.set_ylabel('|v| [m/s]')
ax2.set_title('Speed profile — red dashed = lap seam (should be continuous)')
ax2.grid(alpha=0.35)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_u_cat, a_cat[0], 'r', lw=0.8, label='a_x')
ax3.plot(t_u_cat, a_cat[1], 'g', lw=0.8, label='a_y')
ax3.plot(t_u_cat, a_cat[2], 'b', lw=0.8, label='a_z')
ax3.plot(t_u_cat, a_cat_mag, 'k--', lw=1.0, label='|a|')
ax3.axhline(a_max_mag, color='gray', ls=':', alpha=0.6, label='5g')
for l in range(1, N_LAPS):
    ax3.axvline(l*T_lap, color='red', ls=':', alpha=0.6, lw=1.0)
ax3.set_xlabel('t [s]'); ax3.set_ylabel('a [m/s²]')
ax3.set_title('Acceleration — continuity at seams')
ax3.legend(ncol=5, fontsize=8); ax3.grid(alpha=0.35)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(p_cat[0], p_cat[1], 'b', lw=1.2, alpha=0.8)
for i in range(n_gates):
    theta = np.linspace(0, 2*np.pi, 60)
    ref = np.array([0.,0.,1.]) if abs(gate_normals[i,2]) < 0.9 else np.array([1.,0.,0.])
    t1 = np.cross(gate_normals[i], ref); t1 /= np.linalg.norm(t1)
    t2 = np.cross(gate_normals[i], t1)
    circ = np.array([gate_positions[i] + gate_radius*(np.cos(a)*t1+np.sin(a)*t2) for a in theta])
    ax4.plot(circ[:,0], circ[:,1], color='limegreen', lw=1.5)
    ax4.text(gate_positions[i,0], gate_positions[i,1]+0.18, f'G{i}', fontsize=7, ha='center')
ax4.set_xlabel('x'); ax4.set_ylabel('y')
ax4.set_title('Top view — all laps overlaid (identical)')
ax4.set_aspect('equal'); ax4.grid(alpha=0.3)

fig.suptitle(f"Periodic PMM  —  T_lap={T_lap:.3f}s  |v|_max={v_mag.max():.2f}m/s  "
             f"|a|_max={a_mag.max()/9.81:.2f}g  seam Δp={seam_err_p:.1e}m  Δv={seam_err_v:.1e}m/s",
             fontsize=11, fontweight='bold')
fig.tight_layout()
_fig_dir = os.path.join(SCRIPT_DIR, "figures", "pmm")
os.makedirs(_fig_dir, exist_ok=True)
out_png = os.path.join(_fig_dir, f"pmm_periodic_{N_LAPS}laps_{args.mode}.png")
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {out_png}")
