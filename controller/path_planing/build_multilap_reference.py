"""
build_multilap_reference.py — Concatenate ENTRY + 3× PERIODIC + EXIT.

Assumes path_time_3D_PMM_periodic.py has been run (xref/uref/tref_optimo_3D_PMM_periodic.npy).

Entry OCP:  hover (0,0,1.07) @ v=0  →  gate[0] @ v_steady  (time-optimal, no gates)
Exit OCP:   gate[0] @ v_steady       →  hover              (time-optimal, no gates)
Analysis:   3× periodic lap          (seam-continuous by construction)

Output:
    xref_multilap.npy   (6, N_total)
    uref_multilap.npy   (3, N_total-1)
    tref_multilap.npy   (N_total,)
    multilap_metadata.npz   indices of phase boundaries + durations
"""
import os, argparse
import numpy as np
import casadi as ca
from scipy.interpolate import interp1d

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

HOVER_POS       = np.array([0.0, 0.0, 1.07])   # entry start (P0, original)
HOVER_POS_EXIT  = np.array([2.0, 0.0, 1.5])    # exit landing — safe spot (avoids G7 at +0.97,-0.98,1.17)
HOVER_VEL   = np.zeros(3)
A_MAX_MAG   = 5.0 * 9.81           # 5g L2 (safety cap, non-binding for min-effort bridges)
V_MAX       = 14.0                 # per-axis
Z_MIN, Z_MAX = 0.3, 5.5
# HW cage interior (with r_drone=0.20m clearance margin built-in)
WS_X_MIN, WS_X_MAX = -2.95 + 0.20, 7.95 - 0.20    # = (-2.75, 7.75)
WS_Y_MIN, WS_Y_MAX = -1.95 + 0.20, 1.95 - 0.20    # = (-1.75, 1.75)
WS_Z_MAX           =  3.00 - 0.20                 # =  2.80
WS_Z_MIN           =  0.20                        # floor + r_drone
N_BRIDGE    = 100                  # discretization for entry bridge (no gates)
N_EXIT_PER_GATE = 80               # discretization per gate segment in exit OCP
N_ANALYSIS  = 3                    # 3 analyzed laps; warmup_out (decel) follows immediately
                                   # — eliminates buffer-lap exploit (solver maxing v_θ post-lap3)
# Asymmetric margins:
#   K_MARGIN_IN  = preamble gates before G0 (entry-side buffer)
#                  = 0 → entry bridges hover → G0 directly (analysis starts at G0).
#   K_MARGIN_OUT = suffix gates after G0    (exit-side buffer)
#                  > 0 → controller decelerates while crossing K_MARGIN_OUT more
#                  gates after the last lap before exit-bridge to hover.
K_MARGIN_IN   = 0
K_MARGIN_OUT  = 3
DT_OUT      = 0.010                # 100 Hz final reference
T_ENTRY     = 3.0                  # [s] fixed entry bridge duration (long enough that
                                   #     |v|≤V_WARMUP_START cap doesn't make problem infeasible)
T_EXIT      = 2.5                  # [s] fixed exit bridge duration (enough for decel
                                   #     at centered-PMM gate speeds up to ~11 m/s)

# Warm-up lap: drone enters G0 at V_WARMUP_START (m/s, tangent direction),
# accelerates linearly along arc-length to full periodic speed by end of warmup
# lap.  Analysis window then begins after warmup → laps 1..N_ANALYSIS at full
# periodic speed.  This avoids an entry-side curl from forcing periodic v at
# G0 directly from a static hover.
USE_WARMUP_LAP   = True
V_WARMUP_START   = 3.5    # [m/s] entry-end speed at G0 (tangent direction)

# Mirror: warmup-OUT lap after analysis. Drone decelerates v_full→V_WARMUP_END
# along arc-length within a final lap so reference smoothly guides solver to
# brake (avoids cost-surface inversion at end-of-path → solver maxing v_θ).
USE_WARMUP_OUT   = True
V_WARMUP_END     = V_WARMUP_START  # symmetric entry and exit

# Load periodic
X_per = np.load(os.path.join(SCRIPT_DIR, "xref_optimo_3D_PMM_periodic.npy"))
U_per = np.load(os.path.join(SCRIPT_DIR, "uref_optimo_3D_PMM_periodic.npy"))
t_per = np.load(os.path.join(SCRIPT_DIR, "tref_optimo_3D_PMM_periodic.npy"))

gates_cfg = np.load(os.path.join(SCRIPT_DIR, "gates.npz"))
N_GATES   = gates_cfg['gate_positions'].shape[0]

N_native        = X_per.shape[1] - 1
N_per_segment   = N_native // N_GATES
k_gate_native   = [i * N_per_segment for i in range(N_GATES)]
k_merge_in      = k_gate_native[(N_GATES - K_MARGIN_IN) % N_GATES]
k_merge_out     = k_gate_native[K_MARGIN_OUT % N_GATES] if K_MARGIN_OUT > 0 else 0

p_G0         = X_per[0:3, 0].copy()
v_steady_G0  = X_per[3:6, 0].copy()
T_lap        = float(t_per[-1])

x_merge_in   = X_per[:, k_merge_in].copy()
x_merge_out  = X_per[:, k_merge_out].copy()

# If warmup-lap is enabled, entry ends at G0 with low velocity (tangent direction).
if USE_WARMUP_LAP and K_MARGIN_IN == 0:
    v_tan = v_steady_G0 / (np.linalg.norm(v_steady_G0) + 1e-9)
    x_merge_in[3:6] = V_WARMUP_START * v_tan
    print(f"[WARMUP] entry ends at G0 with |v|={V_WARMUP_START:.2f} m/s (tangent), "
          f"warmup-lap will accelerate to |v|={np.linalg.norm(v_steady_G0):.2f} m/s")

print("=" * 60)
print("BUILD MULTILAP REFERENCE (with preamble + suffix)")
print("=" * 60)
print(f"Periodic lap: T={T_lap:.3f}s, |v_G0|={np.linalg.norm(v_steady_G0):.2f} m/s")
print(f"N_GATES={N_GATES}, K_MARGIN_IN={K_MARGIN_IN}, K_MARGIN_OUT={K_MARGIN_OUT}")
print(f"Entry  merge at G{(N_GATES-K_MARGIN_IN)%N_GATES} (k={k_merge_in:3d}): "
      f"p={x_merge_in[0:3].round(2)}  |v|={np.linalg.norm(x_merge_in[3:6]):.2f}")
print(f"Exit   merge at G{K_MARGIN_OUT%N_GATES} (k={k_merge_out:3d}): "
      f"p={x_merge_out[0:3].round(2)}  |v|={np.linalg.norm(x_merge_out[3:6]):.2f}")


# Generic bridge OCP: solves time-optimal PMM from x_start to x_end, no gates
def solve_bridge(x_start, x_end, T_fixed, N=N_BRIDGE, label="bridge",
                 u_match_start=None, u_match_end=None, v_max_mag=None):
    """
    Minimum-effort bridge with FIXED duration T.
    Cost: minimize  ∫ ‖a‖² dt + w_j ∫ ‖da/dt‖² dt   (smooth & gradual).
    BCs: x(0)=x_start, x(T)=x_end.
    u_match_start: if given, enforce u[0] = u_match_start (C⁰ control continuity at seam)
    u_match_end:   if given, enforce u[-1] = u_match_end
    """
    n_states   = 6
    n_controls = 3
    X = ca.MX.sym('X', n_states, N+1)
    U = ca.MX.sym('U', n_controls, N)
    dt = T_fixed / N

    # Cost:
    #   w_a  · ∫ ‖a‖² dt        → minimum effort  (smoothness)
    #   w_j  · ∫ ‖ȧ‖² dt        → minimum jerk    (C²)
    #   w_p  · ∫ d_straight²    → path-length proxy (avoid overshoot/loops)
    w_a = 1.0
    w_j = 0.1
    w_p = 5.0
    cost = 0
    for k in range(N):
        cost += w_a * ca.sumsqr(U[:, k]) * dt
    for k in range(N-1):
        da = (U[:, k+1] - U[:, k]) / dt
        cost += w_j * ca.sumsqr(da) * dt
    # Path-length proxy: penalise deviation from straight-line interpolation
    p_start = x_start[0:3]; p_end = x_end[0:3]
    for k in range(N+1):
        alpha = k / N
        p_line = (1 - alpha) * p_start + alpha * p_end
        cost += w_p * ca.sumsqr(X[0:3, k] - p_line) * dt

    # Dynamics RK4
    g, lbg, ubg = [], [], []
    def pmm(x, u): return ca.vertcat(x[3:6], u)
    for k in range(N):
        xk = X[:, k]; uk = U[:, k]
        k1 = pmm(xk, uk)
        k2 = pmm(xk + dt/2*k1, uk)
        k3 = pmm(xk + dt/2*k2, uk)
        k4 = pmm(xk + dt*k3,   uk)
        x_next = xk + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        g.append(X[:, k+1] - x_next)
        lbg.extend([0]*n_states); ubg.extend([0]*n_states)

    # BCs
    g.append(X[:, 0] - x_start); lbg.extend([0]*6); ubg.extend([0]*6)
    g.append(X[:, N] - x_end);   lbg.extend([0]*6); ubg.extend([0]*6)

    # Optional control continuity at seams
    if u_match_start is not None:
        g.append(U[:, 0] - u_match_start); lbg.extend([0]*3); ubg.extend([0]*3)
    if u_match_end is not None:
        g.append(U[:, N-1] - u_match_end); lbg.extend([0]*3); ubg.extend([0]*3)

    # Accel magnitude (safety cap, usually non-binding for min-effort)
    for k in range(N):
        g.append(ca.sumsqr(U[:, k]))
        lbg.append(0.0); ubg.append(A_MAX_MAG**2)

    # Optional velocity-magnitude cap (prevents overshoot in low-v entries)
    if v_max_mag is not None:
        for k in range(N+1):
            g.append(ca.sumsqr(X[3:6, k]))
            lbg.append(0.0); ubg.append(v_max_mag**2)

    # Var bounds (cage walls with r_drone clearance + per-axis vel cap)
    n_vars = n_states*(N+1) + n_controls*N
    lbx = np.full(n_vars, -np.inf); ubx = np.full(n_vars, np.inf)
    for k in range(N+1):
        idx = k*n_states
        lbx[idx+0] = WS_X_MIN; ubx[idx+0] = WS_X_MAX
        lbx[idx+1] = WS_Y_MIN; ubx[idx+1] = WS_Y_MAX
        lbx[idx+2] = WS_Z_MIN; ubx[idx+2] = WS_Z_MAX
        for j in range(3, 6):
            lbx[idx+j] = -V_MAX; ubx[idx+j] = V_MAX
    # Thrust-positive hemisphere: a_z ≥ -0.9g (tilt ≤ 84°)
    ctrl_offset = n_states * (N + 1)
    a_z_min = -0.9 * 9.81
    for k in range(N):
        lbx[ctrl_offset + k * n_controls + 2] = a_z_min

    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {'f': cost, 'x': opt_vars, 'g': ca.vertcat(*g)}
    opts = {'ipopt.max_iter': 2000, 'ipopt.tol': 1e-5,
            'ipopt.acceptable_tol': 1e-4, 'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.mu_strategy': 'adaptive'}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Warm start: linear interp state, zero control
    x0_arr = np.zeros(n_vars)
    for k in range(N+1):
        alpha = k / N
        x0_arr[k*n_states:(k+1)*n_states] = (1-alpha)*x_start + alpha*x_end

    sol = solver(x0=x0_arr, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats().get('return_status', '')
    ok = status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
    sol_x = sol['x'].full().flatten()
    X_opt = sol_x[:n_states*(N+1)].reshape((N+1, n_states)).T
    U_opt = sol_x[n_states*(N+1):].reshape((N, n_controls)).T
    a_mag_max = float(np.linalg.norm(U_opt, axis=0).max())
    print(f"  {label}: T={T_fixed:.2f}s (fixed)  |a|max={a_mag_max/9.81:.2f}g  ok={ok}")
    if not ok:
        raise RuntimeError(f"{label} OCP failed: {status}")
    t_opt = np.linspace(0.0, T_fixed, N+1)
    return X_opt, U_opt, t_opt

# Exit OCP through K gates → hover  (FREE T, distance-based decel)
# Drone enters at G0 with full periodic speed, must cross K subsequent gates,
# end at hover (v=0). Time T is a decision variable so deceleration profile
# is determined by dynamics, not arbitrary fixed time.
def solve_exit_through_gates(x_start, x_end, gate_positions_seq, gate_radius,
                              R_clearance_factor=0.6, label="EXIT_GATES"):
    n_gates_pass = len(gate_positions_seq)
    N = N_EXIT_PER_GATE * (n_gates_pass + 1)
    n_states, n_controls = 6, 3
    X = ca.MX.sym('X', n_states, N+1)
    U = ca.MX.sym('U', n_controls, N)
    T = ca.MX.sym('T')
    dt = T / N

    # Estimate sample index for each gate (proportional to straight-line arc length)
    pts = [np.asarray(x_start[0:3])]
    for gp in gate_positions_seq: pts.append(np.asarray(gp))
    pts.append(np.asarray(x_end[0:3]))
    seg_lengths = [float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1)]
    s_total = sum(seg_lengths)
    s_at_gate = np.cumsum(seg_lengths[:-1])
    k_at_gate = [int(round(s/s_total * N)) for s in s_at_gate]

    # Cost: minimize T (time-optimal decel) + small jerk regularizer
    w_T = 5.0
    w_j = 0.005
    cost = w_T * T
    for k in range(N-1):
        da = (U[:,k+1] - U[:,k]) / dt
        cost += w_j * ca.sumsqr(da) * dt

    # Dynamics RK4
    g, lbg, ubg = [], [], []
    def pmm(x,u): return ca.vertcat(x[3:6], u)
    for k in range(N):
        xk, uk = X[:,k], U[:,k]
        k1 = pmm(xk, uk); k2 = pmm(xk+dt/2*k1, uk)
        k3 = pmm(xk+dt/2*k2, uk); k4 = pmm(xk+dt*k3, uk)
        xn = xk + dt/6*(k1+2*k2+2*k3+k4)
        g.append(X[:,k+1] - xn); lbg.extend([0]*n_states); ubg.extend([0]*n_states)

    # BCs
    g.append(X[:,0] - x_start); lbg.extend([0]*6); ubg.extend([0]*6)
    g.append(X[:,N] - x_end);   lbg.extend([0]*6); ubg.extend([0]*6)

    # Gate waypoints: drone must be within R_clearance · R_gate of each gate center
    R_clear = gate_radius * R_clearance_factor
    for gi, gp in enumerate(gate_positions_seq):
        k = k_at_gate[gi]
        dp = X[0:3, k] - gp
        g.append(ca.sumsqr(dp))
        lbg.append(0.0); ubg.append(R_clear**2)

    # Accel magnitude (5g L2)
    for k in range(N):
        g.append(ca.sumsqr(U[:,k])); lbg.append(0.0); ubg.append(A_MAX_MAG**2)

    n_vars = n_states*(N+1) + n_controls*N + 1
    lbx = np.full(n_vars, -np.inf); ubx = np.full(n_vars, np.inf)
    for k in range(N+1):
        idx = k*n_states
        # Cage walls (with r_drone clearance baked into WS_*)
        lbx[idx+0] = WS_X_MIN; ubx[idx+0] = WS_X_MAX
        lbx[idx+1] = WS_Y_MIN; ubx[idx+1] = WS_Y_MAX
        lbx[idx+2] = WS_Z_MIN; ubx[idx+2] = WS_Z_MAX
        for j in range(3,6):
            lbx[idx+j] = -V_MAX; ubx[idx+j] = V_MAX
    ctrl_offset = n_states*(N+1)
    a_z_min = -0.9 * 9.81
    for k in range(N):
        lbx[ctrl_offset + k*n_controls + 2] = a_z_min
    lbx[-1] = 1.0; ubx[-1] = 12.0      # T ∈ [1s, 12s]

    # Warm start: linear interp pos through gate sequence; zero ctrl; T_guess
    T_guess = max(2.0, s_total / 6.0)
    x0_arr = np.zeros(n_vars)
    knots_k = [0] + list(k_at_gate) + [N]
    knots_p = [pts[0]] + list(gate_positions_seq) + [pts[-1]]
    for k in range(N+1):
        # find segment
        for i in range(len(knots_k)-1):
            if knots_k[i] <= k <= knots_k[i+1]:
                a = (k - knots_k[i]) / max(1, (knots_k[i+1] - knots_k[i]))
                p_lin = (1-a)*knots_p[i] + a*knots_p[i+1]
                break
        x0_arr[k*n_states:k*n_states+3] = p_lin
        # vel: linear interp from start to 0
        a_full = k / N
        x0_arr[k*n_states+3:k*n_states+6] = (1 - a_full) * np.asarray(x_start[3:6])
    x0_arr[-1] = T_guess

    opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1), T)
    nlp = {'f': cost, 'x': opt_vars, 'g': ca.vertcat(*g)}
    opts = {'ipopt.max_iter': 3000, 'ipopt.tol': 1e-5, 'ipopt.acceptable_tol': 1e-4,
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.mu_strategy': 'adaptive'}
    solver = ca.nlpsol('solver','ipopt',nlp,opts)
    sol = solver(x0=x0_arr, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats().get('return_status','')
    ok = status in ('Solve_Succeeded','Solved_To_Acceptable_Level')
    sol_x = sol['x'].full().flatten()
    X_opt = sol_x[:n_states*(N+1)].reshape((N+1, n_states)).T
    U_opt = sol_x[n_states*(N+1):n_states*(N+1)+n_controls*N].reshape((N, n_controls)).T
    T_opt = float(sol_x[-1])
    a_mag = float(np.linalg.norm(U_opt, axis=0).max())
    # Verify gate clearances
    miss = []
    for gi, gp in enumerate(gate_positions_seq):
        k = k_at_gate[gi]
        d = float(np.linalg.norm(X_opt[0:3, k] - np.asarray(gp)))
        miss.append(d)
    print(f"  {label}: T={T_opt:.2f}s (free)  |a|max={a_mag/9.81:.2f}g  "
          f"gate_misses={['%.2f' % m for m in miss]}m  ok={ok}")
    if not ok: raise RuntimeError(f"{label} OCP failed: {status}")
    t_opt = np.linspace(0.0, T_opt, N+1)
    return X_opt, U_opt, t_opt


# Solve entry + exit bridges (to/from merge points, NOT directly to G0)
x_hover      = np.concatenate([HOVER_POS,      HOVER_VEL])
x_hover_exit = np.concatenate([HOVER_POS_EXIT, HOVER_VEL])

# Periodic controls at the merge sample indices (U has N native samples; take nearest)
#   u at k_merge_in  = control applied while traversing gate G(N-K) forward → seed entry's u[-1]
#   u at k_merge_out = control applied while leaving G(K) → seed exit's u[0]
u_at_merge_in  = U_per[:, min(k_merge_in,  U_per.shape[1]-1)]
u_at_merge_out = U_per[:, min(k_merge_out, U_per.shape[1]-1)]

v_end_mag = float(np.linalg.norm(x_merge_in[3:6]))
v_max_entry = V_WARMUP_START if USE_WARMUP_LAP else max(v_end_mag * 1.2, 8.0)
print(f"\nSolving ENTRY bridge (hover → merge_in)  T={T_ENTRY}s  "
      f"[|v|_max={v_max_entry:.1f} m/s, |v_end|={v_end_mag:.1f} m/s, u_end free] ...")
X_e, U_e, t_e = solve_bridge(x_hover, x_merge_in, T_fixed=T_ENTRY, label="ENTRY",
                              u_match_end=None,
                              v_max_mag=v_max_entry)

# EXIT: unified OCP from G0 (post-warmup_out, low speed) through K_MARGIN_OUT
# subsequent gates to hover, with FREE T (decel determined by dynamics).
# If USE_WARMUP_OUT, drone arrives at G0 at V_WARMUP_END instead of v_full.
x_start_exit = X_per[:, 0].copy()
if USE_WARMUP_OUT:
    v_tan = v_steady_G0 / (np.linalg.norm(v_steady_G0) + 1e-9)
    x_start_exit[3:6] = V_WARMUP_END * v_tan
exit_gate_pos = [gates_cfg['gate_positions'][i % N_GATES]
                  for i in range(1, K_MARGIN_OUT+1)]
print(f"Solving EXIT-through-gates  start G0 at |v|={np.linalg.norm(x_start_exit[3:6]):.2f}m/s  "
      f"→ {len(exit_gate_pos)} gates → hover ...")
X_x, U_x, t_x = solve_exit_through_gates(
    x_start_exit, x_hover_exit, exit_gate_pos,
    gate_radius=float(gates_cfg['gate_radius']),
    R_clearance_factor=0.2,
    label="EXIT_THROUGH_GATES")
T_entry = t_e[-1]; T_exit = t_x[-1]

# Preamble (entry-side, K_MARGIN_IN gates).  Suffix is now embedded inside the
# unified EXIT OCP (which already passes through K_MARGIN_OUT gates with free T)
# — no separate suffix phase.
USE_PRE = K_MARGIN_IN > 0
USE_SUF = False
if USE_PRE:
    X_pre = X_per[:, k_merge_in:N_native+1].copy()
    U_pre = U_per[:, k_merge_in:N_native].copy()
    t_pre = t_per[k_merge_in:N_native+1] - t_per[k_merge_in]
    print(f"\nPreamble: {X_pre.shape[1]} samples, {t_pre[-1]:.3f}s  "
          f"(G{(N_GATES-K_MARGIN_IN)%N_GATES} → G0)")
else:
    X_pre = U_pre = t_pre = None
    print(f"\nPreamble: SKIPPED (K_MARGIN_IN=0, entry lands at G0)")
X_suf = U_suf = t_suf = None
print(f"Suffix:   N/A (decel through {K_MARGIN_OUT} gates handled inside EXIT OCP)")

# Resample all phases to 100 Hz uniform grid
def resample_phase(X, U, t, dt_out):
    """Resample states (cubic) and controls (zero-order hold) to uniform dt."""
    T = t[-1]
    n_new = max(2, int(round(T / dt_out)) + 1)
    t_new = np.linspace(0.0, T, n_new)          # exact endpoints, no overshoot
    X_new = np.stack([interp1d(t, X[i], kind='cubic')(t_new) for i in range(6)])
    t_u = 0.5 * (t[:-1] + t[1:])
    U_new = np.stack([interp1d(t_u, U[i], kind='previous',
                               bounds_error=False,
                               fill_value=(U[i,0], U[i,-1]))(t_new[:-1]) for i in range(3)])
    return X_new, U_new, t_new

X_e_u, U_e_u, t_e_u = resample_phase(X_e, U_e, t_e, DT_OUT)
X_p_u, U_p_u, t_p_u = resample_phase(X_per, U_per, t_per, DT_OUT)
X_x_u, U_x_u, t_x_u = resample_phase(X_x, U_x, t_x, DT_OUT)

# Warm-up lap: same XYZ path as periodic but speed ramps linearly along
# arc-length from V_WARMUP_START at G0 to full periodic |v| at end of lap.
# Reparametrise time so v(s) = v_low + (v_full - v_low)·(s/s_lap).
# Velocities and accelerations recomputed from finite differences.
def build_warmup_lap(X_per_native, t_per_native, v_low, k_accel_start, k_accel_end=None):
    """Piecewise warmup speed:
       v_low constant   until k_accel_start (≈ G6),
       ramp v_low→v_full from k_accel_start to k_accel_end (≈ G7),
       v_full constant  from k_accel_end to end of lap (G7→G0).
    Drone enters analysis (G0) already in steady-state regime."""
    p = X_per_native[0:3]                              # (3, N)
    v_per_native = X_per_native[3:6]
    v_full = float(np.linalg.norm(v_per_native[:, 0]))
    # Arc-length per native sample
    ds = np.linalg.norm(np.diff(p, axis=1), axis=0)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    s_max  = s[-1]
    s_acc  = s[k_accel_start]
    if k_accel_end is None:
        k_accel_end = len(s) - 1     # ramp to the end of the lap
    s_end  = s[k_accel_end]
    # Piecewise speed profile
    v_s = np.where(
        s <= s_acc,
        v_low,
        np.where(
            s >= s_end,
            v_full,
            v_low + (v_full - v_low) * (s - s_acc) / max(1e-3, s_end - s_acc)
        )
    )
    # Time grid: dt(k) = ds(k) / v_avg(k)
    v_avg = 0.5 * (v_s[:-1] + v_s[1:])
    dt_arr = ds / np.clip(v_avg, 1e-3, None)
    t_warm = np.concatenate([[0.0], np.cumsum(dt_arr)])
    # Velocity vector: same direction as periodic, magnitude v_s(k)
    v_dir = v_per_native / (np.linalg.norm(v_per_native, axis=0, keepdims=True) + 1e-9)
    v_warm = v_dir * v_s[None, :]
    # Accel: finite diff on v_warm over t_warm (N-1 control samples between N states)
    a_warm = (v_warm[:, 1:] - v_warm[:, :-1]) / np.clip(dt_arr[None, :], 1e-3, None)
    # Clip to physical 5g ceiling — kills finite-diff spurious spikes from
    # combining native v_dir (variable v_per_mag) with our independent v_s ramp.
    A_CAP = 49.05
    a_mag = np.linalg.norm(a_warm, axis=0, keepdims=True)
    scale = np.minimum(1.0, A_CAP / np.clip(a_mag, 1e-9, None))
    a_warm = a_warm * scale
    X_warm_native = np.concatenate([p, v_warm], axis=0)
    return X_warm_native, a_warm, t_warm

if USE_WARMUP_LAP and K_MARGIN_IN == 0:
    # Ramp G6 → G7, then constant v_full G7 → G0 (drone settled at analysis entry)
    k_accel_start = k_gate_native[N_GATES - 2]
    k_accel_end   = k_gate_native[N_GATES - 1]
    X_w_native, U_w_native, t_w_native = build_warmup_lap(
        X_per, t_per, V_WARMUP_START, k_accel_start, k_accel_end)
    X_w_u, U_w_u, t_w_u = resample_phase(X_w_native, U_w_native, t_w_native, DT_OUT)
    print(f"[WARMUP] lap built: T_warmup={t_w_u[-1]:.3f}s "
          f"(vs periodic T_lap={t_per[-1]:.3f}s)")

# Warmup-OUT: mirror of warmup-IN. v_full constant G0→G_decel_start, then ramp
# v_full→V_WARMUP_END from G_decel_start→G_decel_end, then v_low constant to G0.
def build_warmup_out_lap(X_per_native, t_per_native, v_low,
                          k_decel_start, k_decel_end):
    """Mirror of build_warmup_lap. Decelerate v_full → v_low along arc-length."""
    p = X_per_native[0:3]
    v_per_native = X_per_native[3:6]
    v_full = float(np.linalg.norm(v_per_native[:, 0]))
    ds = np.linalg.norm(np.diff(p, axis=1), axis=0)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    s_dec_start = s[k_decel_start]
    s_dec_end   = s[k_decel_end]
    v_s = np.where(
        s <= s_dec_start,
        v_full,
        np.where(
            s >= s_dec_end,
            v_low,
            v_full + (v_low - v_full) * (s - s_dec_start) / max(1e-3, s_dec_end - s_dec_start)
        )
    )
    v_avg = 0.5 * (v_s[:-1] + v_s[1:])
    dt_arr = ds / np.clip(v_avg, 1e-3, None)
    t_warm = np.concatenate([[0.0], np.cumsum(dt_arr)])
    v_dir = v_per_native / (np.linalg.norm(v_per_native, axis=0, keepdims=True) + 1e-9)
    v_warm = v_dir * v_s[None, :]
    a_warm = (v_warm[:, 1:] - v_warm[:, :-1]) / np.clip(dt_arr[None, :], 1e-3, None)
    # Cap warmup_out at 3g (gentler decel reference than racing 5g cap)
    A_CAP = 29.43  # 3g
    a_mag = np.linalg.norm(a_warm, axis=0, keepdims=True)
    scale = np.minimum(1.0, A_CAP / np.clip(a_mag, 1e-9, None))
    a_warm = a_warm * scale
    X_warm_native = np.concatenate([p, v_warm], axis=0)
    return X_warm_native, a_warm, t_warm

if USE_WARMUP_OUT:
    # Decel spread over FULL lap (G0→G7 ≈ 24m). Gentle slope (~0.15 m/s²)
    # so drone has arc-length budget to brake without solver overshoot.
    k_decel_start = 0                              # G0 (start of warmup_out lap)
    k_decel_end   = k_gate_native[N_GATES - 1]    # G7 (end-1)
    X_wo_native, U_wo_native, t_wo_native = build_warmup_out_lap(
        X_per, t_per, V_WARMUP_END, k_decel_start, k_decel_end)
    X_wo_u, U_wo_u, t_wo_u = resample_phase(X_wo_native, U_wo_native, t_wo_native, DT_OUT)
    print(f"[WARMUP_OUT] lap built: T_warmup_out={t_wo_u[-1]:.3f}s "
          f"(decel v_full→{V_WARMUP_END:.2f} m/s)")
if USE_PRE:
    X_pre_u, U_pre_u, t_pre_u = resample_phase(X_pre, U_pre, t_pre, DT_OUT)
if USE_SUF:
    X_suf_u, U_suf_u, t_suf_u = resample_phase(X_suf, U_suf, t_suf, DT_OUT)

# Force exact BC match at phase boundaries (replace last sample with exact BC)
# This removes resampling artifacts at seams.
def pin_endpoint(X_u, x_target):
    """Pin the last state sample to exact target (removes interpolation drift)."""
    X_u = X_u.copy()
    X_u[:, -1] = x_target
    return X_u

X_e_u = pin_endpoint(X_e_u, x_merge_in)   # entry ends exactly at merge_in
X_x_u = X_x_u.copy()
X_x_u[:, 0]  = x_start_exit               # exit starts at G0 with periodic velocity
X_x_u[:, -1] = x_hover_exit               # exit ends exactly at hover_exit

# Preamble/suffix already come from native periodic with exact seam matching:
#   X_pre[:, -1] == X_per[:, N] == X_per[:, 0] (periodic)  → matches lap1[0]
#   X_suf[:, 0]  == X_per[:, 0]                           → matches lap3[-1]
# (cubic resampling preserves this within interpolation precision)

# Concatenate: ENTRY ⊕ 3× PERIODIC ⊕ EXIT
# Drop duplicated seam sample (last of each phase except the final one)
phases = [('entry', X_e_u, U_e_u, t_e_u, False)]
if USE_PRE:
    phases.append(('preamble', X_pre_u, U_pre_u, t_pre_u, False))
if USE_WARMUP_LAP and K_MARGIN_IN == 0:
    phases.append(('warmup', X_w_u, U_w_u, t_w_u, False))
for lap in range(N_ANALYSIS):
    phases.append((f'lap{lap+1}', X_p_u, U_p_u, t_p_u, False))
if USE_SUF:
    phases.append(('suffix', X_suf_u, U_suf_u, t_suf_u, False))
if USE_WARMUP_OUT:
    phases.append(('warmup_out', X_wo_u, U_wo_u, t_wo_u, False))
phases.append(('exit', X_x_u, U_x_u, t_x_u, True))

parts_X, parts_U, parts_t = [], [], []
phase_starts = {}
cursor, cursor_t = 0, 0.0
for (name, Xu, Uu, tu, is_last) in phases:
    n = Xu.shape[1] if is_last else Xu.shape[1] - 1
    parts_X.append(Xu[:, :n])
    n_u = min(n, Uu.shape[1])
    parts_U.append(Uu[:, :n_u])
    parts_t.append(tu[:n] + cursor_t)
    phase_starts[name] = cursor
    cursor   += n
    cursor_t += tu[-1]

X_cat = np.concatenate(parts_X, axis=1)
U_cat = np.concatenate(parts_U, axis=1)
t_cat = np.concatenate(parts_t)
N_total = X_cat.shape[1]

# Indices for phase ends
phase_ends = {}
keys = list(phase_starts.keys())
for i, k in enumerate(keys):
    phase_ends[k] = phase_starts[keys[i+1]] - 1 if i+1 < len(keys) else N_total - 1

# Analysis window = lap1.start .. lap3.end
k_analysis_start = phase_starts['lap1']
k_analysis_end   = phase_ends['lap3']

# Sanity check: forward-Euler consistency across seams
# Checks that p[k] = p[k-1] + v*dt within tolerance (i.e., no state jump)
print("\nSeam consistency (forward-Euler residual):")
seam_names = (['preamble'] if USE_PRE else []) + \
             [f'lap{i+1}' for i in range(N_ANALYSIS)] + \
             (['suffix'] if USE_SUF else []) + ['exit']
for name in seam_names:
    k = phase_starts[name]
    dt_local = t_cat[k] - t_cat[k-1]
    p_pred = X_cat[0:3, k-1] + X_cat[3:6, k-1] * dt_local
    p_err  = np.linalg.norm(X_cat[0:3, k] - p_pred)
    v_pred = X_cat[3:6, k-1]                          # zero-acc assumption
    v_err  = np.linalg.norm(X_cat[3:6, k] - v_pred)
    # at 10 m/s × 10 ms, natural forward-Euler residual ~ (a*dt²)/2 ≈ 2 mm at 4g
    ok = p_err < 0.02 and v_err < 0.5
    tag = "✓" if ok else "⚠"
    print(f"  seam→{name:6s} @ k={k:<4d}  Δp_fwd={p_err*1000:5.1f} mm  "
          f"Δv_fwd={v_err*1000:5.1f} mm/s  {tag}")

# Save
np.save(os.path.join(SCRIPT_DIR, "xref_multilap.npy"), X_cat)
np.save(os.path.join(SCRIPT_DIR, "uref_multilap.npy"), U_cat)
np.save(os.path.join(SCRIPT_DIR, "tref_multilap.npy"), t_cat)
np.savez(os.path.join(SCRIPT_DIR, "multilap_metadata.npz"),
         phase_starts=np.array(list(phase_starts.items()), dtype=object),
         phase_ends=np.array(list(phase_ends.items()), dtype=object),
         T_entry=T_entry, T_lap=T_lap, T_exit=T_exit,
         N_analysis_laps=N_ANALYSIS - 1,   # buffer lap not analyzed
         k_analysis_start=k_analysis_start,
         k_analysis_end=k_analysis_end,
         hover_pos=HOVER_POS,
         v_steady_G0=v_steady_G0,
         dt_out=DT_OUT)

print(f"\n✓ Saved  xref_multilap.npy  uref_multilap.npy  tref_multilap.npy  metadata.npz")
print(f"  N_total    = {N_total}  samples  ({t_cat[-1]:.2f} s total at {1/DT_OUT:.0f} Hz)")
print(f"  Phases:")
for k, v in phase_starts.items():
    e = phase_ends[k]
    print(f"    {k:6s} : samples [{v:4d} .. {e:4d}]   t=[{t_cat[v]:5.2f} .. {t_cat[e]:5.2f}] s")
print(f"  Analysis window: [{k_analysis_start} .. {k_analysis_end}]  = {(k_analysis_end-k_analysis_start)*DT_OUT:.2f} s")

# Validation plot
v_cat_mag = np.linalg.norm(X_cat[3:6], axis=0)
a_cat_mag = np.linalg.norm(U_cat, axis=0)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# speed
axes[0].plot(t_cat, v_cat_mag, 'k-', lw=1.2)
axes[0].set_ylabel('|v| [m/s]'); axes[0].grid(alpha=0.35)
axes[0].set_title(f"Multi-lap reference  —  {t_cat[-1]:.2f} s total, analysis = {(k_analysis_end-k_analysis_start)*DT_OUT:.2f} s")

# accel magnitude
axes[1].plot(t_cat[:-1], a_cat_mag, 'k-', lw=0.9)
axes[1].axhline(A_MAX_MAG, color='red', ls=':', alpha=0.6, label='5g')
axes[1].set_ylabel('|a| [m/s²]'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.35)

# z
axes[2].plot(t_cat, X_cat[2], 'k-', lw=1.0)
axes[2].set_ylabel('z [m]'); axes[2].set_xlabel('t [s]'); axes[2].grid(alpha=0.35)

# phase shading
colors = {'entry': '#ddd', 'preamble': '#f5d198',
          'lap1': '#c6e2a3', 'lap2': '#c6e2a3', 'lap3': '#c6e2a3',
          'suffix': '#f5d198', 'exit': '#ddd'}
for ax in axes:
    for name, ks in phase_starts.items():
        ke = phase_ends[name]
        ax.axvspan(t_cat[ks], t_cat[ke], color=colors.get(name, '#eee'), alpha=0.35, lw=0)
    # seam lines
    for name in seam_names:
        ax.axvline(t_cat[phase_starts[name]], color='red', ls=':', alpha=0.6, lw=0.9)
# label phases (on top axis)
for name, ks in phase_starts.items():
    ke = phase_ends[name]
    t_mid = (t_cat[ks] + t_cat[ke]) / 2
    axes[0].text(t_mid, axes[0].get_ylim()[1]*0.92, name.upper(),
                 ha='center', fontsize=9, fontweight='bold')

fig.tight_layout()
_fig_dir = os.path.join(SCRIPT_DIR, "figures", "multilap")
os.makedirs(_fig_dir, exist_ok=True)
out_png = os.path.join(_fig_dir, "multilap_reference.png")
fig.savefig(out_png, dpi=140, bbox_inches='tight')
print(f"✓ Saved: {out_png}")

# 3D view
fig3d = plt.figure(figsize=(10, 8))
try:
    ax3 = fig3d.add_subplot(111, projection='3d')
except Exception:
    ax3 = fig3d.add_subplot(111)
try:
    vn = v_cat_mag / v_cat_mag.max()
    for k in range(0, N_total-1, 3):
        ax3.plot(X_cat[0, k:k+2], X_cat[1, k:k+2], X_cat[2, k:k+2],
                 color=plt.cm.plasma(vn[k]), lw=1.2, alpha=0.85)
    ax3.scatter(*HOVER_POS, color='blue', s=80, zorder=5, label='hover start/end')
    ax3.scatter(*p_G0, color='red',  s=80, zorder=5, label='gate[0]')
    ax3.set_title('Multi-lap reference — 3D')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.legend(fontsize=9)
except Exception:
    pass
fig3d.tight_layout()
out_3d = os.path.join(_fig_dir, "multilap_reference_3d.png")
fig3d.savefig(out_3d, dpi=140, bbox_inches='tight')
print(f"✓ Saved: {out_3d}")
