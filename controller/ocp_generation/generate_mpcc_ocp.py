"""
Generate acados C code for a quadrotor MPCC (Model Predictive Contouring Control).

State  x ∈ ℝ¹⁶ = [p(3), v(3), q(4), ω(3), θ, v_θ, f]
Control u ∈ ℝ⁵  = [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]

Rate-control plant:  ω̇ = (ω_cmd − ω) / τ_rc
Progress dynamics:   θ̇ = v_θ,  v̇_θ = a_θ
Thrust dynamics:     ḟ = Δf  (thrust as state, rate as control)

Cost: MPCC contouring/lag + attitude + control + progress maximisation.
  Waypoint path (pos, tangent, quat) embedded as B-spline in the symbolic graph.
  Runtime params p ∈ ℝ¹⁶ for weights (no rebuild to change weights).

Lag constraint: |e_lag_scalar| ≤ D_MAX_LAG (nonlinear h-constraint).

Run once:  python3 generate_mpcc_ocp.py
Output:    ../c_generated_code_mpcc/
"""

import os
import shutil
import numpy as np
import casadi as ca
from casadi import MX, vertcat, norm_2, if_else, atan2, dot, diag as casadi_diag
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# Local imports (self-contained, no parent project dependency)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from quad_config import (
    MASS, G, TAU_RC,
    T_MAX, T_MIN, W_MAX, DF_MAX,
    VTHETA_MIN, VTHETA_MAX, ATHETA_MIN, ATHETA_MAX,
    D_MAX_LAG, N_WAYPOINTS, S_MAX_MANUAL, THETA_MARGIN,
    TRAJECTORY_T_FINAL, FREC,
    ATTITUDE_REF_SPEED, ATTITUDE_REF_MAX_TILT_DEG,
    trajectory,
)
from path_utils import build_arc_length_parameterisation, build_waypoints

# OCP dimensions
NX = 16   # [p(3), v(3), q(4), ω(3), θ, v_θ, f]
NU = 5    # [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
DT_CONTROL = 0.01    # [s] control loop period (100 Hz)
T_HORIZON  = 1.48    # [s] prediction horizon (non-uniform grid, N=31)

# Three-tier non-uniform grid (maximum step ratio 4), shared by both controllers.
TIME_STEPS = np.concatenate([
    np.full(10, 0.020),  # 10 × 20ms = 0.200s  (control-rate-ish)
    np.full(10, 0.040),  # 10 × 40ms = 0.400s
    np.full(11, 0.080),  # 11 × 80ms = 0.880s
])
N_HORIZON = len(TIME_STEPS)  # = 31
assert abs(TIME_STEPS.sum() - T_HORIZON) < 1e-9, f"Time steps sum {TIME_STEPS.sum()} != T_HORIZON {T_HORIZON}"

# Sub-steps for ERK4 integrator: ~10-15ms per sub-step.
SIM_NUM_STEPS = np.concatenate([
    np.full(10, 2),      # 20ms → 2 sub-steps (10ms each)
    np.full(10, 3),      # 40ms → 3 sub-steps (~13ms)
    np.full(11, 5),      # 80ms → 5 sub-steps (16ms)
]).astype(int)

# Runtime parameters: weights that can change without rebuilding
# p[ 0: 3] = Q_ec      contouring error weights
# p[ 3: 6] = Q_el      lag error weights
# p[ 6: 9] = Q_q       quaternion-log error weights
# p[ 9:13] = U_mat     control effort weights [f_err, ωx_cmd, ωy_cmd, ωz_cmd]
# p[13]    = Q_s       linear progress weight in -Q_s * v_theta
# p[14]    = vtheta_max runtime upper bound helper
# p[15]    = W_df      thrust rate penalty: W_df * Δf²
N_PARAMS = 16


#  B-spline interpolation (CasADi symbolic)

def _bspline_scalar(name, s_sym, s_wp, values_wp):
    """Clamped scalar B-spline interpolant via CasADi interpolant."""
    s_c = ca.fmin(ca.fmax(s_sym, float(s_wp[0])), float(s_wp[-1]))
    lut = ca.interpolant(
        name, "bspline",
        [np.asarray(s_wp, dtype=float)],
        np.asarray(values_wp, dtype=float).reshape(-1),
    )
    return lut(s_c)


def create_gamma_pos(s_sym, s_wp, pos_wp):
    """γ_pos(θ) : ℝ → ℝ³ position on path."""
    px = _bspline_scalar('gamma_pos_x', s_sym, s_wp, pos_wp[0, :])
    py = _bspline_scalar('gamma_pos_y', s_sym, s_wp, pos_wp[1, :])
    pz = _bspline_scalar('gamma_pos_z', s_sym, s_wp, pos_wp[2, :])
    return vertcat(px, py, pz)


def create_gamma_vel(s_sym, s_wp, tang_wp):
    """γ_vel(θ) : ℝ → ℝ³ unit tangent on path (normalised)."""
    tx = _bspline_scalar('gamma_tan_x', s_sym, s_wp, tang_wp[0, :])
    ty = _bspline_scalar('gamma_tan_y', s_sym, s_wp, tang_wp[1, :])
    tz = _bspline_scalar('gamma_tan_z', s_sym, s_wp, tang_wp[2, :])
    tn = ca.sqrt(tx**2 + ty**2 + tz**2 + 1e-10)
    return vertcat(tx / tn, ty / tn, tz / tn)


def create_gamma_quat(s_sym, s_wp, quat_wp):
    """γ_quat(θ) : ℝ → ℝ⁴ quaternion on path (normalised)."""
    qw = _bspline_scalar('gamma_quat_w', s_sym, s_wp, quat_wp[0, :])
    qx = _bspline_scalar('gamma_quat_x', s_sym, s_wp, quat_wp[1, :])
    qy = _bspline_scalar('gamma_quat_y', s_sym, s_wp, quat_wp[2, :])
    qz = _bspline_scalar('gamma_quat_z', s_sym, s_wp, quat_wp[3, :])
    qn = ca.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + 1e-10)
    return vertcat(qw/qn, qx/qn, qy/qn, qz/qn)


def create_gamma_thrust(s_sym, s_wp, thrust_wp):
    """γ_thrust(θ) : ℝ → ℝ  desired thrust [N] from differential flatness."""
    return _bspline_scalar('gamma_thrust', s_sym, s_wp, thrust_wp)


def create_gamma_omega_hat(s_sym, s_wp, omega_hat_wp):
    """γ_ω_hat(θ) : ℝ → ℝ³  angular velocity per unit arc-length [rad/m].
    ω_d(s, v_θ) = ω_hat(s) * v_θ  (computed at runtime)."""
    wx = _bspline_scalar('gamma_omega_hat_x', s_sym, s_wp, omega_hat_wp[0, :])
    wy = _bspline_scalar('gamma_omega_hat_y', s_sym, s_wp, omega_hat_wp[1, :])
    wz = _bspline_scalar('gamma_omega_hat_z', s_sym, s_wp, omega_hat_wp[2, :])
    return vertcat(wx, wy, wz)


def create_gamma_dtube(s_sym, s_wp, d_tube_wp):
    """γ_dtube(θ) : ℝ → ℝ  variable tube radius [m] along arc-length."""
    return _bspline_scalar('gamma_dtube', s_sym, s_wp, d_tube_wp)


#  Quaternion algebra (CasADi symbolic)

def quat_error(q_real, q_desired):
    """q_err = q_real⁻¹ ⊗ q_desired"""
    EPS_M = np.finfo(np.float64).eps  # ARM-safe smoothed norm (C∞ derivative)
    norm_q = ca.sqrt(q_real[0]**2 + q_real[1]**2 + q_real[2]**2 + q_real[3]**2 + EPS_M)
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    w0, x0, y0, z0 = q_inv[0], q_inv[1], q_inv[2], q_inv[3]
    w1, x1, y1, z1 = q_desired[0], q_desired[1], q_desired[2], q_desired[3]
    return vertcat(
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    )


def quat_log(q):
    """Log(q) = 2·atan2(‖q_v‖, qw)·q_v/‖q_v‖

    ARM-safe: smoothed norm sqrt(‖q_v‖²+EPS_M) — C∞ everywhere. Bare norm_2(q_v)
    is non-smooth at q_v=0 (zero attitude error) → CasADi AD gives 0/0 NaN on ARM.
    """
    q = if_else(q[0] < 0, -q, q)
    q_w = q[0]
    q_v = q[1:]
    EPS_M = np.finfo(np.float64).eps
    norm_q_v = ca.sqrt(q_v[0]**2 + q_v[1]**2 + q_v[2]**2 + EPS_M)
    theta = atan2(norm_q_v, q_w)
    return 2.0 * q_v * theta / norm_q_v   # norm_q_v ≥ sqrt(EPS_M) > 0, no extra guard


#  Build trajectory waypoints

def build_trajectory_waypoints():
    """Load trajectory waypoints from the path planner (Phase 2).

    Source: path_planing/ref_flat_waypoints.npz, produced by the pipeline
        path_lissajous.py → path_time_3D_PMM_periodic.py →
        build_multilap_reference.py → build_flat_reference.py.

    Schema matches exactly what acados expects: waypoints are uniformly
    sampled in arc-length and share the attitude-reference convention
    ||a + g·e3|| for thrust.
    """
    pp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "path_planing")
    ref_path    = os.path.join(pp_dir, "ref_flat_waypoints.npz")
    dtube_path  = os.path.join(pp_dir, "d_tube_wp.npy")
    print(f"Loading planner reference: {ref_path}")
    d = np.load(ref_path, allow_pickle=True)

    s_wp         = d["s_wp"].astype(float)           # (N,)
    pos_wp       = d["pos_wp"].astype(float)         # (3, N)
    tang_wp      = d["tang_wp"].astype(float)        # (3, N)
    quat_wp      = d["quat_wp"].astype(float)        # (4, N) wxyz
    thrust_wp    = d["thrust_wp"].astype(float)      # (N,)    units: |a+g·e3|
    omega_hat_wp = d["omega_hat_wp"].astype(float)   # (3, N)
    s_max_plan   = float(d["s_max"][0])

    # Variable tube: gate-aware contouring radius (computed offline).
    if not os.path.isfile(dtube_path):
        raise FileNotFoundError(
            f"{dtube_path} missing. Run: cd {pp_dir} && python3 design_tube.py")
    d_tube_wp = np.load(dtube_path).astype(float)    # (N,)
    assert d_tube_wp.size == s_wp.size, \
        f"d_tube_wp size {d_tube_wp.size} ≠ s_wp size {s_wp.size}"

    # Quaternion hemisphere continuity (double-cover cleanup)
    for i in range(1, quat_wp.shape[1]):
        if np.dot(quat_wp[:, i-1], quat_wp[:, i]) < 0.0:
            quat_wp[:, i] = -quat_wp[:, i]

    # The planner reference already encodes entry + 3 laps + exit; the exit
    # phase naturally decelerates the drone back to hover. So s_max for
    # termination IS the full reference length — we do NOT subtract THETA_MARGIN.
    # The OCP keeps THETA_MARGIN only as θ upper-bound headroom (so the
    # prediction horizon can look past the end of the path, where waypoint
    # interpolation just returns the hover waypoint).
    s_max = s_max_plan
    s_wp_end = s_max_plan

    # thrust_wp is ||a + g·e3||, multiply by mass for T_d [N]
    thrust_wp = np.clip(thrust_wp * MASS, T_MIN, T_MAX)

    print(f"  N waypoints: {s_wp.size}, s_wp_end = {s_wp_end:.2f} m, "
          f"active s_max = {s_max:.2f} m (margin={THETA_MARGIN})")
    print(f"  T_d range:   [{thrust_wp.min():.2f}, {thrust_wp.max():.2f}] N  "
          f"(hover={MASS*G:.2f} N)")
    print(f"  |ω_hat|_max: {np.linalg.norm(omega_hat_wp, axis=0).max():.3f} rad/m")
    print(f"  d_tube:      [{d_tube_wp.min()*100:.1f}, {d_tube_wp.max()*100:.1f}] cm "
          f"mean={d_tube_wp.mean()*100:.1f}")
    print(f"  pos range:   x=[{pos_wp[0].min():.2f},{pos_wp[0].max():.2f}] "
          f"y=[{pos_wp[1].min():.2f},{pos_wp[1].max():.2f}] "
          f"z=[{pos_wp[2].min():.2f},{pos_wp[2].max():.2f}]")

    return s_wp, pos_wp, tang_wp, quat_wp, thrust_wp, omega_hat_wp, d_tube_wp, s_max


#  Build 15-state quadrotor + θ model

def build_mpcc_model():
    """Build 16-state rate-control quadrotor model with arc-length progress and thrust dynamics."""
    model = AcadosModel()
    model.name = "quadrotor_mpcc"

    e3 = MX([0, 0, 1])

    # States (16)
    p = MX.sym("p", 3)         # position (inertial)
    v = MX.sym("v", 3)         # velocity (inertial)
    q = MX.sym("q", 4)         # quaternion [qw, qx, qy, qz]
    w = MX.sym("w", 3)         # angular velocity (body)
    theta = MX.sym("theta")    # arc-length progress
    v_theta = MX.sym("v_theta")  # progress velocity
    f_state = MX.sym("f_state")  # thrust [N] (state, not control)
    x = vertcat(p, v, q, w, theta, v_theta, f_state)

    # Controls (5)
    delta_f = MX.sym("delta_f")   # thrust rate [N/s]
    w_cmd = MX.sym("w_cmd", 3)    # rate command [rad/s]
    a_theta = MX.sym("a_theta")   # progress acceleration [m/s²]
    u = vertcat(delta_f, w_cmd, a_theta)

    # Quaternion → rotation matrix
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    q_hat = MX.zeros(3, 3)
    q_hat[0,1] = -qz;  q_hat[0,2] =  qy
    q_hat[1,0] =  qz;  q_hat[1,2] = -qx
    q_hat[2,0] = -qy;  q_hat[2,1] =  qx
    qn = ca.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 + np.finfo(np.float64).eps)  # ARM-safe
    q_normed = q / qn
    Rot = MX.eye(3) + 2*q_hat@q_hat + 2*q_normed[0]*q_hat

    # Dynamics
    dp = v
    dv = -e3 * G + (Rot @ vertcat(MX(0), MX(0), f_state)) / MASS  # thrust from state
    # Quaternion kinematics: q̇ = ½ q ⊗ [0, ω]
    omega_quat = vertcat(MX(0), w)
    w0_, x0_, y0_, z0_ = q[0], q[1], q[2], q[3]
    w1_, x1_, y1_, z1_ = omega_quat[0], omega_quat[1], omega_quat[2], omega_quat[3]
    dq = 0.5 * vertcat(
        w0_*w1_ - x0_*x1_ - y0_*y1_ - z0_*z1_,
        w0_*x1_ + x0_*w1_ + y0_*z1_ - z0_*y1_,
        w0_*y1_ - x0_*z1_ + y0_*w1_ + z0_*x1_,
        w0_*z1_ + x0_*y1_ - y0_*x1_ + z0_*w1_,
    )
    # Rate controller: ω̇ = (ω_cmd − ω) / τ_rc
    dw = (w_cmd - w) / TAU_RC
    # Progress dynamics
    dtheta = v_theta
    dv_theta = a_theta
    # Thrust dynamics: ḟ = Δf
    df = delta_f

    f_expl = vertcat(dp, dv, dq, dw, dtheta, dv_theta, df)

    # Implicit form
    x_dot = MX.sym("x_dot", NX)
    model.f_impl_expr = x_dot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u

    return model


#  Build MPCC OCP

def build_mpcc_ocp():
    """Build the acados OCP for MPCC with embedded B-spline path."""

    # Build trajectory waypoints
    s_wp, pos_wp, tang_wp, quat_wp, thrust_wp, omega_hat_wp, d_tube_wp, s_max = build_trajectory_waypoints()

    # Build model
    ocp = AcadosOcp()
    model = build_mpcc_model()
    ocp.model = model

    # Code export directory
    code_dir = os.path.join(SCRIPT_DIR, "..", "c_generated_code_mpcc")
    ocp.code_export_directory = code_dir

    # Dimensions
    ocp.solver_options.N_horizon = N_HORIZON

    # Runtime parameters
    p_sym = MX.sym("p_runtime", N_PARAMS)
    model.p = p_sym

    Q_ec_diag  = p_sym[0:3]
    Q_el_diag  = p_sym[3:6]
    Q_q_diag   = p_sym[6:9]
    U_mat_diag = p_sym[9:13]
    Q_s        = p_sym[13]
    # p_sym[14] = vtheta_max (runtime bound, not in cost)
    W_df       = p_sym[15]   # thrust rate penalty: W_df * Δf²

    Q_ec  = casadi_diag(Q_ec_diag)
    Q_el  = casadi_diag(Q_el_diag)
    Q_q   = casadi_diag(Q_q_diag)
    U_rate = casadi_diag(U_mat_diag[1:4])   # rate command penalty

    # Cost function
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Path reference at θ (B-spline interpolation embedded in graph)
    theta_state   = model.x[13]
    v_theta_state = model.x[14]
    f_state       = model.x[15]   # thrust state

    sd      = create_gamma_pos(theta_state, s_wp, pos_wp)
    tangent = create_gamma_vel(theta_state, s_wp, tang_wp)
    qd      = create_gamma_quat(theta_state, s_wp, quat_wp)
    T_d     = create_gamma_thrust(theta_state, s_wp, thrust_wp)
    # ω_d(s, v_θ) = ω_hat(s) * v_θ — feedforward angular velocity reference
    omega_hat = create_gamma_omega_hat(theta_state, s_wp, omega_hat_wp)
    omega_d   = omega_hat * v_theta_state
    # Variable tube radius d_tube(θ) for the contouring soft constraint
    d_tube = create_gamma_dtube(theta_state, s_wp, d_tube_wp)

    # Quaternion error: q_err = q_real⁻¹ ⊗ q_desired, then log map
    # NOTE: q_d(s) is precomputed at fixed speed (15 m/s), approximate at other v_θ.
    # Kept as soft guidance (low Q_q) to prevent degenerate orientations.
    # The angular rate feedforward ω_d(s,v_θ) provides speed-correct rate tracking.
    q_err = quat_error(model.x[6:10], qd)
    log_q = quat_log(q_err)

    # Contouring/lag decomposition
    e_t          = sd - model.x[0:3]
    e_lag_scalar = dot(tangent, e_t)
    e_lag        = e_lag_scalar * tangent
    P_ec         = MX.eye(3) - tangent @ tangent.T
    ec           = P_ec @ e_t

    # Control cost — ω_d(s,v_θ) feedforward, thrust rate smoothing
    delta_f_ctrl = model.u[0]                   # Δf = thrust rate [N/s]
    rates_err    = model.u[1:4] - omega_d       # ω_cmd still at u[1:4]

    control_cost        = (W_df * delta_f_ctrl**2
                           + rates_err.T @ U_rate @ rates_err)
    attitude_cost       = log_q.T @ Q_q @ log_q
    contour_cost        = ec.T @ Q_ec @ ec
    lag_cost            = e_lag.T @ Q_el @ e_lag
    progress_cost       = -Q_s * v_theta_state

    # Stage cost
    ocp.model.cost_expr_ext_cost = (
        contour_cost + lag_cost + attitude_cost
        + control_cost + progress_cost
    )
    # Terminal cost (no control, no progress)
    ocp.model.cost_expr_ext_cost_e = (
        contour_cost + lag_cost + attitude_cost
    )

    # Default parameter values
    p_default = np.zeros(N_PARAMS)
    p_default[0:3]  = [100.0, 100.0, 100.0]   # Q_ec
    p_default[3:6]  = [13.0, 13.0, 13.0]      # Q_el
    p_default[6:9]  = [0.5, 0.5, 0.5]         # Q_q
    p_default[9:13] = [0.1, 0.3, 0.3, 0.3]    # U_mat
    p_default[13]   = 15.0                      # Q_s
    p_default[14]   = VTHETA_MAX                # vtheta_max
    p_default[15]   = 0.001                      # W_df (thrust rate penalty)
    ocp.parameter_values = p_default

    # Constraints
    # Input bounds: u = [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
    ocp.constraints.lbu = np.array([-DF_MAX, -W_MAX, -W_MAX, -W_MAX, ATHETA_MIN])
    ocp.constraints.ubu = np.array([ DF_MAX,  W_MAX,  W_MAX,  W_MAX, ATHETA_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # State bounds on pz, θ, v_θ, and f (thrust)
    #   pz: keep drone off the floor — critical when the contouring tube
    #   extends vertically below z_min near low gates (e.g., z_gate≈0.8 m).
    Z_MIN = 0.20      # [m]  hard floor margin (MuJoCo's floor crash at 0.15)
    Z_MAX = 5.0       # [m]  ceiling (as in SilConfig)
    ocp.constraints.lbx = np.array([Z_MIN, 0.0, VTHETA_MIN, T_MIN])
    ocp.constraints.ubx = np.array([Z_MAX, s_max + THETA_MARGIN, VTHETA_MAX, T_MAX])
    ocp.constraints.idxbx = np.array([2, 13, 14, 15])   # pz, θ, v_θ, f

    # Initial state
    x0 = np.zeros(NX)
    x0[0] = 2.5; x0[2] = 1.5; x0[6] = 1.0  # pos + identity quat
    x0[15] = MASS * G                         # hover thrust
    ocp.constraints.x0 = x0

    # Soft h-constraints (slack-handled):
    #   1) Lag:   |e_lag_scalar| ≤ D_MAX_LAG
    #   2) Tube:  ||e_c||² ≤ d_tube(θ)²   →   (||e_c||² - d_tube²) ≤ 0
    #      (one-sided; lh set to a large negative so it's effectively unbounded below)
    ec_norm_sq     = ec.T @ ec              # scalar  (||e_c||²)
    tube_violation = ec_norm_sq - d_tube**2

    ocp.model.con_h_expr   = vertcat(e_lag_scalar, tube_violation)
    ocp.model.con_h_expr_e = vertcat(e_lag_scalar, tube_violation)
    ocp.dims.nh   = 2
    ocp.dims.nh_e = 2
    # Lag: [-D_MAX_LAG, +D_MAX_LAG]. Tube: [-inf, 0] emulated with a large negative.
    TUBE_UNBOUNDED = -1e2
    ocp.constraints.lh   = np.array([-D_MAX_LAG, TUBE_UNBOUNDED])
    ocp.constraints.uh   = np.array([ D_MAX_LAG, 0.0])
    ocp.constraints.lh_e = np.array([-D_MAX_LAG, TUBE_UNBOUNDED])
    ocp.constraints.uh_e = np.array([ D_MAX_LAG, 0.0])

    # Soft slack: high L2 penalty keeps constraints bounded but allows transients
    #   index 0 = lag       (existing weights: 200 / 200)
    #   index 1 = tube      (heavier: 500 / 500 — gate-compliance should be strict)
    W_LAG = 200.0
    W_TUBE = 1500.0   # stricter tube respect — max_miss drops noticeably
    ocp.cost.Zl   = np.array([W_LAG, W_TUBE])
    ocp.cost.Zu   = np.array([W_LAG, W_TUBE])
    ocp.cost.zl   = np.array([W_LAG, W_TUBE])
    ocp.cost.zu   = np.array([W_LAG, W_TUBE])
    ocp.cost.Zl_e = np.array([W_LAG, W_TUBE])
    ocp.cost.Zu_e = np.array([W_LAG, W_TUBE])
    ocp.cost.zl_e = np.array([W_LAG, W_TUBE])
    ocp.cost.zu_e = np.array([W_LAG, W_TUBE])
    ocp.constraints.idxsh   = np.array([0, 1])   # both lag and tube are soft
    ocp.constraints.idxsh_e = np.array([0, 1])

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = min(N_HORIZON // 5, 20)
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = T_HORIZON
    ocp.solver_options.time_steps = TIME_STEPS
    ocp.solver_options.sim_method_num_steps = SIM_NUM_STEPS

    return ocp


def main():
    ocp = build_mpcc_ocp()
    code_dir = ocp.code_export_directory

    # Clean previous build
    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)

    json_file = os.path.join(os.path.dirname(code_dir),
                              f"acados_ocp_{ocp.model.name}.json")
    if os.path.isfile(json_file):
        os.remove(json_file)

    print(f"\nGenerating MPCC solver C code in {code_dir} ...")
    print(f"  NX={NX}, NU={NU}, N={N_HORIZON}, T={T_HORIZON}s, dt_min={TIME_STEPS.min()}s, dt_max={TIME_STEPS.max()}s")
    print(f"  N_PARAMS={N_PARAMS}, D_MAX_LAG={D_MAX_LAG}m")

    solver = AcadosOcpSolver(ocp, json_file=json_file)

    print("\nDone! Generated files:")
    for f in sorted(os.listdir(code_dir)):
        print(f"  {f}")
    print(f"\nJSON: {json_file}")
    print(f"\nNow run: cd ../build && cmake .. && make -j$(nproc)")


if __name__ == "__main__":
    main()
