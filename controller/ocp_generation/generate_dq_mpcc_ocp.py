"""
Generate acados C code for DQ-MPCC (dual quaternion SE(3) contouring control).

State  x ∈ ℝ¹⁷ = [dq(8), ω(3), v_b(3), θ, v_θ, f]
    dq   = [qr_w, qr_x, qr_y, qr_z, qd_w, qd_x, qd_y, qd_z]
    ω    = body angular velocity [rad/s]
    v_b  = body-frame linear velocity [m/s]
    θ    = arc-length progress [m]
    v_θ  = progress velocity [m/s]
    f    = thrust [N] (state, rate-limited)

Control u ∈ ℝ⁵  = [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]

Dynamics:
    DQ kinematics:  Q̇ = ½ H_dual · Ξ + K·correction
    Rate controller: ω̇ = (ω_cmd − ω) / τ_rc
    Body-frame:     v̇_b = v_b × ω − R⁻¹ g e₃ + (f/m) e₃
    Progress:       θ̇ = v_θ,  v̇_θ = a_θ
    Thrust:         ḟ = Δf

Cost: SE(3) log-map with J_l⁻¹ rotation-translation coupling.
    ℓ = φᵀ Q_φ φ + ρ_cᵀ Q_ec ρ_c + ρ_lᵀ Q_el ρ_l − Q_s v_θ + uᵀ U u

Runtime params p ∈ ℝ¹⁶ (no rebuild for weight changes):
    p[ 0: 3] = Q_phi    rotation-log error weights
    p[ 3: 6] = Q_ec     contouring error weights
    p[ 6: 9] = Q_el     lag error weights
    p[ 9:13] = U_mat    control effort [f_err, ωx, ωy, ωz]
    p[13]    = Q_s      progress weight
    p[14]    = vtheta_max
    p[15]    = W_df     thrust rate penalty: W_df * Δf²

Run once:  python3 generate_dq_mpcc_ocp.py    # exact translational coordinate, Eq. (21) of the article
Output:    ../c_generated_code_dq_mpcc/
"""

import os
import shutil
import numpy as np
import casadi as ca
from casadi import MX, DM, vertcat, horzcat, norm_2, if_else, atan2, dot
from casadi import diag as casadi_diag

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
NX = 17   # [dq(8), ω(3), v_b(3), θ, v_θ, f]
NU = 5    # [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
DT_CONTROL = 0.01    # [s] control loop period (100 Hz)
T_HORIZON  = 1.48    # [s] prediction horizon (non-uniform grid, N=31)

# 3-tier with smaller max-ratio (4× instead of 13×) — better SE(3) Hessian
# conditioning for DQ Lie cost. Same grid as MPCC + ablation for fair comparison.
TIME_STEPS = np.concatenate([
    np.full(10, 0.020),  # 10 × 20ms = 0.200s
    np.full(10, 0.040),  # 10 × 40ms = 0.400s
    np.full(11, 0.080),  # 11 × 80ms = 0.880s
])
N_HORIZON = len(TIME_STEPS)  # = 31
assert abs(TIME_STEPS.sum() - T_HORIZON) < 1e-9, f"Time steps sum {TIME_STEPS.sum()} != T_HORIZON {T_HORIZON}"

SIM_NUM_STEPS = np.concatenate([
    np.full(10, 2),      # 20ms → 2 sub-steps (10ms each)
    np.full(10, 3),      # 40ms → 3 sub-steps (~13ms)
    np.full(11, 5),      # 80ms → 5 sub-steps (16ms)
]).astype(int)
N_PARAMS   = 16


#  Quaternion Hamilton product (H+ matrix)

def H_plus_matrix(q):
    """Left-multiplication Hamilton matrix: H+(q) · p = q ⊗ p."""
    return vertcat(
        horzcat(q[0], -q[1], -q[2], -q[3]),
        horzcat(q[1],  q[0], -q[3],  q[2]),
        horzcat(q[2],  q[3],  q[0], -q[1]),
        horzcat(q[3], -q[2],  q[1],  q[0]),
    )


def quat_product(p, q):
    """Hamilton product p ⊗ q."""
    return H_plus_matrix(p) @ q


#  Dual quaternion operations

def dq_from_pose(quat, trans):
    """Build unit DQ from rotation quaternion + translation.
    Q = q_r + ε (½) t ⊗ q_r
    """
    t_pure = vertcat(MX(0), trans)
    q_d = 0.5 * quat_product(t_pure, quat)
    return vertcat(quat, q_d)


def dq_conjugate(dq):
    """DQ conjugate: Q* = conj(Qr) + ε conj(Qd)."""
    return vertcat(
        dq[0], -dq[1], -dq[2], -dq[3],
        dq[4], -dq[5], -dq[6], -dq[7],
    )


def dq_product(dq1, dq2):
    """Dual quaternion product dq1 * dq2."""
    q1r, q1d = dq1[0:4], dq1[4:8]
    q2r, q2d = dq2[0:4], dq2[4:8]
    real = quat_product(q1r, q2r)
    dual = quat_product(q1r, q2d) + quat_product(q1d, q2r)
    return vertcat(real, dual)


def dq_error(dq_desired, dq_actual):
    """DQ error: Q_err = Q_d* ⊗ Q."""
    dq_d_conj = dq_conjugate(dq_desired)
    qr_c = dq_d_conj[0:4]
    qd_c = dq_d_conj[4:8]
    H_r = H_plus_matrix(qr_c)
    H_d = H_plus_matrix(qd_c)
    zeros = DM.zeros(4, 4)
    H_dual = vertcat(
        horzcat(H_r, zeros),
        horzcat(H_d, H_r),
    )
    return H_dual @ dq_actual


def dq_get_position(dq):
    """Extract position [tx, ty, tz] from unit DQ: t = 2·Qd ⊗ Qr*."""
    qr = dq[0:4]
    qd = dq[4:8]
    qr_c = vertcat(qr[0], -qr[1], -qr[2], -qr[3])
    t = 2.0 * quat_product(qd, qr_c)
    return t[1:4]


#  Rotation helpers

def rotation_inverse_expr(quat, vector):
    """Rotate inertial → body: v_body = q* ⊗ [0,v] ⊗ q."""
    v_pure = vertcat(MX(0), vector)
    qc = vertcat(quat[0], -quat[1], -quat[2], -quat[3])
    tmp = quat_product(qc, v_pure)
    result = quat_product(tmp, quat)
    return result[1:4]


#  SE(3) logarithmic map

def left_jacobian_SO3_inv(phi):
    """Analytical J_l(φ)⁻¹ (Solà 2018). ARM-safe: smoothed norm avoids NaN AD at φ=0."""
    EPS_M = np.finfo(np.float64).eps
    eps = 1e-7
    I3 = DM.eye(3)

    # Smoothed norm: sqrt(‖φ‖² + EPS²) — C∞ everywhere, derivative defined at φ=0.
    # Bare norm_2(phi) is non-smooth at origin → CasADi AD produces 0/0 NaN
    # in the inactive if_else branch on ARM targets.
    theta = ca.sqrt(phi[0]**2 + phi[1]**2 + phi[2]**2 + EPS_M)

    phi_x = vertcat(
        horzcat(MX(0),  -phi[2],  phi[1]),
        horzcat(phi[2],  MX(0),  -phi[0]),
        horzcat(-phi[1], phi[0],  MX(0)),
    )

    n_hat = phi / theta
    n_hat_col = ca.reshape(n_hat, 3, 1)
    nnT = n_hat_col @ n_hat_col.T
    n_hat_x = phi_x / theta

    half_theta = 0.5 * theta
    alpha = half_theta * ca.cos(half_theta) / (ca.sin(half_theta) + EPS_M)

    # Removed if_else: smoothed norm theta = sqrt(|phi|²+EPS_M) avoids singularity
    # naturally, AND if_else itself is non-smooth piecewise → ARM AD edge cases.
    J_l_inv = alpha * I3 - half_theta * n_hat_x + (1 - alpha) * nnT
    return J_l_inv


def ln_dual(dq_err):
    """Pose-error coordinate [φ; ρ] ∈ ℝ⁶.

    φ = Log(q_r) = 2·atan2(‖q_v‖, q_w) · q_v/‖q_v‖ = θ·n   (rotation vector; Barfoot, Sec. 6.2)
    ρ = J_l⁻¹(φ) · t_err                                   (exact translational coordinate, Eq. (21) of the article)
    Deployed controller (generate_dq_mpcc_ablation_ocp.py): J_l⁻¹ ≈ I  →  ρ = t_err.
    """
    sign = if_else(dq_err[0] < 0, MX(-1.0), MX(1.0))
    q_real = sign * dq_err[0:4]
    q_dual = sign * dq_err[4:8]

    q_real_c = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3])

    EPS_M = np.finfo(np.float64).eps  # ARM-stable smoothed norm
    # Squared+EPS pattern: C∞ derivative everywhere, no 0/0 NaN at q_v=0
    norm_v = ca.sqrt(q_real[1]**2 + q_real[2]**2 + q_real[3]**2 + EPS_M)
    angle = 2.0 * atan2(norm_v, q_real[0])
    phi = vertcat(                                    # phi = Log(q_r) = theta*n, the rotation vector
        angle * q_real[1] / norm_v,
        angle * q_real[2] / norm_v,
        angle * q_real[3] / norm_v,
    )

    t_err = 2.0 * quat_product(q_dual, q_real_c)
    t_vec = t_err[1:4]

    J_l_inv = left_jacobian_SO3_inv(phi)             # evaluated at the rotation vector, Eq. (21)
    rho = J_l_inv @ t_vec

    return vertcat(phi, rho)


#  B-spline interpolation (CasADi symbolic)

def _bspline_scalar(name, s_sym, s_wp, values_wp):
    """Clamped scalar B-spline interpolant."""
    s_c = ca.fmin(ca.fmax(s_sym, float(s_wp[0])), float(s_wp[-1]))
    lut = ca.interpolant(
        name, "bspline",
        [np.asarray(s_wp, dtype=float)],
        np.asarray(values_wp, dtype=float).reshape(-1),
    )
    return lut(s_c)


def create_gamma_pos(s_sym, s_wp, pos_wp):
    """γ_pos(θ) : ℝ → ℝ³."""
    px = _bspline_scalar('dq_gamma_pos_x', s_sym, s_wp, pos_wp[0, :])
    py = _bspline_scalar('dq_gamma_pos_y', s_sym, s_wp, pos_wp[1, :])
    pz = _bspline_scalar('dq_gamma_pos_z', s_sym, s_wp, pos_wp[2, :])
    return vertcat(px, py, pz)


def create_gamma_vel(s_sym, s_wp, tang_wp):
    """γ_vel(θ) : ℝ → ℝ³ unit tangent (normalised)."""
    tx = _bspline_scalar('dq_gamma_tan_x', s_sym, s_wp, tang_wp[0, :])
    ty = _bspline_scalar('dq_gamma_tan_y', s_sym, s_wp, tang_wp[1, :])
    tz = _bspline_scalar('dq_gamma_tan_z', s_sym, s_wp, tang_wp[2, :])
    tn = ca.sqrt(tx**2 + ty**2 + tz**2 + 1e-10)
    return vertcat(tx / tn, ty / tn, tz / tn)


def create_gamma_quat(s_sym, s_wp, quat_wp):
    """γ_quat(θ) : ℝ → ℝ⁴ (normalised)."""
    qw = _bspline_scalar('dq_gamma_quat_w', s_sym, s_wp, quat_wp[0, :])
    qx = _bspline_scalar('dq_gamma_quat_x', s_sym, s_wp, quat_wp[1, :])
    qy = _bspline_scalar('dq_gamma_quat_y', s_sym, s_wp, quat_wp[2, :])
    qz = _bspline_scalar('dq_gamma_quat_z', s_sym, s_wp, quat_wp[3, :])
    qn = ca.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + 1e-10)
    return vertcat(qw/qn, qx/qn, qy/qn, qz/qn)


def create_gamma_thrust(s_sym, s_wp, thrust_wp):
    """γ_thrust(θ) : ℝ → ℝ  desired thrust [N] from differential flatness."""
    return _bspline_scalar('dq_gamma_thrust', s_sym, s_wp, thrust_wp)


def create_gamma_omega_hat(s_sym, s_wp, omega_hat_wp):
    """γ_ω_hat(θ) : ℝ → ℝ³  angular velocity per unit arc-length [rad/m].
    ω_d(s, v_θ) = ω_hat(s) * v_θ  (computed at runtime)."""
    wx = _bspline_scalar('dq_gamma_omega_hat_x', s_sym, s_wp, omega_hat_wp[0, :])
    wy = _bspline_scalar('dq_gamma_omega_hat_y', s_sym, s_wp, omega_hat_wp[1, :])
    wz = _bspline_scalar('dq_gamma_omega_hat_z', s_sym, s_wp, omega_hat_wp[2, :])
    return vertcat(wx, wy, wz)


def create_gamma_dtube(s_sym, s_wp, d_tube_wp):
    """γ_dtube(θ) : ℝ → ℝ  variable tube radius [m] along arc-length."""
    return _bspline_scalar('dq_gamma_dtube', s_sym, s_wp, d_tube_wp)


#  Build trajectory waypoints

def build_trajectory_waypoints():
    """Load trajectory waypoints from the path planner (Phase 2).

    Source: path_planing/ref_flat_waypoints.npz (produced by the pipeline
    path_lissajous → path_time_3D_PMM_periodic → build_multilap_reference
    → build_flat_reference). Already in arc-length; matches the schema
    acados expects.
    """
    pp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "path_planing")
    ref_path   = os.path.join(pp_dir, "ref_flat_waypoints.npz")
    dtube_path = os.path.join(pp_dir, "d_tube_wp.npy")
    print(f"Loading planner reference: {ref_path}")
    d = np.load(ref_path, allow_pickle=True)

    s_wp         = d["s_wp"].astype(float)
    pos_wp       = d["pos_wp"].astype(float)
    tang_wp      = d["tang_wp"].astype(float)
    quat_wp      = d["quat_wp"].astype(float)
    thrust_wp    = d["thrust_wp"].astype(float)
    omega_hat_wp = d["omega_hat_wp"].astype(float)
    s_max_plan   = float(d["s_max"][0])

    # Variable tube (gate-aware smoothstep profile)
    if not os.path.isfile(dtube_path):
        raise FileNotFoundError(
            f"{dtube_path} missing. Run: cd {pp_dir} && python3 design_tube.py")
    d_tube_wp = np.load(dtube_path).astype(float)
    assert d_tube_wp.size == s_wp.size

    # Quaternion hemisphere continuity
    for i in range(1, quat_wp.shape[1]):
        if np.dot(quat_wp[:, i-1], quat_wp[:, i]) < 0.0:
            quat_wp[:, i] = -quat_wp[:, i]

    # Full-reference length (entry + 3 laps + exit — the exit decelerates to hover).
    s_max    = s_max_plan
    s_wp_end = s_max_plan

    thrust_wp = np.clip(thrust_wp * MASS, T_MIN, T_MAX)

    print(f"  N waypoints: {s_wp.size}, s_max = {s_max:.2f} m (margin={THETA_MARGIN})")
    print(f"  T_d range:   [{thrust_wp.min():.2f}, {thrust_wp.max():.2f}] N  "
          f"(hover={MASS*G:.2f} N)")
    print(f"  |ω_hat|_max: {np.linalg.norm(omega_hat_wp, axis=0).max():.3f} rad/m")
    print(f"  d_tube:      [{d_tube_wp.min()*100:.1f}, {d_tube_wp.max()*100:.1f}] cm "
          f"mean={d_tube_wp.mean()*100:.1f}")

    return s_wp, pos_wp, tang_wp, quat_wp, thrust_wp, omega_hat_wp, d_tube_wp, s_max


#  Build 16-state DQ quadrotor model

def build_dq_mpcc_model():
    """Build 17-state DQ rate-control model with arc-length progress and thrust dynamics."""
    model = AcadosModel()
    model.name = "quadrotor_dq_mpcc"

    e3 = MX([0, 0, 1])

    # States (17)
    dq = MX.sym("dq", 8)       # dual quaternion [qr(4), qd(4)]
    w  = MX.sym("w", 3)        # angular velocity (body) [rad/s]
    vb = MX.sym("vb", 3)       # linear velocity (body) [m/s]
    theta   = MX.sym("theta")     # arc-length progress [m]
    v_theta = MX.sym("v_theta")   # progress velocity [m/s]
    f_state = MX.sym("f_state")   # thrust [N] (state, not control)
    x = vertcat(dq, w, vb, theta, v_theta, f_state)

    # Controls (5)
    delta_f = MX.sym("delta_f")      # thrust rate [N/s]
    w_cmd   = MX.sym("w_cmd", 3)     # rate command [rad/s]
    a_theta = MX.sym("a_theta")      # progress acceleration [m/s²]
    u = vertcat(delta_f, w_cmd, a_theta)

    # DQ kinematics: Q̇ = ½ H_dual · Ξ + K·correction
    qr = dq[0:4]
    qd_part = dq[4:8]

    K_quat = 10.0
    # Smoothed norm — qr drifts near norm 1 but ARM AD safer with squared form
    norm_r = ca.sqrt(qr[0]**2 + qr[1]**2 + qr[2]**2 + qr[3]**2 + np.finfo(np.float64).eps)
    quat_err = 1.0 - norm_r
    correction = vertcat(qr * (K_quat * quat_err),
                         qd_part * MX(0))

    H_r = H_plus_matrix(qr)
    H_d = H_plus_matrix(qd_part)
    zeros = DM.zeros(4, 4)
    H_dual = vertcat(
        horzcat(H_r, zeros),
        horzcat(H_d, H_r),
    )

    # Twist as 8-vector: [0, ωx, ωy, ωz, 0, vx, vy, vz]
    twist_dq = vertcat(
        MX(0), w[0], w[1], w[2],
        MX(0), vb[0], vb[1], vb[2],
    )
    dq_dot = 0.5 * (H_dual @ twist_dq) + correction

    # Angular dynamics: rate-control lag
    dw = (w_cmd - w) / TAU_RC

    # Body-frame translational dynamics
    # v̇_b = v_b × ω − R⁻¹ g e₃ + (f/m) e₃   (thrust from state)
    g_body = rotation_inverse_expr(qr, G * e3)
    dvb = ca.cross(vb, w) - g_body + (f_state / MASS) * e3

    # Progress dynamics
    dtheta   = v_theta
    dv_theta = a_theta
    # Thrust dynamics
    df = delta_f

    f_expl = vertcat(dq_dot, dw, dvb, dtheta, dv_theta, df)

    # Implicit form
    x_dot = MX.sym("x_dot", NX)
    model.f_impl_expr = x_dot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u

    return model


#  Build DQ-MPCC OCP

def build_dq_mpcc_ocp():
    """Build the acados OCP for DQ-MPCC with SE(3) cost and embedded B-spline."""

    # Trajectory
    s_wp, pos_wp, tang_wp, quat_wp, thrust_wp, omega_hat_wp, d_tube_wp, s_max = build_trajectory_waypoints()

    # Model
    ocp = AcadosOcp()
    model = build_dq_mpcc_model()
    ocp.model = model

    code_dir = os.path.join(SCRIPT_DIR, "..", "c_generated_code_dq_mpcc")
    ocp.code_export_directory = code_dir
    ocp.solver_options.N_horizon = N_HORIZON

    # Runtime parameters
    p_sym = MX.sym("p_runtime", N_PARAMS)
    model.p = p_sym

    Q_phi_diag = p_sym[0:3]
    Q_ec_diag  = p_sym[3:6]
    Q_el_diag  = p_sym[6:9]
    U_mat_diag = p_sym[9:13]
    Q_s        = p_sym[13]
    # p_sym[14] = vtheta_max (runtime bound)
    W_df       = p_sym[15]   # thrust rate penalty: W_df * Δf²

    Q_phi  = casadi_diag(Q_phi_diag)
    Q_ec   = casadi_diag(Q_ec_diag)
    Q_el   = casadi_diag(Q_el_diag)
    U_rate = casadi_diag(U_mat_diag[1:4])

    # SE(3) cost function
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    theta_state   = model.x[14]   # θ at index 14 in 17-state
    v_theta_state = model.x[15]   # v_θ at index 15
    f_state       = model.x[16]   # f at index 16 (thrust state)

    # Path reference at θ (B-spline)
    sd      = create_gamma_pos(theta_state, s_wp, pos_wp)
    tangent = create_gamma_vel(theta_state, s_wp, tang_wp)
    qd      = create_gamma_quat(theta_state, s_wp, quat_wp)
    T_d     = create_gamma_thrust(theta_state, s_wp, thrust_wp)
    # ω_d(s, v_θ) = ω_hat(s) * v_θ — feedforward angular velocity reference
    omega_hat = create_gamma_omega_hat(theta_state, s_wp, omega_hat_wp)
    omega_d   = omega_hat * v_theta_state

    # Build desired DQ and compute SE(3) error
    dq_desired = dq_from_pose(qd, sd)
    dq_actual  = model.x[0:8]
    dq_err     = dq_error(dq_desired, dq_actual)
    log_err    = ln_dual(dq_err)

    phi = log_err[0:3]   # rotation error [rad]
    rho = log_err[3:6]   # translation error (body frame, with J_l⁻¹ coupling)

    # Lag/contouring decomposition in desired body frame
    tangent_body = rotation_inverse_expr(qd, tangent)  # τ_body = R_d⁻¹ · τ
    proj_scalar  = dot(tangent_body, rho)
    rho_lag  = proj_scalar * tangent_body
    rho_cont = rho - rho_lag

    # Physical lag for h-constraint (R³, same semantics as MPCC)
    p_drone_phys = dq_get_position(dq_actual)
    e_t_phys     = sd - p_drone_phys
    e_lag_phys   = dot(tangent, e_t_phys)

    # Control cost — ω_d(s,v_θ) feedforward, thrust rate smoothing
    # T_d(s) is not in the cost: it is precomputed at a fixed speed (15 m/s),
    # whereas v_θ is optimized online.
    delta_f_ctrl = model.u[0]               # Δf = thrust rate [N/s]
    rates_err    = model.u[1:4] - omega_d   # ω_cmd still at u[1:4]

    orientation_cost    = phi.T @ Q_phi @ phi
    contouring_cost     = rho_cont.T @ Q_ec @ rho_cont
    lag_cost            = rho_lag.T @ Q_el @ rho_lag
    control_cost        = (W_df * delta_f_ctrl**2
                           + rates_err.T @ U_rate @ rates_err)
    progress_cost       = -Q_s * v_theta_state

    # Stage cost
    ocp.model.cost_expr_ext_cost = (
        orientation_cost + contouring_cost + lag_cost
        + control_cost + progress_cost
    )
    # Terminal cost (no control, no progress)
    ocp.model.cost_expr_ext_cost_e = (
        orientation_cost + contouring_cost + lag_cost
    )

    # Default parameter values
    p_default = np.zeros(N_PARAMS)
    p_default[0:3]  = [5.0, 5.0, 5.0]          # Q_phi
    p_default[3:6]  = [100.0, 100.0, 100.0]    # Q_ec
    p_default[6:9]  = [10.0, 10.0, 10.0]       # Q_el
    p_default[9:13] = [0.1, 0.3, 0.3, 0.3]     # U_mat
    p_default[13]   = 12.0                       # Q_s
    p_default[14]   = VTHETA_MAX                 # vtheta_max
    p_default[15]   = 0.001                       # W_df (thrust rate penalty)
    ocp.parameter_values = p_default

    # Constraints
    # Input bounds: u = [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
    ocp.constraints.lbu = np.array([-DF_MAX, -W_MAX, -W_MAX, -W_MAX, ATHETA_MIN])
    ocp.constraints.ubu = np.array([ DF_MAX,  W_MAX,  W_MAX,  W_MAX, ATHETA_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # State bounds: θ (idx 14), v_θ (idx 15), f (idx 16)
    # pz is NOT a direct state in DQ (position is encoded in the dual part);
    # we add it as a soft h-constraint below via dq_get_position.
    ocp.constraints.lbx = np.array([0.0, VTHETA_MIN, T_MIN])
    ocp.constraints.ubx = np.array([s_max + THETA_MARGIN, VTHETA_MAX, T_MAX])
    ocp.constraints.idxbx = np.array([14, 15, 16])

    # Initial state: identity DQ at start position
    x0 = np.zeros(NX)
    x0[0] = 1.0   # qr_w = 1 (identity rotation)
    # qd = ½ t ⊗ q_r = ½ [0, 2.5, 0, 1.5] ⊗ [1, 0, 0, 0] = [0, 1.25, 0, 0.75]
    x0[4] = 0.0    # qd_w
    x0[5] = 1.25   # qd_x = 0.5 * 2.5
    x0[6] = 0.0    # qd_y
    x0[7] = 0.75   # qd_z = 0.5 * 1.5
    x0[16] = MASS * G  # hover thrust
    ocp.constraints.x0 = x0

    # Soft h-constraints (slack-handled)
    #   1) Lag:   |e_lag_phys| ≤ D_MAX_LAG
    #   2) Tube:  ||ρ_cont||² ≤ d_tube(θ)²  →  (‖ρ_cont‖² − d_tube²) ≤ 0
    #   3) Floor: pz ≥ Z_MIN                →  (Z_MIN − pz)         ≤ 0
    d_tube = create_gamma_dtube(theta_state, s_wp, d_tube_wp)
    rho_cont_norm_sq = rho_cont.T @ rho_cont
    tube_violation   = rho_cont_norm_sq - d_tube**2
    pz               = dq_get_position(dq_actual)[2]
    Z_MIN = 0.20
    floor_violation  = Z_MIN - pz

    TUBE_UNBOUNDED = -1e2
    ocp.model.con_h_expr   = vertcat(e_lag_phys, tube_violation, floor_violation)
    ocp.model.con_h_expr_e = vertcat(e_lag_phys, tube_violation, floor_violation)
    ocp.dims.nh   = 3
    ocp.dims.nh_e = 3
    ocp.constraints.lh   = np.array([-D_MAX_LAG, TUBE_UNBOUNDED, TUBE_UNBOUNDED])
    ocp.constraints.uh   = np.array([ D_MAX_LAG, 0.0,            0.0])
    ocp.constraints.lh_e = np.array([-D_MAX_LAG, TUBE_UNBOUNDED, TUBE_UNBOUNDED])
    ocp.constraints.uh_e = np.array([ D_MAX_LAG, 0.0,            0.0])

    # Soft slack: idx 0 = lag, idx 1 = tube, idx 2 = floor
    W_LAG   = 200.0
    W_TUBE  = 1500.0
    W_FLOOR = 1000.0
    w_arr = np.array([W_LAG, W_TUBE, W_FLOOR])
    ocp.cost.Zl   = w_arr.copy(); ocp.cost.Zu   = w_arr.copy()
    ocp.cost.zl   = w_arr.copy(); ocp.cost.zu   = w_arr.copy()
    ocp.cost.Zl_e = w_arr.copy(); ocp.cost.Zu_e = w_arr.copy()
    ocp.cost.zl_e = w_arr.copy(); ocp.cost.zu_e = w_arr.copy()
    ocp.constraints.idxsh   = np.array([0, 1, 2])
    ocp.constraints.idxsh_e = np.array([0, 1, 2])

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = min(N_HORIZON // 5, 20)
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 2
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "PROJECT"
    ocp.solver_options.levenberg_marquardt = 1e-2  # racing baseline (matches tuning JSON; ARM-stable when U_w ≥ 0.3)
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.tf = T_HORIZON
    ocp.solver_options.time_steps = TIME_STEPS
    ocp.solver_options.sim_method_num_steps = SIM_NUM_STEPS

    return ocp


def main():
    ocp = build_dq_mpcc_ocp()
    code_dir = ocp.code_export_directory

    if os.path.isdir(code_dir):
        shutil.rmtree(code_dir)

    json_file = os.path.join(os.path.dirname(code_dir),
                              f"acados_ocp_{ocp.model.name}.json")
    if os.path.isfile(json_file):
        os.remove(json_file)

    print(f"\nGenerating DQ-MPCC solver C code in {code_dir} ...")
    print(f"  NX={NX}, NU={NU}, N={N_HORIZON}, T={T_HORIZON}s, dt_min={TIME_STEPS.min()}s, dt_max={TIME_STEPS.max()}s")
    print(f"  N_PARAMS={N_PARAMS}, D_MAX_LAG={D_MAX_LAG}m")
    print(f"  SE(3) cost with J_l⁻¹ coupling, PROJECT+LM regularization")

    solver = AcadosOcpSolver(ocp, json_file=json_file)

    print("\nDone! Generated files:")
    for f in sorted(os.listdir(code_dir)):
        print(f"  {f}")
    print(f"\nJSON: {json_file}")
    print(f"\nNow run: cd ../build && cmake .. && make -j$(nproc)")


if __name__ == "__main__":
    main()
