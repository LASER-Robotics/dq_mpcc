"""
path_utils.py — Arc-length parameterisation and waypoint generation.

Self-contained: no imports from the parent Python project.
Used only by generate_*_ocp.py to build B-spline waypoints for acados.
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid


def build_arc_length_parameterisation(
    xd, yd, zd,
    xd_p, yd_p, zd_p,
    t_range: np.ndarray,
):
    """Build cubic-spline arc-length mapping for a parametric curve.

    Returns
    -------
    arc_lengths      : ndarray (M,)
    positions        : ndarray (3, M)
    position_by_arc  : callable (s) -> ndarray (3,)
    tangent_by_arc   : callable (s) -> ndarray (3,)
    s_max            : float
    """
    speeds = np.linalg.norm(
        np.column_stack([xd_p(t_range), yd_p(t_range), zd_p(t_range)]),
        axis=1,
    )
    arc_lengths = np.concatenate(
        [[0.0], cumulative_trapezoid(speeds, t_range)]
    )

    positions = np.column_stack([xd(t_range), yd(t_range), zd(t_range)]).T  # (3, M)

    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    def position_by_arc(s: float) -> np.ndarray:
        s = np.clip(s, arc_lengths[0], arc_lengths[-1])
        te = spline_t(s)
        return np.array([spline_x(te), spline_y(te), spline_z(te)])

    def tangent_by_arc(s: float, ds: float = 1e-4) -> np.ndarray:
        s_lo = np.clip(s - ds, arc_lengths[0], arc_lengths[-1])
        s_hi = np.clip(s + ds, arc_lengths[0], arc_lengths[-1])
        tang = (position_by_arc(s_hi) - position_by_arc(s_lo)) / (s_hi - s_lo + 1e-10)
        norm = np.linalg.norm(tang)
        return tang / norm if norm > 1e-8 else tang

    return arc_lengths, positions, position_by_arc, tangent_by_arc, arc_lengths[-1]


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [qw, qx, qy, qz]."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_hemisphere_correction(quats: np.ndarray) -> np.ndarray:
    """Ensure consecutive quaternions lie in the same hemisphere."""
    q = quats.copy()
    for i in range(1, q.shape[1]):
        if np.dot(q[:, i], q[:, i - 1]) < 0:
            q[:, i] *= -1
    return q


def build_waypoints(
    s_max: float,
    n_waypoints: int,
    position_by_arc,
    tangent_by_arc,
    euler_to_quat_fn=None,
    reference_speed: float = 0.0,
    gravity: float = 9.81,
    max_tilt_deg: float = 60.0,
):
    """Sample the path uniformly in arc-length and build waypoint arrays.

    Attitude reference via flatness:
      b3 = (a_lat + g*e3) / ||...||   (thrust direction)
      b1 = (I - b3*b3^T)*tang / ||...|| (tangent projected)
      b2 = b3 x b1

    Returns
    -------
    s_wp    : ndarray (N,)    — arc-length knots
    pos_wp  : ndarray (3, N)  — positions
    tang_wp : ndarray (3, N)  — unit tangents
    quat_wp : ndarray (4, N)  — hemisphere-consistent quaternions
    """
    s_wp    = np.linspace(0.0, s_max, n_waypoints)
    pos_wp  = np.zeros((3, n_waypoints))
    tang_wp = np.zeros((3, n_waypoints))
    quat_wp = np.zeros((4, n_waypoints))

    for i, sv in enumerate(s_wp):
        pos_wp[:, i]  = position_by_arc(sv)
        tang_wp[:, i] = tangent_by_arc(sv)

    ds = s_wp[1] - s_wp[0] if n_waypoints > 1 else 1.0
    curvature_wp = np.zeros_like(tang_wp)
    if n_waypoints > 1:
        curvature_wp[:, 0] = (tang_wp[:, 1] - tang_wp[:, 0]) / ds
        curvature_wp[:, -1] = (tang_wp[:, -1] - tang_wp[:, -2]) / ds
    for i in range(1, n_waypoints - 1):
        curvature_wp[:, i] = (tang_wp[:, i + 1] - tang_wp[:, i - 1]) / (2.0 * ds)

    max_tilt_rad = math.radians(max_tilt_deg)
    b1_prev = None

    for i in range(n_waypoints):
        tang_i = tang_wp[:, i]
        tang_i = tang_i / (np.linalg.norm(tang_i) + 1e-12)
        a_lat = (reference_speed ** 2) * curvature_wp[:, i]

        # b3: thrust direction
        a_lat_z_clamped = max(a_lat[2], -0.9 * gravity)
        thrust_dir = np.array([a_lat[0], a_lat[1], gravity + a_lat_z_clamped], dtype=float)
        if thrust_dir[2] < 0.1 * gravity:
            thrust_dir[2] = 0.1 * gravity
        horiz_norm = np.linalg.norm(thrust_dir[:2])
        max_horiz = thrust_dir[2] * math.tan(max_tilt_rad)
        if horiz_norm > max_horiz:
            thrust_dir[:2] *= max_horiz / horiz_norm
        b3 = thrust_dir / (np.linalg.norm(thrust_dir) + 1e-12)

        # b1: tangent projected onto plane perpendicular to b3
        b1 = tang_i - np.dot(tang_i, b3) * b3
        if np.linalg.norm(b1) < 1e-8:
            yaw_i = math.atan2(tang_i[1], tang_i[0])
            heading = np.array([math.cos(yaw_i), math.sin(yaw_i), 0.0])
            b1 = heading - np.dot(heading, b3) * b3
        if np.linalg.norm(b1) < 1e-8:
            b1 = np.array([1.0, 0.0, 0.0]) - b3[0] * b3
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)

        if b1_prev is not None and np.dot(b1, b1_prev) < 0:
            b1 = -b1

        b2 = np.cross(b3, b1)
        b2 = b2 / (np.linalg.norm(b2) + 1e-12)
        b1 = np.cross(b2, b3)
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)

        b1_prev = b1.copy()

        R = np.column_stack((b1, b2, b3))
        quat_wp[:, i] = rotation_matrix_to_quaternion(R)

    quat_wp = quaternion_hemisphere_correction(quat_wp)

    # Thrust reference: T_d(s) = m * ||a_lat + g*e3||
    thrust_wp = np.zeros(n_waypoints)
    for i in range(n_waypoints):
        a_lat = (reference_speed ** 2) * curvature_wp[:, i]
        a_lat_z_clamped = max(a_lat[2], -0.9 * gravity)
        thrust_dir = np.array([a_lat[0], a_lat[1], gravity + a_lat_z_clamped])
        if thrust_dir[2] < 0.1 * gravity:
            thrust_dir[2] = 0.1 * gravity
        horiz_norm = np.linalg.norm(thrust_dir[:2])
        max_horiz = thrust_dir[2] * math.tan(max_tilt_rad)
        if horiz_norm > max_horiz:
            thrust_dir[:2] *= max_horiz / horiz_norm
        # T_d = m * ||thrust_dir||  (mass is passed as gravity/g... we use gravity here)
        # Actually thrust_dir = a_lat + g*e3 so T_d = mass * ||thrust_dir||
        # But we don't have mass here. Store ||thrust_dir|| and multiply by mass in OCP.
        thrust_wp[i] = np.linalg.norm(thrust_dir)

    # Angular velocity reference: ω_hat(s) = vex(R_d^T dR_d/ds)
    # ω_d(s, v_θ) = ω_hat(s) * v_θ  (multiply by progress speed at runtime)
    omega_hat_wp = np.zeros((3, n_waypoints))
    for i in range(n_waypoints):
        q_i = quat_wp[:, i]
        # Central difference for dq/ds
        if i == 0:
            q_next = quat_wp[:, 1]
            if np.dot(q_i, q_next) < 0:
                q_next = -q_next
            dq_ds = (q_next - q_i) / ds
        elif i == n_waypoints - 1:
            q_prev = quat_wp[:, i - 1]
            if np.dot(q_i, q_prev) < 0:
                q_prev = -q_prev
            dq_ds = (q_i - q_prev) / ds
        else:
            q_prev = quat_wp[:, i - 1]
            q_next = quat_wp[:, i + 1]
            if np.dot(q_i, q_prev) < 0:
                q_prev = -q_prev
            if np.dot(q_i, q_next) < 0:
                q_next = -q_next
            dq_ds = (q_next - q_prev) / (2.0 * ds)

        # ω_hat = 2 * Im(q_conj * dq/ds)  (body-frame angular velocity per unit s)
        qw, qx, qy, qz = q_i
        dqw, dqx, dqy, dqz = dq_ds
        # q_conj * dq/ds (Hamilton product)
        omega_hat_wp[0, i] = 2.0 * (-qx*dqw + qw*dqx - qz*dqy + qy*dqz)
        omega_hat_wp[1, i] = 2.0 * (-qy*dqw + qz*dqx + qw*dqy - qx*dqz)
        omega_hat_wp[2, i] = 2.0 * (-qz*dqw - qy*dqx + qx*dqy + qw*dqz)

    return s_wp, pos_wp, tang_wp, quat_wp, thrust_wp, omega_hat_wp
