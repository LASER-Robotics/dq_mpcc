"""
quad_config.py — Physical and OCP constants for C++ solver generation.

Self-contained: no imports from the parent Python project.
These values must match params.hpp in the C++ codebase.
Trajectory parameters loaded from config/trajectory.json (single source of truth).
"""

import json
import numpy as np
from pathlib import Path

# Load trajectory parameters from JSON
_traj_json = Path(__file__).resolve().parent.parent / "config" / "trajectory.json"
with open(_traj_json) as _f:
    _traj = json.load(_f)

TRAJ_AX      = _traj["Ax"]
TRAJ_AY      = _traj["Ay"]
TRAJ_AZ      = _traj["Az"]
TRAJ_CX      = _traj["Cx"]
TRAJ_CZ      = _traj["Cz"]
TRAJ_W       = _traj["w"]
TRAJ_T_FINAL = _traj["t_final"]

# Gravity
G = 9.81  # [m/s^2]

# Quadrotor physical parameters
MASS     = 1.08   # [kg]
R_DRONE  = 0.20   # [m] half-extent (prop-tip to prop-tip / 2), 6-inch racing frame
TAU_RC   = 0.03   # [s] rate-controller time constant

# Thrust limits [N]
T_MAX = 5 * G   # ~49.05 N  (MuJoCo-consistent; real-drone 80 N is for HW only)
T_MIN = 0.0

# Angular rate command limits [rad/s]
W_MAX = 20.0

# Progress dynamics limits
VTHETA_MIN =  0.0        # [m/s]
VTHETA_MAX = 14.0         # [m/s]
ATHETA_MIN = -5 * G       # [m/s^2]
ATHETA_MAX =  5 * G       # [m/s^2]

# Thrust rate limits [N/s]
DF_MAX = 500.0            # [N/s] max thrust rate of change

# Lag constraint
D_MAX_LAG = 0.5  # [m] max arc-length gap between θ and drone position

# Trajectory
TRAJ_VALUE = TRAJ_W                 # [rad/s] Lissajous base angular frequency
TRAJECTORY_T_FINAL = int(TRAJ_T_FINAL)  # [s] parameter interval for path geometry
FREC = 100                # [Hz] control frequency
N_WAYPOINTS = 400         # B-spline interpolation knots
S_MAX_MANUAL = 100        # [m] arc-length limit
THETA_MARGIN = 30.0       # [m] extra headroom for θ beyond s_max (≈ vtheta_max * T_horizon)

# Attitude reference construction
ATTITUDE_REF_SPEED = 15.0        # [m/s] nominal speed for attitude reference
ATTITUDE_REF_MAX_TILT_DEG = 60.0 # [deg] max tilt in reference quaternion


def trajectory():
    """Return 6 lambdas: (xd, yd, zd, xdp, ydp, zdp).

    Lissajous 3D, frequency ratio 1:2:3.
    Parameters loaded from config/trajectory.json.
    """
    w  = TRAJ_W
    ax, ay, az = TRAJ_AX, TRAJ_AY, TRAJ_AZ
    cx, cz     = TRAJ_CX, TRAJ_CZ
    xd  = lambda t: ax * np.sin(1.0 * w * t) + cx
    yd  = lambda t: ay * np.sin(2.0 * w * t)
    zd  = lambda t: az * np.sin(3.0 * w * t) + cz
    xdp = lambda t: ax * 1.0 * w * np.cos(1.0 * w * t)
    ydp = lambda t: ay * 2.0 * w * np.cos(2.0 * w * t)
    zdp = lambda t: az * 3.0 * w * np.cos(3.0 * w * t)
    return xd, yd, zd, xdp, ydp, zdp
