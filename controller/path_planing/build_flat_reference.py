"""
build_flat_reference.py — Phase 1B: smoothing + differential flatness + arc-length.

Input:   xref_multilap.npy, uref_multilap.npy, tref_multilap.npy (from Phase 1A).
Output:  ref_flat_waypoints.npz   — waypoint arrays in the format consumed by
                                    ocp_generation/generate_{mpcc,dq_mpcc}_ocp.py
                                    {s_wp, pos_wp, vel_wp, a_wp, tang_wp,
                                     quat_wp, thrust_wp, omega_hat_wp, s_max,
                                     kernel_ms}

Pipeline:
  1. Load multilap (p, v, a) @100 Hz
  2. Align controls to state grid (U extended last sample)
  3. Gaussian-smooth acceleration  (FWHM kernel_ms, C^2 profile)
  4. Differential flatness:
        b_3 = (a_smooth + g·e_z) / ‖...‖
        b_1 from fixed yaw ψ=0 (singularity-free for tilt < 80°)
        b_2 = b_3 × b_1 ; re-orthogonalise b_1
  5. Arc-length s(t) = ∫‖v‖dt (monotonised to strict increasing)
  6. Resample onto uniform s grid of N_WP waypoints (cubic spline)
  7. Quaternion re-extraction on s grid, hemisphere-consistent
  8. ω_0(s) = (R^T dR/ds)^∨   (central differences on waypoint R)
  9. Save waypoints + validation plot

Interface compatibility: matches `path_utils.build_waypoints()` return shapes
so the OCP generators can load either parametric (Lissajous+v̄_s) or
data-driven (this script) references via the same downstream code.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAVITY    = 9.81
G_VEC      = np.array([0.0, 0.0, GRAVITY])
KERNEL_MS      = 60.0       # Gaussian FWHM for a(t) smoothing
KERNEL_MS_YAW  = 100.0      # extra smoothing for yaw (soft anchor)
KERNEL_MS_QUAT = 200.0      # post-smoothing of quaternion — damps rapid PMM tilt swings (G5 pendular)
N_WP           = 502        # uniform arc-length grid; sized for ds≈0.30m
                            # (preserves tune-time ds so weights remain valid)
YAW_MODE       = 'fixed'    # 'velocity' (FPV-style) or 'fixed' (ψ=0)
V_THRESH       = 0.5        # [m/s] below which yaw is held (hover moments)


# 1. Load multilap reference
X = np.load(os.path.join(SCRIPT_DIR, "xref_multilap.npy"))     # (6, N)
U = np.load(os.path.join(SCRIPT_DIR, "uref_multilap.npy"))     # (3, N-1)
t = np.load(os.path.join(SCRIPT_DIR, "tref_multilap.npy"))     # (N,)
meta = np.load(os.path.join(SCRIPT_DIR, "multilap_metadata.npz"), allow_pickle=True)

N  = X.shape[1]
dt = float(t[1] - t[0])
p  = X[0:3]
v  = X[3:6]
v_mag = np.linalg.norm(v, axis=0)

# Extend controls to match state grid (ZOH for the last sample)
a_raw = np.column_stack([U, U[:, -1:]])                        # (3, N)

print("=" * 70)
print("PHASE 1B — Smoothing + Differential Flatness + Arc-length")
print("=" * 70)
print(f"Input: multilap  N={N}  T={t[-1]:.2f}s  dt={dt*1e3:.1f}ms")
print(f"|v|: min={v_mag.min():.3f}  max={v_mag.max():.3f}  mean={v_mag.mean():.3f} m/s")
print(f"|a_raw|: max={np.linalg.norm(a_raw, axis=0).max():.2f} m/s² "
      f"= {np.linalg.norm(a_raw, axis=0).max()/GRAVITY:.2f} g")


# 2. Gaussian smoothing of a(t) — C^2 profile for well-defined ω
sigma_samples = (KERNEL_MS * 1e-3) / dt / 2.355          # FWHM → sigma
a_smooth = gaussian_filter1d(a_raw, sigma=sigma_samples, axis=1, mode='nearest')

jerk_raw    = np.gradient(a_raw,    t, axis=1)
jerk_smooth = np.gradient(a_smooth, t, axis=1)
print(f"\nSmoothing kernel: FWHM={KERNEL_MS:.1f} ms  (σ={sigma_samples:.2f} samples)")
print(f"|jerk|_max:  raw={np.linalg.norm(jerk_raw, axis=0).max():7.1f} m/s³  "
      f"smooth={np.linalg.norm(jerk_smooth, axis=0).max():7.1f} m/s³")


# 3. Differential flatness on time grid — b_3, b_1, b_2, q
# PMM already enforces a_z ≥ -0.9 g at the OCP level so thrust stays in the
# upper hemisphere (tilt ≤ 84°).  We only sanity-check here.
n_viol = int((a_smooth[2] < -0.95 * GRAVITY).sum())
if n_viol > 0:
    print(f"Warning: {n_viol} samples with a_z < -0.95g — upstream OCP not "
          f"enforcing thrust-positive hemisphere.")

f_vec = a_smooth + G_VEC[:, None]                         # (3, N) specific-thrust vec
f_mag = np.linalg.norm(f_vec, axis=0)
b3    = f_vec / f_mag

# Yaw reference — anchor x_c for building b_1.
# Switched to 3D velocity tangent (instead of horizontal [cos ψ, sin ψ, 0]):
# this removes the flat-map singularity that occurs at high tilt when b_3's
# horizontal projection aligns with the velocity direction. In 3D, v is
# typically near-perpendicular to b_3 in banking turns, so b_3 × v is robust.
if YAW_MODE == 'velocity':
    sigma_y = (KERNEL_MS_YAW * 1e-3) / dt / 2.355
    v_smooth_anchor = np.stack([gaussian_filter1d(v[i], sigma=sigma_y, mode='nearest')
                                for i in range(3)])
    v_mag_sm = np.linalg.norm(v_smooth_anchor, axis=0)
    x_c_t = np.zeros_like(v_smooth_anchor)
    prev = np.array([1.0, 0.0, 0.0])
    for k in range(N):
        if v_mag_sm[k] > V_THRESH:
            prev = v_smooth_anchor[:, k] / v_mag_sm[k]
        x_c_t[:, k] = prev
    # Diagnostic psi for logging
    psi = np.arctan2(x_c_t[1], x_c_t[0])
    print(f"Yaw anchor: 3D smoothed velocity tangent (yaw kernel={KERNEL_MS_YAW:.0f} ms)")
else:
    psi = np.zeros(N)
    x_c_t = np.tile(np.array([1.0, 0.0, 0.0]).reshape(-1, 1), (1, N))
    print("Yaw anchor: fixed [1,0,0]")

b1 = np.zeros_like(b3)
for k in range(N):
    xc = x_c_t[:, k]
    b1k = xc - np.dot(xc, b3[:, k]) * b3[:, k]
    n = np.linalg.norm(b1k)
    if n < 1e-3:
        # Only triggers if v happens to align with b_3 (rare in racing).
        # Fallback: pick any direction orthogonal to b_3.
        ref = np.array([0.0, 0.0, 1.0]) if abs(b3[2, k]) < 0.9 else np.array([1.0, 0.0, 0.0])
        b1k = ref - np.dot(ref, b3[:, k]) * b3[:, k]
        n = np.linalg.norm(b1k)
    b1[:, k] = b1k / max(n, 1e-12)

b2 = np.cross(b3.T, b1.T).T
b2 /= (np.linalg.norm(b2, axis=0) + 1e-12)
# Re-project b1 to guarantee orthonormality
b1 = np.cross(b2.T, b3.T).T
b1 /= (np.linalg.norm(b1, axis=0) + 1e-12)

# R(t) shape (N, 3, 3): columns = [b1 b2 b3]
R = np.stack([b1.T, b2.T, b3.T], axis=2)

# Quaternion [w,x,y,z] with hemisphere consistency
q_xyzw = Rotation.from_matrix(R).as_quat()
q_wxyz_t = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1).T   # (4, N)
for k in range(1, N):
    if np.dot(q_wxyz_t[:, k], q_wxyz_t[:, k-1]) < 0:
        q_wxyz_t[:, k] *= -1

# Post-smoothing of quaternion — attenuates flat-map singularity spikes at
# tilt ≈ 80°+ where x_c and b_3's horizontal projection nearly align.
sigma_q = (KERNEL_MS_QUAT * 1e-3) / dt / 2.355
q_wxyz_t = gaussian_filter1d(q_wxyz_t, sigma=sigma_q, axis=1, mode='nearest')
q_wxyz_t /= (np.linalg.norm(q_wxyz_t, axis=0, keepdims=True) + 1e-12)
# Re-enforce hemisphere after smoothing
for k in range(1, N):
    if np.dot(q_wxyz_t[:, k], q_wxyz_t[:, k-1]) < 0:
        q_wxyz_t[:, k] *= -1

# Re-derive b_3 from smoothed quaternion for diagnostics
R_smooth = Rotation.from_quat(np.column_stack([q_wxyz_t[1:4].T, q_wxyz_t[0:1].T])).as_matrix()
b3 = R_smooth[:, :, 2].T     # col 3 of each R
tilt = np.degrees(np.arccos(np.clip(b3[2, :], -1, 1)))
print(f"\nDiff-flat on time grid:")
print(f"  tilt_max = {tilt.max():.1f}°  (b_3 from vertical)")
print(f"  |f|/g:  min={f_mag.min()/GRAVITY:.2f}  max={f_mag.max()/GRAVITY:.2f}")


# 4. Arc-length parameterisation s(t) = ∫‖v‖dt (monotonised)
s_t = np.concatenate([[0.0], cumulative_trapezoid(v_mag, t)])
# Make strictly increasing (hover endpoints have v=0 → flat segments)
s_mono = s_t.copy()
for k in range(1, N):
    if s_mono[k] <= s_mono[k-1]:
        s_mono[k] = s_mono[k-1] + 1e-6
s_max = float(s_mono[-1])
print(f"\nArc-length: s_max = {s_max:.2f} m  (total path length)")


# 5. Resample onto uniform s grid
s_wp = np.linspace(0.0, s_max, N_WP)

def _interp_vec(vals):
    """vals shape (d, N) → (d, N_WP) via cubic spline on s_mono."""
    return np.stack([CubicSpline(s_mono, vals[i])(s_wp) for i in range(vals.shape[0])])

pos_wp = _interp_vec(p)
vel_wp = _interp_vec(v)
a_wp   = _interp_vec(a_smooth)

# Unit tangent from velocity (fallback to x_c when |v|<V_THRESH near hover endpoints)
v_wp_mag = np.linalg.norm(vel_wp, axis=0)
tang_wp = np.zeros_like(vel_wp)
for k in range(N_WP):
    if v_wp_mag[k] > V_THRESH:
        tang_wp[:, k] = vel_wp[:, k] / v_wp_mag[k]
    else:
        # Use nearest non-hover tangent (or first, for k=0)
        idx = np.argmax(v_wp_mag > V_THRESH) if v_wp_mag[0] <= V_THRESH else 0
        tang_wp[:, k] = vel_wp[:, idx] / max(v_wp_mag[idx], 1e-6)

# Quaternion: interpolate component-by-component (hemisphere already fixed on t-grid)
quat_wp = _interp_vec(q_wxyz_t)
quat_wp /= (np.linalg.norm(quat_wp, axis=0) + 1e-12)

# Specific thrust magnitude at waypoints (‖a_smooth + g·e_z‖)
thrust_wp = np.linalg.norm(a_wp + G_VEC[:, None], axis=0)

print(f"\nResampled onto uniform arc-length grid: N_WP={N_WP}  ds_wp={s_wp[1]-s_wp[0]:.3f} m")


# 6. ω_0(s) = per-unit-arc-length body rate = (R^T dR/ds)^∨
def _quat_to_R(q):
    """[w,x,y,z] → 3×3 rotation."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

omega_hat_wp = np.zeros((3, N_WP))
ds = s_wp[1] - s_wp[0]
for k in range(N_WP):
    if k == 0:
        Rk = _quat_to_R(quat_wp[:, k]); Rkp1 = _quat_to_R(quat_wp[:, k+1])
        dR = (Rkp1 - Rk) / ds
    elif k == N_WP - 1:
        Rk = _quat_to_R(quat_wp[:, k]); Rkm1 = _quat_to_R(quat_wp[:, k-1])
        dR = (Rk - Rkm1) / ds
    else:
        Rk = _quat_to_R(quat_wp[:, k])
        Rkp1 = _quat_to_R(quat_wp[:, k+1]); Rkm1 = _quat_to_R(quat_wp[:, k-1])
        dR = (Rkp1 - Rkm1) / (2 * ds)
    W = Rk.T @ dR                              # skew-symmetric (body frame)
    omega_hat_wp[:, k] = np.array([W[2, 1], W[0, 2], W[1, 0]])

# Expected ω at operating speed v_s(s) = v_mag interpolated: ω(s) = ω_0(s) · v_s(s)
omega_at_op_speed = omega_hat_wp * v_wp_mag[None, :]
print(f"ω_0: |ω_0|_max = {np.linalg.norm(omega_hat_wp, axis=0).max():.2f} rad/m")
print(f"ω(s)·v_s at operating speed: |ω|_max = {np.linalg.norm(omega_at_op_speed, axis=0).max():.2f} rad/s  "
      f"(W_max drone = 20 rad/s)")


# 7. Save
out = os.path.join(SCRIPT_DIR, "ref_flat_waypoints.npz")
np.savez(out,
    s_wp=s_wp, pos_wp=pos_wp, vel_wp=vel_wp, a_wp=a_wp,
    tang_wp=tang_wp, quat_wp=quat_wp,
    thrust_wp=thrust_wp, omega_hat_wp=omega_hat_wp,
    s_max=np.array([s_max]),
    kernel_ms=np.array([KERNEL_MS]),
    psi=psi,
    yaw_mode=YAW_MODE,
    # Time-grid references (for diagnostics / optional consumption)
    t=t, p_t=p, v_t=v, a_raw_t=a_raw, a_smooth_t=a_smooth,
    q_t=q_wxyz_t, f_mag_t=f_mag, s_t=s_t,
)
print(f"\n✓ Saved: {out}")
print(f"  Contents: s_wp, pos_wp(3×{N_WP}), vel_wp, a_wp, tang_wp, "
      f"quat_wp(4×{N_WP}), thrust_wp, omega_hat_wp")
print(f"  + time-grid diagnostics (t, p_t, v_t, a_raw_t, a_smooth_t, q_t, f_mag_t, s_t)")


# 8. Validation plot
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=False)

# (a) raw vs smooth acceleration magnitude (time grid)
axes[0].plot(t, np.linalg.norm(a_raw,    axis=0), color='gray', lw=0.6, alpha=0.7, label='raw')
axes[0].plot(t, np.linalg.norm(a_smooth, axis=0), 'k', lw=1.2, label='smooth (Gaussian)')
axes[0].axhline(5*GRAVITY, color='red', ls=':', alpha=0.5, label='5g')
axes[0].set_ylabel('|a| [m/s²]'); axes[0].set_xlabel('t [s]')
axes[0].set_title(f"Acceleration smoothing (kernel FWHM = {KERNEL_MS:.0f} ms)")
axes[0].legend(fontsize=9, ncol=3); axes[0].grid(alpha=0.35)

# (b) |f|/g — specific thrust
axes[1].plot(s_wp, thrust_wp/GRAVITY, 'k', lw=1.1)
axes[1].axhline(49.05/1.08/GRAVITY, color='red', ls=':', alpha=0.6, label='TWR limit')
axes[1].set_ylabel('|f|/g'); axes[1].set_xlabel('s [m]')
axes[1].set_title('Specific thrust magnitude (arc-length)')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.35)

# (c) quaternion components
colors = ['black', 'red', 'green', 'blue']
labels = ['q_w', 'q_x', 'q_y', 'q_z']
for i, (c, l) in enumerate(zip(colors, labels)):
    axes[2].plot(s_wp, quat_wp[i], color=c, lw=0.9, label=l)
axes[2].set_ylabel('quat'); axes[2].set_xlabel('s [m]')
axes[2].set_title('Reference quaternion q_r^d(s) (hemisphere-consistent)')
axes[2].legend(fontsize=9, ncol=4); axes[2].grid(alpha=0.35)
axes[2].set_ylim(-1.1, 1.1)

# (d) ω_0(s) components  +  ω(s)·v_s(s)
for i, (c, l) in enumerate(zip(['r','g','b'], ['ω_x','ω_y','ω_z'])):
    axes[3].plot(s_wp, omega_at_op_speed[i], color=c, lw=0.8, alpha=0.7,
                 label=f'{l} (at v_s)')
axes[3].plot(s_wp, np.linalg.norm(omega_at_op_speed, axis=0), 'k', lw=1.2, label='|ω|')
axes[3].axhline( 20, color='red', ls=':', alpha=0.6, label='W_max=20')
axes[3].axhline(-20, color='red', ls=':', alpha=0.6)
axes[3].set_ylabel('ω [rad/s]'); axes[3].set_xlabel('s [m]')
axes[3].set_title('Body-rate reference at operating speed: ω(s)=ω_0(s)·v_s(s)')
axes[3].legend(fontsize=9, ncol=5); axes[3].grid(alpha=0.35)

# (e) s(t) — arc-length progress profile
axes[4].plot(t, s_t, 'k', lw=1.2)
axes[4].set_xlabel('t [s]'); axes[4].set_ylabel('s [m]')
axes[4].set_title(f'Arc-length mapping s(t)  (s_max = {s_max:.2f} m)')
axes[4].grid(alpha=0.35)

fig.tight_layout()
_fig_dir = os.path.join(SCRIPT_DIR, "figures", "flat")
os.makedirs(_fig_dir, exist_ok=True)
out_png = os.path.join(_fig_dir, "ref_flat_validation.png")
fig.savefig(out_png, dpi=140, bbox_inches='tight')
print(f"✓ Saved: {out_png}")
