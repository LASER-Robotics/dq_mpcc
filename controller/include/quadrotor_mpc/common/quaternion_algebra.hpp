#pragma once

#include "quadrotor_mpc/common/types.hpp"

namespace quadrotor_mpc {

// All quaternions use Hamilton convention: q = [qw, qx, qy, qz] (scalar first)

/// Quaternion → rotation matrix R ∈ SO(3) (body→inertial)
Mat3 quat_to_rot(const Quat4& q);

/// Hamilton product q1 ⊗ q2
Quat4 quat_multiply(const Quat4& q1, const Quat4& q2);

/// Quaternion conjugate (= inverse for unit quaternions)
Quat4 quat_conjugate(const Quat4& q);

/// Quaternion error: q_err = q_real⁻¹ ⊗ q_desired
Quat4 quat_error(const Quat4& q_real, const Quat4& q_desired);

/// Quaternion logarithm: Log(q) = 2·atan2(||qv||, qw)·qv/||qv|| ∈ so(3)
Vec3 quat_log(const Quat4& q);

/// Quaternion kinematics: q̇ = ½ q ⊗ [0, ω]
Quat4 quat_kinematics(const Quat4& q, const Vec3& omega);

/// Normalise quaternion to unit length
Quat4 quat_normalize(const Quat4& q);

/// Ensure qw ≥ 0 (resolve double cover)
Quat4 quat_positive_hemisphere(const Quat4& q);

/// Rotation matrix → quaternion [qw, qx, qy, qz]
Quat4 rot_to_quat(const Mat3& R);

/// Quaternion to Euler angles [roll, pitch, yaw] (ZYX convention)
Vec3 quat_to_euler(const Quat4& q);

}  // namespace quadrotor_mpc
