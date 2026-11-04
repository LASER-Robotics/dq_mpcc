#pragma once

#include <Eigen/Dense>

namespace quadrotor_mpc {

// NMPC: 13 states [p(3), v(3), q(4), ω(3)], 4 controls [T, ωx, ωy, ωz]
using State13  = Eigen::Matrix<double, 13, 1>;
using Control4 = Eigen::Matrix<double, 4, 1>;

// MPCC: 15 states [p(3), v(3), q(4), ω(3), θ, vθ], 5 controls [T, ωx, ωy, ωz, aθ]
using State15  = Eigen::Matrix<double, 15, 1>;
using Control5 = Eigen::Matrix<double, 5, 1>;

// DQ-MPCC: 16 states [dq(8), ω(3), vb(3), θ, vθ]
using State16  = Eigen::Matrix<double, 16, 1>;

// Quaternion [qw, qx, qy, qz]
using Quat4 = Eigen::Vector4d;

// Rotation matrix
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

}  // namespace quadrotor_mpc
