#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include <cmath>

namespace quadrotor_mpc {

Mat3 quat_to_rot(const Quat4& q) {
    const double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    Mat3 R;
    R(0,0) = 1.0 - 2.0*(qy*qy + qz*qz);
    R(0,1) = 2.0*(qx*qy - qz*qw);
    R(0,2) = 2.0*(qx*qz + qy*qw);
    R(1,0) = 2.0*(qx*qy + qz*qw);
    R(1,1) = 1.0 - 2.0*(qx*qx + qz*qz);
    R(1,2) = 2.0*(qy*qz - qx*qw);
    R(2,0) = 2.0*(qx*qz - qy*qw);
    R(2,1) = 2.0*(qy*qz + qx*qw);
    R(2,2) = 1.0 - 2.0*(qx*qx + qy*qy);
    return R;
}

Quat4 quat_multiply(const Quat4& q1, const Quat4& q2) {
    const double w0 = q1(0), x0 = q1(1), y0 = q1(2), z0 = q1(3);
    const double w1 = q2(0), x1 = q2(1), y1 = q2(2), z1 = q2(3);
    Quat4 r;
    r(0) = w0*w1 - x0*x1 - y0*y1 - z0*z1;
    r(1) = w0*x1 + x0*w1 + y0*z1 - z0*y1;
    r(2) = w0*y1 - x0*z1 + y0*w1 + z0*x1;
    r(3) = w0*z1 + x0*y1 - y0*x1 + z0*w1;
    return r;
}

Quat4 quat_conjugate(const Quat4& q) {
    return Quat4(q(0), -q(1), -q(2), -q(3));
}

Quat4 quat_error(const Quat4& q_real, const Quat4& q_desired) {
    Quat4 q_inv = quat_conjugate(q_real);
    q_inv /= q_real.norm();
    return quat_multiply(q_inv, q_desired);
}

Vec3 quat_log(const Quat4& q_in) {
    // Enforce positive hemisphere
    Quat4 q = (q_in(0) < 0.0) ? -q_in : q_in;

    const double qw = q(0);
    const Vec3 qv(q(1), q(2), q(3));
    const double norm_qv = qv.norm();
    const double theta = std::atan2(norm_qv, qw);
    const double safe_norm = norm_qv + 1e-9;
    return 2.0 * qv * theta / safe_norm;
}

Quat4 quat_kinematics(const Quat4& q, const Vec3& omega) {
    Quat4 omega_quat(0.0, omega(0), omega(1), omega(2));
    return 0.5 * quat_multiply(q, omega_quat);
}

Quat4 quat_normalize(const Quat4& q) {
    double n = q.norm();
    if (n < 1e-12) return Quat4(1.0, 0.0, 0.0, 0.0);
    return q / n;
}

Quat4 quat_positive_hemisphere(const Quat4& q) {
    return (q(0) < 0.0) ? -q : q;
}

Quat4 rot_to_quat(const Mat3& R) {
    double trace = R.trace();
    Quat4 q;
    if (trace > 0.0) {
        double s = std::sqrt(trace + 1.0) * 2.0;
        q(0) = 0.25 * s;
        q(1) = (R(2,1) - R(1,2)) / s;
        q(2) = (R(0,2) - R(2,0)) / s;
        q(3) = (R(1,0) - R(0,1)) / s;
    } else if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
        double s = std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2)) * 2.0;
        q(0) = (R(2,1) - R(1,2)) / s;
        q(1) = 0.25 * s;
        q(2) = (R(0,1) + R(1,0)) / s;
        q(3) = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        double s = std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2)) * 2.0;
        q(0) = (R(0,2) - R(2,0)) / s;
        q(1) = (R(0,1) + R(1,0)) / s;
        q(2) = 0.25 * s;
        q(3) = (R(1,2) + R(2,1)) / s;
    } else {
        double s = std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1)) * 2.0;
        q(0) = (R(1,0) - R(0,1)) / s;
        q(1) = (R(0,2) + R(2,0)) / s;
        q(2) = (R(1,2) + R(2,1)) / s;
        q(3) = 0.25 * s;
    }
    return quat_normalize(q);
}

Vec3 quat_to_euler(const Quat4& q) {
    const double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    // Roll (x)
    double sinr = 2.0 * (qw*qx + qy*qz);
    double cosr = 1.0 - 2.0 * (qx*qx + qy*qy);
    double roll = std::atan2(sinr, cosr);
    // Pitch (y)
    double sinp = 2.0 * (qw*qy - qz*qx);
    sinp = std::clamp(sinp, -1.0, 1.0);
    double pitch = std::asin(sinp);
    // Yaw (z)
    double siny = 2.0 * (qw*qz + qx*qy);
    double cosy = 1.0 - 2.0 * (qy*qy + qz*qz);
    double yaw = std::atan2(siny, cosy);
    return Vec3(roll, pitch, yaw);
}

}  // namespace quadrotor_mpc
