#include "quadrotor_mpc/trajectory/attitude_reference.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace quadrotor_mpc {

PathWaypoints build_waypoints(
    const ArcLengthPath& path,
    double s_max,
    int n_waypoints,
    double reference_speed,
    double gravity,
    double max_tilt_deg)
{
    PathWaypoints wp;
    wp.s_max = s_max;
    wp.s.resize(n_waypoints);
    wp.pos.resize(n_waypoints);
    wp.tang.resize(n_waypoints);
    wp.quat.resize(n_waypoints);

    // Sample uniformly in arc-length
    for (int i = 0; i < n_waypoints; ++i) {
        wp.s[i] = s_max * static_cast<double>(i) / static_cast<double>(n_waypoints - 1);
        wp.pos[i]  = path.position(wp.s[i]);
        wp.tang[i] = path.tangent(wp.s[i]);
    }

    // Compute curvature: κ(s) = dt/ds via central differences
    double ds = wp.s[1] - wp.s[0];
    std::vector<Vec3> curvature(n_waypoints, Vec3::Zero());
    if (n_waypoints > 1) {
        curvature[0] = (wp.tang[1] - wp.tang[0]) / ds;
        curvature[n_waypoints-1] = (wp.tang[n_waypoints-1] - wp.tang[n_waypoints-2]) / ds;
    }
    for (int i = 1; i < n_waypoints - 1; ++i) {
        curvature[i] = (wp.tang[i+1] - wp.tang[i-1]) / (2.0 * ds);
    }

    const double max_tilt_rad = max_tilt_deg * M_PI / 180.0;
    Vec3 b1_prev = Vec3::Zero();
    bool has_prev = false;

    for (int i = 0; i < n_waypoints; ++i) {
        Vec3 tang_i = wp.tang[i];
        double tn = tang_i.norm();
        if (tn > 1e-12) tang_i /= tn;

        Vec3 a_lat = (reference_speed * reference_speed) * curvature[i];

        // b3: thrust direction — eq. (12)
        double a_lat_z_clamped = std::max(a_lat(2), -0.9 * gravity);
        Vec3 thrust_dir(a_lat(0), a_lat(1), gravity + a_lat_z_clamped);
        if (thrust_dir(2) < 0.1 * gravity) {
            thrust_dir(2) = 0.1 * gravity;
        }
        double horiz_norm = thrust_dir.head<2>().norm();
        double max_horiz = thrust_dir(2) * std::tan(max_tilt_rad);
        if (horiz_norm > max_horiz) {
            thrust_dir.head<2>() *= max_horiz / horiz_norm;
        }
        Vec3 b3 = thrust_dir.normalized();

        // b1: tangent projected onto plane perp to b3 — eq. (13)
        Vec3 b1 = tang_i - tang_i.dot(b3) * b3;
        if (b1.norm() < 1e-8) {
            double yaw_i = std::atan2(tang_i(1), tang_i(0));
            Vec3 heading(std::cos(yaw_i), std::sin(yaw_i), 0.0);
            b1 = heading - heading.dot(b3) * b3;
        }
        if (b1.norm() < 1e-8) {
            b1 = Vec3(1.0, 0.0, 0.0) - b3(0) * b3;
        }
        b1.normalize();

        // Anti-flip using previous b1
        if (has_prev && b1.dot(b1_prev) < 0.0) {
            b1 = -b1;
        }

        // b2 = b3 × b1 — eq. (14), re-orthogonalise
        Vec3 b2 = b3.cross(b1).normalized();
        b1 = b2.cross(b3).normalized();

        b1_prev = b1;
        has_prev = true;

        Mat3 R;
        R.col(0) = b1;
        R.col(1) = b2;
        R.col(2) = b3;
        wp.quat[i] = rot_to_quat(R);
    }

    // Hemisphere correction
    for (int i = 1; i < n_waypoints; ++i) {
        if (wp.quat[i].dot(wp.quat[i-1]) < 0.0) {
            wp.quat[i] = -wp.quat[i];
        }
    }

    return wp;
}

static int find_segment(double s, const std::vector<double>& s_wp) {
    auto it = std::upper_bound(s_wp.begin(), s_wp.end(), s);
    int idx = static_cast<int>(it - s_wp.begin()) - 1;
    return std::clamp(idx, 0, static_cast<int>(s_wp.size()) - 2);
}

Quat4 quat_interp(double s, const PathWaypoints& wp) {
    s = std::clamp(s, wp.s.front(), wp.s.back());
    int i = find_segment(s, wp.s);
    double ds = wp.s[i+1] - wp.s[i];
    double alpha = (ds > 1e-12) ? (s - wp.s[i]) / ds : 0.0;

    Quat4 q0 = wp.quat[i];
    Quat4 q1 = wp.quat[i+1];
    // Ensure shortest path
    if (q0.dot(q1) < 0.0) q1 = -q1;
    Quat4 q = (1.0 - alpha) * q0 + alpha * q1;
    return quat_normalize(q);
}

Vec3 pos_interp(double s, const PathWaypoints& wp) {
    s = std::clamp(s, wp.s.front(), wp.s.back());
    int i = find_segment(s, wp.s);
    double ds = wp.s[i+1] - wp.s[i];
    double alpha = (ds > 1e-12) ? (s - wp.s[i]) / ds : 0.0;
    return (1.0 - alpha) * wp.pos[i] + alpha * wp.pos[i+1];
}

Vec3 tang_interp(double s, const PathWaypoints& wp) {
    s = std::clamp(s, wp.s.front(), wp.s.back());
    int i = find_segment(s, wp.s);
    double ds = wp.s[i+1] - wp.s[i];
    double alpha = (ds > 1e-12) ? (s - wp.s[i]) / ds : 0.0;
    Vec3 t = (1.0 - alpha) * wp.tang[i] + alpha * wp.tang[i+1];
    double n = t.norm();
    return (n > 1e-12) ? t / n : Vec3(1.0, 0.0, 0.0);
}

}  // namespace quadrotor_mpc
