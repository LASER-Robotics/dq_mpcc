#include "quadrotor_mpc/trajectory/lissajous.hpp"
#include <cmath>

namespace quadrotor_mpc {

Vec3 LissajousTrajectory::position(double t) const {
    return Vec3(
        Ax * std::sin(1.0 * w * t) + Cx,
        Ay * std::sin(2.0 * w * t),
        Az * std::sin(3.0 * w * t) + Cz
    );
}

Vec3 LissajousTrajectory::velocity(double t) const {
    return Vec3(
        Ax * 1.0 * w * std::cos(1.0 * w * t),
        Ay * 2.0 * w * std::cos(2.0 * w * t),
        Az * 3.0 * w * std::cos(3.0 * w * t)
    );
}

void LissajousTrajectory::sample(int N_samples,
                                  std::vector<Vec3>& positions,
                                  std::vector<Vec3>& velocities,
                                  std::vector<double>& t_values) const {
    positions.resize(N_samples);
    velocities.resize(N_samples);
    t_values.resize(N_samples);

    for (int i = 0; i < N_samples; ++i) {
        double t = t_final * static_cast<double>(i) / static_cast<double>(N_samples - 1);
        t_values[i]   = t;
        positions[i]   = position(t);
        velocities[i]  = velocity(t);
    }
}

}  // namespace quadrotor_mpc
