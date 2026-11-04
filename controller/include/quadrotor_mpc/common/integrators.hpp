#pragma once

#include <Eigen/Dense>
#include <functional>

namespace quadrotor_mpc {

/// Generic RK4 integrator for any state/control size.
/// f(x, u) → ẋ
template <int NX, int NU>
Eigen::Matrix<double, NX, 1> rk4_step(
    const std::function<Eigen::Matrix<double, NX, 1>(
        const Eigen::Matrix<double, NX, 1>&,
        const Eigen::Matrix<double, NU, 1>&)>& f,
    const Eigen::Matrix<double, NX, 1>& x,
    const Eigen::Matrix<double, NU, 1>& u,
    double dt)
{
    auto k1 = f(x, u);
    auto k2 = f(x + 0.5 * dt * k1, u);
    auto k3 = f(x + 0.5 * dt * k2, u);
    auto k4 = f(x + dt * k3, u);
    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

}  // namespace quadrotor_mpc
