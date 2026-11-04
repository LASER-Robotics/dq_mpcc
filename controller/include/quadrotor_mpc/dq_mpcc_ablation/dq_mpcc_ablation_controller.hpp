#pragma once
/**
 * DQ-MPCC Controller — C++ wrapper over acados-generated DQ-MPCC solver.
 *
 * State  x ∈ ℝ¹⁷ = [dq(8), ω(3), v_b(3), θ, v_θ, f]
 * Control u ∈ ℝ⁵  = [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
 *
 * Runtime parameters p ∈ ℝ¹⁶:
 *   p[ 0: 3] = Q_phi     rotation-log error weights
 *   p[ 3: 6] = Q_ec      contouring error weights
 *   p[ 6: 9] = Q_el      lag error weights
 *   p[ 9:13] = U_mat     control effort [f, ωx, ωy, ωz]
 *   p[13]    = Q_s       progress weight  (-Q_s * v_θ)
 *   p[14]    = vtheta_max
 *   p[15]    = W_df     thrust rate penalty: W_df * Δf²
 */

#include "quadrotor_mpc/common/types.hpp"
#include <Eigen/Dense>

// Forward declaration of acados capsule type
struct quadrotor_dq_mpcc_ablation_solver_capsule;

namespace quadrotor_mpc {

using State17  = Eigen::Matrix<double, 17, 1>;
using Control5 = Eigen::Matrix<double, 5, 1>;

/// DQ-MPCC weight set (runtime parameters)
struct DqMpccWeights {
    Vec3 Q_phi  = Vec3(5.0, 5.0, 5.0);             // orientation (rotation log)
    Vec3 Q_ec   = Vec3(100.0, 100.0, 100.0);       // contouring
    Vec3 Q_el   = Vec3(10.0, 10.0, 10.0);          // lag
    Eigen::Vector4d U_mat = Eigen::Vector4d(0.1, 0.3, 0.3, 0.3); // control
    double Q_s       = 12.0;                         // progress
    double vtheta_max = 14.0;                        // max progress velocity
    double W_df      = 0.001;                         // thrust rate penalty (Δf²)
};

class DqMpccAblationController {
public:
    static const int N;     // horizon length (from generated header)
    static const int NX = 17;
    static const int NU = 5;
    static const int NP = 16;

    DqMpccAblationController();
    ~DqMpccAblationController();

    /// Initialise the acados solver capsule
    bool init();

    /// Set initial state constraint (pin x0)
    void set_x0(const State17& x0);

    /// Set runtime parameters (weights) for a given stage
    void set_params(int stage, const DqMpccWeights& w);

    /// Set the same weights for ALL stages (0..N)
    void set_params_all(const DqMpccWeights& w);

    /// Solve one SQP-RTI iteration
    int solve();

    /// Get optimal control at stage 0
    Control5 get_u0() const;

    /// Get predicted state at stage k
    State17 get_x(int k) const;

    /// Set state initial guess for stage k
    void set_x_init(int k, const State17& x);

    /// Set control initial guess for stage k
    void set_u_init(int k, const Control5& u);

    /// Get SQP-RTI wall-clock solve time [s]
    double get_solve_time() const;

    /// Release solver resources
    void free();

private:
    quadrotor_dq_mpcc_ablation_solver_capsule* capsule_ = nullptr;

    /// Pack weights into parameter array p[15]
    static void weights_to_param(double* p, const DqMpccWeights& w);
};

}  // namespace quadrotor_mpc
