#pragma once

namespace quadrotor_mpc {

struct QuadParams {
    // Physical
    double mass    = 1.08;      // [kg]
    double g       = 9.81;      // [m/s²]
    double tau_rc  = 0.03;      // [s] first-order rate lag

    // Inertia (unused in rate-control model, kept for torque model)
    double Jxx = 3.2086e-03;    // [kg·m²]
    double Jyy = 1.6731e-03;
    double Jzz = 1.6730e-03;

    // Control limits
    double T_max       = 5.0 * 9.81;   // ≈49.05 N (MuJoCo-consistent)
    double T_min       = 0.0;
    double W_max       = 20.0;          // [rad/s] max body rate command
    double vtheta_min  = 0.0;
    double vtheta_max  = 14.0;          // [m/s]
    double atheta_min  = -5.0 * 9.81;   // [m/s²]
    double atheta_max  =  5.0 * 9.81;
    double df_max      = 500.0;         // [N/s] max thrust rate of change

    // Lag constraint
    double D_max_lag = 0.5;             // [m]

    // θ headroom: OCP bound is s_max + theta_margin so solver doesn't brake
    // near the end. Termination is based on s_nearest (physical), not θ.
    double theta_margin = 30.0;         // [m] ≈ vtheta_max * T_horizon

    // Attitude reference
    double att_ref_speed       = 15.0;  // [m/s]
    double att_ref_max_tilt_deg = 60.0; // [deg]
};

struct SimParams {
    double dt       = 0.01;     // [s] control period (100 Hz)
    double t_final  = 85.0;     // [s] max simulation time
    double s_max    = 100.0;    // [m] active path length
};

struct NmpcParams {
    double dt           = 0.01;   // [s] control loop period (100 Hz)
    double t_prediction = 1.0;    // [s] prediction horizon

    /// N is computed automatically: N = t_prediction / dt
    int N_horizon() const { return static_cast<int>(t_prediction / dt + 0.5); }

    // Cost weights (default, can be overridden at runtime)
    Vec3 Q_pos   = Vec3(150.0, 150.0, 150.0);
    Vec3 Q_att   = Vec3(50.0, 50.0, 50.0);
    Eigen::Vector4d R_u = Eigen::Vector4d(0.1, 0.5, 0.5, 0.5);
};

struct MpccParams {
    double dt           = 0.01;   // [s] control loop period (100 Hz)
    double t_prediction = 1.5;    // [s] prediction horizon (N=31, non-uniform dt)

    // MPCC weights (runtime parameters, no rebuild needed)
    Vec3 Q_ec   = Vec3(100.0, 100.0, 100.0);      // contouring error
    Vec3 Q_el   = Vec3(10.0, 10.0, 10.0);         // lag error
    Vec3 Q_q    = Vec3(5.0, 5.0, 5.0);            // attitude (quat log)
    Eigen::Vector4d U_mat = Eigen::Vector4d(0.1, 0.3, 0.3, 0.3); // control [T, ωx, ωy, ωz]
    double Q_s  = 12.0;                            // progress
    double W_df = 0.001;                           // thrust rate penalty (Δf²)
};

struct DqMpccParams {
    double dt           = 0.01;   // [s] control loop period (100 Hz)
    double t_prediction = 1.5;    // [s] prediction horizon (N=31, non-uniform dt)

    // DQ-MPCC weights (runtime parameters, no rebuild needed)
    Vec3 Q_phi  = Vec3(5.0, 5.0, 5.0);            // rotation log error
    Vec3 Q_ec   = Vec3(100.0, 100.0, 100.0);      // contouring error (body frame)
    Vec3 Q_el   = Vec3(10.0, 10.0, 10.0);         // lag error (body frame)
    Eigen::Vector4d U_mat = Eigen::Vector4d(0.1, 0.3, 0.3, 0.3); // control [T, ωx, ωy, ωz]
    double Q_s  = 12.0;                            // progress
    double W_df = 0.001;                           // thrust rate penalty (Δf²)
};

}  // namespace quadrotor_mpc
