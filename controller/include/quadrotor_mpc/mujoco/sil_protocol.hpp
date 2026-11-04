#pragma once
/**
 * SiL Protocol — Standard execution sequence for MuJoCo simulations.
 *
 * Reusable by NMPC, MPCC, DQ-MPCC. The controller only needs to provide
 * a callback that receives the current state and returns a command.
 *
 * Protocol:
 *   1. RELOAD  — reset simulator, clear crash flag, wait for odom
 *   2. PD HOLD — fly to P0, wait until converged within tolerance
 *   3. RUN     — call controller at 100 Hz until path done / crash / timeout
 *   4. RELOAD  — reset simulator again, ready for next run
 *
 * Usage:
 *   SilProtocol proto(muj, params);
 *   proto.set_controller([&](const DroneState& s, double t) {
 *       // solve your OCP, return AcroCommand
 *   });
 *   auto result = proto.execute();
 */

#include "quadrotor_mpc/mujoco/mujoco_interface.hpp"
#include "quadrotor_mpc/common/types.hpp"

#include <Eigen/Dense>
#include <functional>
#include <string>

namespace quadrotor_mpc {

struct SilConfig {
    std::string trial_label = "";                 // e.g. "MPCC v=10 run03"
    Vec3   P0            = Vec3(2.5, 0.0, 1.5);  // initial position
    double mass          = 1.08;
    double gravity       = 9.81;
    double dt            = 0.01;        // control period [s] (100 Hz)
    double t_final       = 85.0;        // max simulation time [s]
    double s_max         = 100.0;       // path length to complete [m]

    // PD convergence
    double pd_settle_dist = 0.30;       // [m]
    double pd_settle_time = 1.0;        // [s]
    double pd_timeout     = 20.0;       // [s]

    // Termination
    double s_done_tol     = 0.1;        // path complete when s >= s_max - tol
    bool   abort_on_crash = true;
    double z_max          = 5.0;        // [m] abort if drone exceeds this height
    double z_min          = 0.15;       // [m] abort if drone drops below (floor crash)
    double stall_vel_thr  = 0.3;        // [m/s] velocity below this = "stalled"
    double stall_timeout  = 5.0;        // [s] abort if stalled for this long

    // Post-RUN behaviour
    //  false (default, tuning): hard-reset MuJoCo to zero state, drone snaps
    //                            back to origin. Safe when drone may be crashed.
    //  true  (production):      keep drone where it is and PD-hold at P0 so
    //                            it stays visible at hover for inspection.
    //                            Only meaningful when stop_reason==PATH_COMPLETE.
    bool   post_run_hold  = false;
};

/// Result of a single SiL run
struct SilResult {
    bool   completed  = false;
    bool   crashed    = false;
    bool   stalled    = false;
    std::string stop_reason = "timeout";
    int    n_steps    = 0;
    double t_final    = 0.0;
    double s_final    = 0.0;
    int    solve_fails = 0;

    // Log matrices (columns = time steps)
    Eigen::MatrixXd x;          // (13, n_steps) states
    Eigen::MatrixXd a_imu;      // (3, n_steps) body-frame accel from /imu [m/s²]
    Eigen::MatrixXd u;          // (4, n_steps) controls
    Eigen::MatrixXd p_ref;      // (3, n_steps) position reference
    Eigen::MatrixXd q_ref;      // (4, n_steps) quaternion reference
    Eigen::VectorXd t;          // (n_steps) time
    Eigen::VectorXd s;          // (n_steps) arc-length progress
    Eigen::VectorXd solve_ms;   // (n_steps) solver time
    Eigen::VectorXd loop_ms;    // (n_steps) full loop time (solve + publish + sleep)
    Eigen::VectorXi status;     // (n_steps) solver status per step
    // Virtual states (MPCC / DQ-MPCC)
    Eigen::VectorXd theta;      // (n_steps) arc-length progress θ [m]
    Eigen::VectorXd vtheta;     // (n_steps) progress velocity v_θ [m/s]
    Eigen::VectorXd a_theta;    // (n_steps) progress acceleration a_θ [m/s²]
    // Geometrically nearest reference (for paper metrics)
    Eigen::VectorXd s_nearest;  // (n_steps) nearest arc-length [m]
    Eigen::MatrixXd p_ref_nearest; // (3, n_steps) position ref at s_nearest
    Eigen::MatrixXd q_ref_nearest; // (4, n_steps) quaternion ref at s_nearest
};

/// Controller callback: receives state + time, returns command + solver status
struct ControlOutput {
    AcroCommand cmd;
    double      solve_time_s = 0.0;  // wall-clock solver time [s]
    int         solver_status = 0;   // 0 = success
    Vec3        p_ref = Vec3::Zero();
    Quat4       q_ref = Quat4(1,0,0,0);
    // Virtual states (MPCC / DQ-MPCC)
    double      theta   = 0.0;      // arc-length progress [m]
    double      vtheta  = 0.0;      // progress velocity [m/s]
    double      a_theta = 0.0;      // progress acceleration [m/s²]
    // Geometrically nearest reference (for paper metrics)
    double      s_nearest = 0.0;    // nearest arc-length [m]
    Vec3        p_ref_nearest = Vec3::Zero();
    Quat4       q_ref_nearest = Quat4(1,0,0,0);
};

using ControllerCallback = std::function<ControlOutput(
    const DroneState& state, double t, double s_current)>;

class SilProtocol {
public:
    SilProtocol(std::shared_ptr<MujocoInterface> muj, const SilConfig& cfg);

    /// Set the controller callback (must be set before execute())
    void set_controller(ControllerCallback cb);

    /// Set function to track path progress (receives position, returns arc-length)
    using ProgressCallback = std::function<double(const Vec3& pos, double s_prev)>;
    void set_progress_tracker(ProgressCallback cb);

    /// Execute the full protocol: reload → PD → run → reload
    SilResult execute();

private:
    std::shared_ptr<MujocoInterface> muj_;
    SilConfig cfg_;
    ControllerCallback controller_;
    ProgressCallback progress_tracker_;

    /// Phase 1 & 4: reset simulator and clear state
    bool reload_();

    /// Phase 2: PD hold to P0 and wait convergence
    bool pd_hold_();

    /// Phase 3: main control loop
    SilResult run_loop_();
};

}  // namespace quadrotor_mpc
