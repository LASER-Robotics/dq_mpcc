#include "quadrotor_mpc/mujoco/sil_protocol.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace quadrotor_mpc {

SilProtocol::SilProtocol(std::shared_ptr<MujocoInterface> muj, const SilConfig& cfg)
    : muj_(muj), cfg_(cfg) {}

void SilProtocol::set_controller(ControllerCallback cb) {
    controller_ = std::move(cb);
}

void SilProtocol::set_progress_tracker(ProgressCallback cb) {
    progress_tracker_ = std::move(cb);
}

//  Phase 1 & 4: RELOAD

bool SilProtocol::reload_() {
    const auto& tag = cfg_.trial_label;
    RCLCPP_INFO(muj_->get_logger(), "[%s] Phase: RELOAD", tag.c_str());

    // Step 1: Zero all commands to stop any residual actuation
    for (int i = 0; i < 10; ++i) {
        muj_->send_cmd(0.0, 0.0, 0.0, 0.0);
        std::this_thread::sleep_for(10ms);
    }

    // Step 2: Reset MuJoCo (positions + velocities → zero)
    bool ok = muj_->reset_sim();
    if (!ok) {
        RCLCPP_ERROR(muj_->get_logger(), "[%s] Simulator reset failed!", tag.c_str());
        return false;
    }
    muj_->clear_crash();

    // Step 3: Verify clean state (tight thresholds)
    // Send zero commands while waiting — drone is on the ground, no need
    // for hover thrust here.  PD HOLD will lift it afterwards.
    for (int attempt = 0; attempt < 50; ++attempt) {  // 50 × 20ms = 1s max
        muj_->send_cmd(0.0, 0.0, 0.0, 0.0);
        std::this_thread::sleep_for(20ms);
        DroneState st = muj_->get_state();
        double dist_to_origin = st.pos.head<2>().norm();
        if (dist_to_origin < 0.3 && st.vel.norm() < 0.1) {
            muj_->clear_crash();
            break;
        }
        if (attempt % 10 == 9) {
            RCLCPP_WARN(muj_->get_logger(),
                "[%s] RELOAD waiting: pos=[%.2f,%.2f,%.2f] v=%.2f (attempt %d)",
                tag.c_str(), st.pos.x(), st.pos.y(), st.pos.z(),
                st.vel.norm(), attempt + 1);
        }
        // Retry reset if state is still dirty at halfway point
        if (attempt == 25) {
            RCLCPP_WARN(muj_->get_logger(),
                "[%s] RELOAD: state not clean — retrying reset", tag.c_str());
            muj_->reset_sim();
            muj_->clear_crash();
        }
    }
    muj_->clear_crash();

    if (!muj_->is_connected()) {
        if (!muj_->wait_for_connection()) return false;
    }

    DroneState st = muj_->get_state();
    RCLCPP_INFO(muj_->get_logger(),
        "[%s] RELOAD complete — pos=[%.2f,%.2f,%.2f]  v=%.3f m/s",
        tag.c_str(), st.pos.x(), st.pos.y(), st.pos.z(), st.vel.norm());
    return true;
}

//  Phase 2: PD HOLD

bool SilProtocol::pd_hold_() {
    const auto& tag = cfg_.trial_label;
    RCLCPP_INFO(muj_->get_logger(),
        "[%s] Phase: PD HOLD → [%.2f, %.2f, %.2f]",
        tag.c_str(), cfg_.P0.x(), cfg_.P0.y(), cfg_.P0.z());

    muj_->start_pd_hold(cfg_.P0, cfg_.mass, cfg_.gravity);

    bool converged = muj_->wait_for_pd_convergence(
        cfg_.P0, cfg_.pd_settle_dist, cfg_.pd_settle_time, cfg_.pd_timeout);

    muj_->stop_pd_hold();

    // NOTE: crash flag is NOT cleared here. It is cleared once in
    // run_loop_() right before the controller starts, guaranteeing
    // that any collision during flight is always detected.

    if (!converged) {
        RCLCPP_WARN(muj_->get_logger(),
            "[%s] PD did not converge — proceeding anyway", tag.c_str());
    }

    // Verify position one more time
    DroneState st = muj_->get_state();
    double dist = (st.pos - cfg_.P0).norm();
    RCLCPP_INFO(muj_->get_logger(),
        "[%s] PD HOLD complete — dist=%.3f m, pos=[%.2f, %.2f, %.2f]",
        tag.c_str(), dist, st.pos.x(), st.pos.y(), st.pos.z());

    return converged;
}

//  Phase 3: CONTROL LOOP

SilResult SilProtocol::run_loop_() {
    const auto& tag = cfg_.trial_label;
    RCLCPP_INFO(muj_->get_logger(), "[%s] Phase: RUN (controller active)", tag.c_str());

    const int N_max = static_cast<int>(cfg_.t_final / cfg_.dt);
    SilResult res;

    // Pre-allocate
    res.x.resize(13, N_max);
    res.a_imu.resize(3, N_max);
    res.u.resize(4, N_max);
    res.p_ref.resize(3, N_max);
    res.q_ref.resize(4, N_max);
    res.t.resize(N_max);
    res.s.resize(N_max);
    res.solve_ms.resize(N_max);
    res.loop_ms.resize(N_max);
    res.status.resize(N_max);
    res.theta.resize(N_max);
    res.vtheta.resize(N_max);
    res.a_theta.resize(N_max);
    res.s_nearest.resize(N_max);
    res.p_ref_nearest.resize(3, N_max);
    res.q_ref_nearest.resize(4, N_max);

    double s_current = 0.0;
    double stall_timer = 0.0;   // accumulated time with velocity near zero
    bool   stall_armed = false; // don't check stall until drone has started moving
    std::string stop_reason = "timeout";  // default if loop exhausts N_max

    using Clock = std::chrono::steady_clock;

    // Clear crash flag ONE TIME right before control starts
    // PD hold already brought the drone to P0. Any collision from
    // takeoff/ground contact is stale. From here on, any collision
    // topic message is a REAL crash (wall or floor during flight).
    muj_->clear_crash();

    for (int k = 0; k < N_max; ++k) {
        auto tic = Clock::now();
        double t_k = k * cfg_.dt;

        // Read state
        DroneState ds = muj_->get_state();
        ds.quat = ds.quat / (ds.quat.norm() + 1e-12);

        // Update path progress
        if (progress_tracker_) {
            s_current = progress_tracker_(ds.pos, s_current);
        }

        // Check termination: path complete
        if (s_current >= cfg_.s_max - cfg_.s_done_tol) {
            stop_reason = "PATH_COMPLETE";
            res.completed = true;
            res.n_steps = k;
            res.t_final = t_k;
            res.s_final = s_current;
            break;
        }

        // Call controller
        ControlOutput out = controller_(ds, t_k, s_current);

        if (out.solver_status != 0) ++res.solve_fails;

        // Send command to MuJoCo
        muj_->send_cmd(out.cmd);

        // Log
        res.x.col(k)     = ds.to_vector();
        res.a_imu.col(k) = ds.accel;
        res.u.col(k)     << out.cmd.thrust, out.cmd.omega_cmd;
        res.p_ref.col(k) = out.p_ref;
        res.q_ref.col(k) = out.q_ref;
        res.t(k)         = t_k;
        res.s(k)         = s_current;
        res.solve_ms(k)  = out.solve_time_s * 1e3;
        res.status(k)    = out.solver_status;
        res.theta(k)     = out.theta;
        res.vtheta(k)    = out.vtheta;
        res.a_theta(k)   = out.a_theta;
        res.s_nearest(k)       = out.s_nearest;
        res.p_ref_nearest.col(k) = out.p_ref_nearest;
        res.q_ref_nearest.col(k) = out.q_ref_nearest;
        res.n_steps      = k + 1;
        res.t_final      = t_k;
        res.s_final      = s_current;

        // Rate control
        double elapsed = std::chrono::duration<double>(Clock::now() - tic).count();
        double remaining = cfg_.dt - elapsed;
        if (remaining > 0.0)
            std::this_thread::sleep_for(std::chrono::duration<double>(remaining));

        double loop_ms_k = std::chrono::duration<double, std::milli>(
            Clock::now() - tic).count();
        res.loop_ms(k) = loop_ms_k;

        // Periodic log
        if (k % static_cast<int>(5.0 / cfg_.dt) == 0) {
            double e_pos = (ds.pos - out.p_ref).norm();
            RCLCPP_INFO(muj_->get_logger(),
                "[%s] t=%.1f  s=%.1f/%.0f  |e|=%.3f  solve=%.2f ms  "
                "loop=%.1f ms  fails=%d",
                tag.c_str(), t_k, s_current, cfg_.s_max, e_pos,
                out.solve_time_s*1e3, loop_ms_k, res.solve_fails);
        }

        // Crash check: collision topic (walls & floor)
        // Flag was cleared once before loop. Never cleared again during
        // RUN — any collision sets it permanently until next trial.
        if (cfg_.abort_on_crash && muj_->is_crashed()) {
            stop_reason = "COLLISION (MuJoCo contact)";
            RCLCPP_WARN(muj_->get_logger(),
                "[%s] STOP: %s at t=%.1f s  pos=[%.2f, %.2f, %.2f]",
                tag.c_str(), stop_reason.c_str(), t_k, ds.pos.x(), ds.pos.y(), ds.pos.z());
            res.crashed = true;
            res.n_steps = k + 1;
            break;
        }

        // Floor crash (z too low during flight)
        if (ds.pos.z() < cfg_.z_min) {
            stop_reason = "FLOOR_CRASH (z < " + std::to_string(cfg_.z_min) + "m)";
            RCLCPP_WARN(muj_->get_logger(),
                "[%s] STOP: %s  z=%.3f at t=%.1f s",
                tag.c_str(), stop_reason.c_str(), ds.pos.z(), t_k);
            res.crashed = true;
            res.n_steps = k + 1;
            break;
        }

        // Height check (escaped workspace)
        if (ds.pos.z() > cfg_.z_max) {
            stop_reason = "CEILING_CRASH (z > " + std::to_string(cfg_.z_max) + "m)";
            RCLCPP_WARN(muj_->get_logger(),
                "[%s] STOP: %s  z=%.1f at t=%.1f s",
                tag.c_str(), stop_reason.c_str(), ds.pos.z(), t_k);
            res.crashed = true;
            res.n_steps = k + 1;
            break;
        }

        // Stall detection (velocity near zero for too long)
        {
            double v_norm = ds.vel.norm();
            // Arm stall detection once the drone reaches meaningful speed
            if (!stall_armed && v_norm > 1.0) {
                stall_armed = true;
            }
            if (stall_armed) {
                if (v_norm < cfg_.stall_vel_thr) {
                    stall_timer += cfg_.dt;
                } else {
                    stall_timer = 0.0;
                }
                if (stall_timer >= cfg_.stall_timeout) {
                    stop_reason = "STALL (v < " + std::to_string(cfg_.stall_vel_thr)
                                + " m/s for " + std::to_string(cfg_.stall_timeout) + "s)";
                    RCLCPP_WARN(muj_->get_logger(),
                        "[%s] STOP: %s  v=%.3f at t=%.1f s (s=%.1f m)",
                        tag.c_str(), stop_reason.c_str(), v_norm, t_k, s_current);
                    res.stalled = true;
                    res.n_steps = k + 1;
                    break;
                }
            }
        }
    }

    // Store stop reason in result
    res.stop_reason = stop_reason;

    RCLCPP_INFO(muj_->get_logger(),
        "[%s] RUN ended — reason: %s  t=%.1f s  s=%.1f m",
        tag.c_str(), stop_reason.c_str(), res.t_final, res.s_final);

    // Post-RUN: motors off (production) OR hard-reset (tuning)
    const bool clean_finish = (stop_reason == "PATH_COMPLETE");
    if (cfg_.post_run_hold && clean_finish) {
        // Production: MPCC already landed the drone at hover via the exit
        // phase of the planner reference. Just kill motors and stop — no PD,
        // no reset, no mixing with another controller.
        RCLCPP_INFO(muj_->get_logger(),
            "[%s] Post-RUN: motors OFF (drone stays where MPCC left it)", tag.c_str());
        muj_->send_zero();
    } else {
        // Tuning (or crash/timeout): hard-reset MuJoCo to clean state.
        // Zero commands to stop actuation, then reset.
        for (int i = 0; i < 10; ++i) {          // 100 ms of zero commands
            muj_->send_cmd(0.0, 0.0, 0.0, 0.0);
            std::this_thread::sleep_for(10ms);
        }
        muj_->reset_sim();
        muj_->clear_crash();
        for (int attempt = 0; attempt < 50; ++attempt) {
            muj_->send_cmd(0.0, 0.0, 0.0, 0.0);
            std::this_thread::sleep_for(20ms);
            DroneState st = muj_->get_state();
            if (st.pos.head<2>().norm() < 0.3 && st.vel.norm() < 0.1) break;
            if (attempt == 49) {
                RCLCPP_WARN(muj_->get_logger(),
                    "[%s] Post-RUN reset: state not clean v=%.2f — retrying reset",
                    tag.c_str(), st.vel.norm());
                muj_->reset_sim();
                muj_->clear_crash();
            }
        }
        RCLCPP_INFO(muj_->get_logger(),
            "[%s] Post-RUN reset complete — clean state", tag.c_str());
    }

    // Trim logs to actual steps
    int n = res.n_steps;
    res.x.conservativeResize(Eigen::NoChange, n);
    res.a_imu.conservativeResize(Eigen::NoChange, n);
    res.u.conservativeResize(Eigen::NoChange, n);
    res.p_ref.conservativeResize(Eigen::NoChange, n);
    res.q_ref.conservativeResize(Eigen::NoChange, n);
    res.t.conservativeResize(n);
    res.s.conservativeResize(n);
    res.solve_ms.conservativeResize(n);
    res.loop_ms.conservativeResize(n);
    res.status.conservativeResize(n);
    res.theta.conservativeResize(n);
    res.vtheta.conservativeResize(n);
    res.a_theta.conservativeResize(n);
    res.s_nearest.conservativeResize(n);
    res.p_ref_nearest.conservativeResize(Eigen::NoChange, n);
    res.q_ref_nearest.conservativeResize(Eigen::NoChange, n);

    return res;
}

//  execute(): full protocol

SilResult SilProtocol::execute() {
    const auto& tag = cfg_.trial_label;
    RCLCPP_INFO(muj_->get_logger(),
        "\n╔══════════════════════════════════════════════════╗\n"
        "║  SiL PROTOCOL — %-33s ║\n"
        "║  P0 = [%6.2f, %6.2f, %5.2f]                     ║\n"
        "║  s_max = %.0f m   dt = %.0f ms   t_max = %.0f s          ║\n"
        "╚══════════════════════════════════════════════════╝",
        tag.c_str(),
        cfg_.P0.x(), cfg_.P0.y(), cfg_.P0.z(),
        cfg_.s_max, cfg_.dt * 1e3, cfg_.t_final);

    // Phase 1: RELOAD
    if (!reload_()) {
        RCLCPP_ERROR(muj_->get_logger(), "[%s] RELOAD failed — aborting", tag.c_str());
        return {};
    }

    // Phase 2: PD HOLD
    pd_hold_();

    // Phase 3: RUN
    SilResult res = run_loop_();

    // No reload here — drone stays visible. Next execute() starts with reload.
    RCLCPP_INFO(muj_->get_logger(), "[%s] Drone stabilised at P0 — ready", tag.c_str());

    // Summary
    int n = res.n_steps;
    if (n > 0) {
        RCLCPP_INFO(muj_->get_logger(),
            "\n╔══════════════════════════════════════════════════╗\n"
            "║  SiL SUMMARY — %-33s ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Stop reason: %-34s ║\n"
            "║  Steps:       %6d                               ║\n"
            "║  Duration:    %6.1f s                              ║\n"
            "║  Progress:    %5.1f / %.0f m                        ║\n"
            "║  Completed:   %s                                   ║\n"
            "║  Crashed:     %s                                   ║\n"
            "║  Stalled:     %s                                   ║\n"
            "║  Solve fails: %d / %d                              ║\n"
            "║  Solve avg:   %.2f ms                              ║\n"
            "║  Solve max:   %.2f ms                              ║\n"
            "╚══════════════════════════════════════════════════╝",
            tag.c_str(),
            res.stop_reason.c_str(),
            n, res.t_final, res.s_final, cfg_.s_max,
            res.completed ? "YES" : "NO",
            res.crashed   ? "YES" : "NO",
            res.stalled   ? "YES" : "NO",
            res.solve_fails, n,
            res.solve_ms.mean(), res.solve_ms.maxCoeff());
    }

    return res;
}

}  // namespace quadrotor_mpc
