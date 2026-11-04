/**
 * dq_mpcc_ros2_node.cpp  —  DQ-MPCC node for the integrated flight stack.
 *
 * Same structure as mpcc_ros2_node.cpp for the dual-quaternion controller
 * (17-dimensional state: [dq(8), ω, v_b, θ, v_θ, f]).
 * Normally started from launch/experiment.launch.py.
 *
 * ROS 2 parameters (config/dq_mpcc_params.yaml):
 *   vtheta_max   [double]  maximum progress speed [m/s]
 *   weights_file [string]  JSON gain file; "" = default deployed gains
 *   quad_name    [string]  vehicle namespace
 *
 * Initialization on activation:
 *   x0 = odometry snapshot converted to a dual quaternion
 *   θ  = 0,  v_θ = 0
 */

#include "quadrotor_mpc/dq_mpcc_ablation/dq_mpcc_ablation_controller.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include "quadrotor_mpc/common/params.hpp"
#include "quadrotor_mpc/trajectory/lissajous.hpp"
#include "quadrotor_mpc/trajectory/planner_ref.hpp"
#include "quadrotor_mpc/trajectory/arc_length.hpp"
#include "quadrotor_mpc/trajectory/attitude_reference.hpp"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <quadrotor_msgs/msg/trpy_command.hpp>
#include <mav_manager_srv/srv/vec4.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <mutex>
#include <atomic>
#include <cmath>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

using namespace quadrotor_mpc;

// Odometry snapshot

struct OdomSnap {
    Vec3  pos   = Vec3::Zero();
    Vec3  vel   = Vec3::Zero();
    Quat4 quat  = Quat4(1, 0, 0, 0);
    Vec3  omega = Vec3::Zero();
    bool  valid = false;
};

// Per-step log (same CSV format as dq_mpcc_sil)

struct StepLog {
    double t;
    double px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz;
    double T, wx_cmd, wy_cmd, wz_cmd;
    double solve_ms, loop_ms, s;
    double px_ref, py_ref, pz_ref, qw_ref, qx_ref, qy_ref, qz_ref;
    double theta, vtheta, a_theta;
    double s_nearest;
    double px_ref_sn, py_ref_sn, pz_ref_sn, qw_ref_sn, qx_ref_sn, qy_ref_sn, qz_ref_sn;
    int status;
};

// Conversion to dual quaternion (same as dq_mpcc_sil.cpp)

static State17 odom_to_dq17(const OdomSnap& s, double theta, double vtheta, double f)
{
    // Rotation quaternion [qw, qx, qy, qz]
    double qr_w = s.quat(0), qr_x = s.quat(1),
           qr_y = s.quat(2), qr_z = s.quat(3);

    // Dual part: qd = ½ [0, px, py, pz] ⊗ qr
    double tx = s.pos(0), ty = s.pos(1), tz = s.pos(2);
    double qd_w = 0.5*(-tx*qr_x - ty*qr_y - tz*qr_z);
    double qd_x = 0.5*( tx*qr_w + ty*qr_z - tz*qr_y);
    double qd_y = 0.5*(-tx*qr_z + ty*qr_w + tz*qr_x);
    double qd_z = 0.5*( tx*qr_y - ty*qr_x + tz*qr_w);

    // Body-frame velocity: v_b = R^T · v_inertial
    Eigen::Matrix3d R = quat_to_rot(s.quat);
    Eigen::Vector3d v_body = R.transpose() * s.vel;

    State17 x;
    x(0)  = qr_w;  x(1)  = qr_x;  x(2)  = qr_y;  x(3)  = qr_z;
    x(4)  = qd_w;  x(5)  = qd_x;  x(6)  = qd_y;  x(7)  = qd_z;
    x(8)  = s.omega(0);  x(9)  = s.omega(1);  x(10) = s.omega(2);
    x(11) = v_body(0);   x(12) = v_body(1);   x(13) = v_body(2);
    x(14) = theta;
    x(15) = vtheta;
    x(16) = f;
    return x;
}

// Gain loading from JSON

// Apply weights from JSON trial (prefers params, fallback weights)
static void apply_weights_json_dq(const nlohmann::json& pt, DqMpccWeights& w) {
    if (pt.contains("params")) {
        const auto& p = pt["params"];
        double qphi=p["Q_phi"]; double qec=p["Q_ec"]; double qel=p["Q_el"];
        double ut=p["U_T"]; double uw=p["U_w"]; double qs=p["Q_s"];
        w.Q_phi=Vec3(qphi,qphi,qphi); w.Q_ec=Vec3(qec,qec,qec); w.Q_el=Vec3(qel,qel,qel);
        w.U_mat=Eigen::Vector4d(ut,uw,uw,uw); w.Q_s=qs;
        if (p.contains("W_df")) w.W_df=p["W_df"];
    } else if (pt.contains("weights")) {
        const auto& wj = pt["weights"];
        double qphi=wj["Q_phi"][0]; double qec=wj["Q_ec"][0]; double qel=wj["Q_el"][0];
        double ut=wj["U_mat"][0]; double uw=wj["U_mat"][1]; double qs=wj["Q_s"];
        w.Q_phi=Vec3(qphi,qphi,qphi); w.Q_ec=Vec3(qec,qec,qec); w.Q_el=Vec3(qel,qel,qel);
        w.U_mat=Eigen::Vector4d(ut,uw,uw,uw); w.Q_s=qs;
        if (wj.contains("W_df")) w.W_df=wj["W_df"];
    }
}

static bool load_pareto_best(const std::string& path, DqMpccWeights& w,
                             double* vtheta_max_out, int* trial_out)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    if (!j.contains("front") || j["front"].empty()) return false;
    int best_idx = 0; double best_obj = 1e18;
    auto& front = j["front"];
    for (int i = 0; i < (int)front.size(); ++i) {
        double o = front[i].value("lap_time_min", 1e18);
        if (o < best_obj) { best_obj = o; best_idx = i; }
    }
    auto& pt = front[best_idx];
    apply_weights_json_dq(pt, w);
    if (vtheta_max_out && pt.contains("params") && pt["params"].contains("vtheta_max"))
        *vtheta_max_out = pt["params"]["vtheta_max"];
    if (trial_out) *trial_out = pt.value("trial", -1);
    return true;
}

static bool load_weights_json(const std::string& path, DqMpccWeights& w,
                              double* vtheta_max_out = nullptr)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j  = nlohmann::json::parse(f);
    auto wj = j["weights"];
    double qphi = wj["Q_phi"][0];
    double qec  = wj["Q_ec"][0];
    double qel  = wj["Q_el"][0];
    double ut   = wj["U_mat"][0];
    double uw   = wj["U_mat"][1];
    double qs   = wj["Q_s"];
    w.Q_phi = Vec3(qphi, qphi, qphi);
    w.Q_ec  = Vec3(qec,  qec,  qec);
    w.Q_el  = Vec3(qel,  qel,  qel);
    w.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
    w.Q_s   = qs;
    if (wj.contains("W_df")) w.W_df = wj["W_df"];
    if (vtheta_max_out && j.contains("vtheta_max")) {
        *vtheta_max_out = j["vtheta_max"];
    }
    return true;
}


class DqMpccAblationNode : public rclcpp::Node
{
public:
    DqMpccAblationNode() : Node("dq_mpcc_ablation_node")
    {
        // Parameters
        declare_parameter("quad_name",    "quadrotor");
        declare_parameter("weights_file", "");

        const std::string quad  = get_parameter("quad_name").as_string();
        const std::string wfile = get_parameter("weights_file").as_string();
        const std::string ns    = "/" + quad;

        // Reference: same as SiL (planner_ref)
        RCLCPP_INFO(get_logger(), "Loading the planner reference...");
        const std::string pkg_share = ament_index_cpp::get_package_share_directory("quadrotor_mpc");
        const std::string ref_json  = pkg_share + "/path_planing/ref_flat_waypoints.json";
        PlannerRef pref;
        if (!load_planner_ref(ref_json, pref)) {
            RCLCPP_FATAL(get_logger(),
                "Could not load %s; run path_planing/export_ref_json.py", ref_json.c_str());
            throw std::runtime_error("planner_ref load failed");
        }
        wp_    = pref.wp;
        s_max_ = pref.wp.s_max;

        QuadParams qp;
        theta_margin_ = qp.theta_margin;
        mass_g_       = qp.mass * qp.g;

        RCLCPP_INFO(get_logger(),
            "Planner ref: s_max=%.1f m  hover=[%.2f,%.2f,%.2f]  T_lap=%.3fs",
            s_max_, pref.hover_pos(0), pref.hover_pos(1), pref.hover_pos(2),
            pref.T_lap);

        // Solver
        if (!ctrl_.init()) {
            RCLCPP_FATAL(get_logger(), "DQ-MPCC solver initialization failed");
            throw std::runtime_error("DQ-MPCC init failed");
        }

        // Gains: load the JSON file given by the launch file
        // The gain file path is set in launch/experiment.launch.py.
        double json_vth = -1.0;
        int trial_id = -1;
        bool loaded = load_pareto_best(wfile, weights_, &json_vth, &trial_id);
        if (loaded) {
            RCLCPP_INFO(get_logger(),
                "Gains loaded from: %s (trial %d)", wfile.c_str(), trial_id);
        } else {
            RCLCPP_FATAL(get_logger(),
                "Could not load gain file '%s'; aborting", wfile.c_str());
            throw std::runtime_error("DQ-MPCC pareto load failed");
        }
        if (loaded && json_vth > 0) {
            weights_.vtheta_max = json_vth;
            RCLCPP_INFO(get_logger(),
                "vtheta_max from JSON Pareto knee: %.2f m/s", json_vth);
        }


        ctrl_.set_params_all(weights_);

        // Print the loaded weights
        const bool is_manual_json = (wfile.find("manual") != std::string::npos);
        if (is_manual_json)
            RCLCPP_WARN(get_logger(), "═══ DQ-MPCC MANUAL weights (pareto_dq_mpcc_ablation_manual.json) ═══");
        else
            RCLCPP_INFO(get_logger(), "═══ DQ-MPCC weights loaded (JSON) ═══");
        RCLCPP_INFO(get_logger(), "  Q_phi       = %.4f", weights_.Q_phi(0));
        RCLCPP_INFO(get_logger(), "  Q_ec        = %.4f", weights_.Q_ec(0));
        RCLCPP_INFO(get_logger(), "  Q_el        = %.4f", weights_.Q_el(0));
        RCLCPP_INFO(get_logger(), "  U_T (mat[0])= %.4f", weights_.U_mat(0));
        RCLCPP_INFO(get_logger(), "  U_w (mat[1])= %.4f", weights_.U_mat(1));
        RCLCPP_INFO(get_logger(), "  Q_s         = %.4f", weights_.Q_s);
        RCLCPP_INFO(get_logger(), "  W_df        = %.4f", weights_.W_df);
        RCLCPP_INFO(get_logger(), "  vtheta_max  = %.4f m/s", weights_.vtheta_max);
        RCLCPP_INFO(get_logger(), "═════════════════════════════════");

        // ROS2 I/O
        cmd_pub_ = create_publisher<quadrotor_msgs::msg::TRPYCommand>(
            ns + "/trpy_cmd", 10);

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            ns + "/odom", 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg){ odom_cb(msg); });

        motors_sub_ = create_subscription<std_msgs::msg::Bool>(
            ns + "/motors", 10,
            [this](std_msgs::msg::Bool::SharedPtr msg){ motors_on_ = msg->data; });

        cb_timer_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_srv_   = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        srv_ = create_service<std_srvs::srv::SetBool>(
            ns + "/activate_dq_mpcc_ablation",
            [this](std_srvs::srv::SetBool::Request::SharedPtr  req,
                   std_srvs::srv::SetBool::Response::SharedPtr res){ activate_cb(req, res); },
            rmw_qos_profile_services_default,
            cb_srv_);

        pd_cli_     = create_client<std_srvs::srv::SetBool>(ns + "/activate_pd");
        land_cli_   = create_client<std_srvs::srv::Trigger>(ns + "/mav_services/land");
        goto_cli_   = create_client<mav_manager_srv::srv::Vec4>(ns + "/mav_services/goTo");
        motors_cli_ = create_client<std_srvs::srv::SetBool>(ns + "/mav_services/motors");

        timer_ = create_wall_timer(std::chrono::milliseconds(10),
                                   [this](){ control_loop(); },
                                   cb_timer_);

        RCLCPP_INFO(get_logger(), "dq_mpcc_ablation_node on standby — namespace: %s", ns.c_str());
    }

private:

    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(odom_mtx_);
        auto& p = msg->pose.pose.position;
        auto& v = msg->twist.twist.linear;
        auto& q = msg->pose.pose.orientation;
        auto& w = msg->twist.twist.angular;
        odom_.pos   = {p.x, p.y, p.z};
        odom_.vel   = {v.x, v.y, v.z};
        odom_.quat  = {q.w, q.x, q.y, q.z};
        odom_.omega = {w.x, w.y, w.z};
        odom_.valid = true;
    }

    void activate_cb(std_srvs::srv::SetBool::Request::SharedPtr  req,
                     std_srvs::srv::SetBool::Response::SharedPtr res)
    {
        const bool was_active = active_.exchange(req->data);

        if (req->data && !was_active) {
            motors_on_ = true;  // see mpcc_ros2_node.cpp note
            {
                std::lock_guard<std::mutex> lk(odom_mtx_);
                init_snap_ = odom_;
            }
            // Match Mode A SiL exactly: start from theta=0, vtheta=0, f=hover.
            // Racing controller handles entry/warmup transition from hover state.
            theta_current_  = 0.0;
            vtheta_current_ = 0.0;
            f_current_      = mass_g_;  // hover thrust
            s_nearest_prev_ = 0.0;
            first_call_     = true;
            decelerating_   = false;
            last_loop_time_ = {};
            log_.clear();
            log_.reserve(2000);
            t_start_ = std::chrono::steady_clock::now();
            RCLCPP_INFO(get_logger(),
                "DQ-MPCC ACTIVATED, pos=[%.2f %.2f %.2f]  vtheta_max=%.1f",
                init_snap_.pos.x(), init_snap_.pos.y(), init_snap_.pos.z(),
                weights_.vtheta_max);

            // Wait for ≥2 publish ticks before returning so caller can
            // safely deactivate PD without leaving a no-command gap.
            const auto deadline = std::chrono::steady_clock::now()
                                  + std::chrono::milliseconds(500);
            const int target = publish_counter_.load() + 2;
            while (publish_counter_.load() < target &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            RCLCPP_INFO(get_logger(),
                "DQ-MPCC publishing confirmed (counter=%d)",
                publish_counter_.load());
        } else if (!req->data && was_active) {
            send_zero();
            RCLCPP_INFO(get_logger(), "DQ-MPCC DEACTIVATED");
        }

        res->success = true;
        res->message = req->data ? "ACTIVATED" : "DEACTIVATED";
    }

    void control_loop()
    {
        if (!active_) return;
        auto now = std::chrono::steady_clock::now();
        const double dt_real_ms = last_loop_time_.time_since_epoch().count() == 0
            ? 10.0
            : std::chrono::duration<double, std::milli>(now - last_loop_time_).count();
        last_loop_time_ = now;

        OdomSnap odom;
        {
            std::lock_guard<std::mutex> lk(odom_mtx_);
            odom = odom_;
        }
        if (!odom.valid) return;

        if (!motors_on_) {
            send_zero();
            return;
        }

        // Quat normalization (match SiL): ROS odom quat may be slightly non-unit
        OdomSnap odom_safe = odom;
        odom_safe.quat = odom.quat / (odom.quat.norm() + 1e-12);

        // (ARM eps quat-perturb + vth=0.5 force REMOVED — caused Mode B divergence vs SiL)

        // 17-dimensional state: [dq(8), ω, v_b, θ, v_θ, f]
        State17 x = odom_to_dq17(odom_safe, theta_current_, vtheta_current_, f_current_);

        ctrl_.set_x0(x);
        ctrl_.set_params_all(weights_);

        // Warm-up on the first iteration after activation
        if (first_call_) {
            constexpr double v0 = 2.0;   // virtual progress seed (match SiL warm-start)
            const int Nh = DqMpccAblationController::N;

            // Seed the horizon from the current state (as in SiL)
            for (int k = 0; k <= Nh; ++k) {
                State17 xk = x;
                double tk = k * 0.01;
                xk(14) = std::clamp(theta_current_ + v0 * tk, 0.0, s_max_ + theta_margin_); // θ
                xk(15) = v0;  // v_θ
                xk(16) = f_current_;
                ctrl_.set_x_init(k, xk);
                if (k < Nh) {
                    Control5 uk;
                    uk << 0.0, 0.0, 0.0, 0.0, 0.0;  // Δf=0, zero rates, zero a_θ
                    ctrl_.set_u_init(k, uk);
                }
            }
            // DIAG: log init state + warmup solver statuses
            {
                double qn = std::sqrt(init_snap_.quat(0)*init_snap_.quat(0) +
                                      init_snap_.quat(1)*init_snap_.quat(1) +
                                      init_snap_.quat(2)*init_snap_.quat(2) +
                                      init_snap_.quat(3)*init_snap_.quat(3));
                RCLCPP_WARN(get_logger(),
                    "[DIAG] init_snap pos=[%.4f,%.4f,%.4f] quat=[%.4f,%.4f,%.4f,%.4f] |q|=%.6f vth_max=%.2f",
                    init_snap_.pos(0), init_snap_.pos(1), init_snap_.pos(2),
                    init_snap_.quat(0), init_snap_.quat(1), init_snap_.quat(2), init_snap_.quat(3),
                    qn, weights_.vtheta_max);
                RCLCPP_WARN(get_logger(),
                    "[DIAG] x dq=[%.3f,%.3f,%.3f,%.3f / %.3f,%.3f,%.3f,%.3f] f=%.3f",
                    x(0),x(1),x(2),x(3),
                    x(4),x(5),x(6),x(7), f_current_);
            }
            int n_fail = 0, n_ok = 0;
            for (int i = 0; i < N_WARMUP; ++i) {
                int s = ctrl_.solve();
                if (s == 0) n_ok++; else n_fail++;
            }
            Control5 u_warm = ctrl_.get_u0();
            RCLCPP_WARN(get_logger(),
                "[DIAG] WARMUP done: n_ok=%d n_fail=%d  u_last=[Δf=%.3f, ω=%.3f,%.3f,%.3f, aθ=%.3f]",
                n_ok, n_fail, u_warm(0), u_warm(1), u_warm(2), u_warm(3), u_warm(4));
            first_call_ = false;
        }

        auto solve_start = std::chrono::steady_clock::now();
        const int    status = ctrl_.solve();
        const double solve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - solve_start).count();
        const Control5 u   = ctrl_.get_u0();

        // DIAG: log first 5 solves post-warmup + log on any status fail
        static int diag_count = 0;
        if (diag_count < 5 || status != 0) {
            RCLCPP_WARN(get_logger(),
                "[DIAG#%d] status=%d u=[Δf=%.3f, ω=%.3f,%.3f,%.3f, aθ=%.3f] θ=%.2f vth=%.2f f=%.2f",
                diag_count, status, u(0), u(1), u(2), u(3), u(4),
                theta_current_, vtheta_current_, f_current_);
        }
        diag_count++;

        // Integrate θ, v_θ and f locally
        constexpr double dt = 0.01;
        const double delta_f = u(0);
        const double a_theta = u(4);
        theta_current_  = std::clamp(theta_current_ + vtheta_current_ * dt,
                                     0.0, s_max_ + theta_margin_);
        vtheta_current_ = std::clamp(vtheta_current_ + a_theta * dt,
                                     0.0, weights_.vtheta_max);
        QuadParams qp_lim;
        f_current_      = std::clamp(f_current_ + delta_f * dt,
                                     qp_lim.T_min, qp_lim.T_max);

        s_nearest_prev_ = find_nearest_s(odom.pos, wp_, s_nearest_prev_, s_max_);

        // References and per-step logging
        {
            Vec3  p_ref_th = pos_interp(theta_current_,   wp_);
            Quat4 q_ref_th = quat_interp(theta_current_,  wp_);
            Vec3  p_ref_sn = pos_interp(s_nearest_prev_,  wp_);
            Quat4 q_ref_sn = quat_interp(s_nearest_prev_, wp_);
            const double t_now   = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start_).count();
            const double loop_ms = dt_real_ms;
            StepLog e;
            e.t       = t_now;
            e.px      = odom.pos(0);  e.py  = odom.pos(1);  e.pz  = odom.pos(2);
            e.vx      = odom.vel(0);  e.vy  = odom.vel(1);  e.vz  = odom.vel(2);
            e.qw      = odom.quat(0); e.qx  = odom.quat(1); e.qy  = odom.quat(2); e.qz = odom.quat(3);
            e.wx      = odom.omega(0);e.wy  = odom.omega(1);e.wz  = odom.omega(2);
            e.T       = f_current_; e.wx_cmd = u(1); e.wy_cmd = u(2); e.wz_cmd = u(3);
            e.solve_ms = solve_ms;  e.loop_ms = loop_ms;
            e.s        = s_nearest_prev_;
            e.px_ref   = p_ref_th(0); e.py_ref = p_ref_th(1); e.pz_ref = p_ref_th(2);
            e.qw_ref   = q_ref_th(0); e.qx_ref = q_ref_th(1); e.qy_ref = q_ref_th(2); e.qz_ref = q_ref_th(3);
            e.theta    = theta_current_;  e.vtheta = vtheta_current_;  e.a_theta = a_theta;
            e.s_nearest    = s_nearest_prev_;
            e.px_ref_sn    = p_ref_sn(0); e.py_ref_sn = p_ref_sn(1); e.pz_ref_sn = p_ref_sn(2);
            e.qw_ref_sn    = q_ref_sn(0); e.qx_ref_sn = q_ref_sn(1); e.qy_ref_sn = q_ref_sn(2); e.qz_ref_sn = q_ref_sn(3);
            e.status   = status;
            log_.push_back(e);
        }

        // Deceleration phase
        // After s_max the controller stays active and ramps v_theta_max to 0
        // over the margin segment (100-110 m), so the solver brakes optimally.
        // Below DECEL_VEL_THR the vehicle is nearly stopped and control returns to the flight controller.

        if (!decelerating_ && s_nearest_prev_ >= s_max_ - 0.1) {
            decelerating_ = true;
            vtheta_nominal_ = weights_.vtheta_max;
            RCLCPP_INFO(get_logger(),
                "Path completed (s=%.2f m), starting DQ-MPCC deceleration", s_nearest_prev_);
            save_and_plot();
        }

        if (decelerating_) {
            // Linear ramp: v_theta_max = nominal * (s_end - s) / margin
            const double s_end = s_max_ + theta_margin_;
            const double ratio = std::clamp((s_end - theta_current_) / theta_margin_, 0.0, 1.0);
            weights_.vtheta_max = vtheta_nominal_ * ratio;
            ctrl_.set_params_all(weights_);

            // Measured speed is low: land in place under MPCC
            const double vel_norm = odom.vel.norm();
            if (vel_norm < DECEL_VEL_THR && vtheta_current_ < 0.5) {
                decelerating_ = false;
                landing_ = true;
                land_start_ = this->now();
                RCLCPP_INFO(get_logger(),
                    "Deceleration complete (|v|=%.2f m/s), landing in place", vel_norm);
                weights_.vtheta_max = vtheta_nominal_;
            }
        }

        // LANDING in-place (bypass mav_manager)
        // Publish TRPY directly: thrust below hover gives a vertical descent.
        // Motors off when z < 0.15 m or after a 12 s timeout.
        if (landing_) {
            const double z = odom.pos[2];
            const double elapsed = (this->now() - land_start_).seconds();
            quadrotor_msgs::msg::TRPYCommand cmd;
            if (z < 0.15 || elapsed > 12.0) {
                cmd.aux.enable_motors = false;
                cmd.thrust = 0.0;
                RCLCPP_INFO_ONCE(get_logger(),
                    "Touchdown z=%.3f (elapsed=%.1fs) — motors OFF", z, elapsed);
                landing_ = false;
                active_ = false;
            } else {
                // hover thrust × 0.92 → ~5% gravity deficit → ~0.5 m/s descent
                const double m = 1.05, g = 9.81;
                cmd.aux.enable_motors = true;
                cmd.thrust = 0.92 * m * g;
                cmd.angular_velocity.x = 0.0;
                cmd.angular_velocity.y = 0.0;
                cmd.angular_velocity.z = 0.0;
            }
            cmd_pub_->publish(cmd);
            return;
        }

        // Publish the command: thrust comes from the (smooth) state, not from the control Δf
        quadrotor_msgs::msg::TRPYCommand cmd;
        cmd.aux.enable_motors  = true;
        cmd.thrust             = f_current_;
        cmd.angular_velocity.x = u(1);
        cmd.angular_velocity.y = u(2);
        cmd.angular_velocity.z = u(3);
        cmd_pub_->publish(cmd);
        publish_counter_.fetch_add(1);

        if (status != 0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "solver status=%d  θ=%.2f  s=%.2f", status, theta_current_, s_nearest_prev_);
        }
    }

    void send_zero()
    {
        quadrotor_msgs::msg::TRPYCommand cmd;
        cmd.aux.enable_motors = false;
        cmd.thrust = 0.0;
        cmd_pub_->publish(cmd);
    }

    // Save the CSV log

    void save_and_plot()
    {
        if (log_.empty()) return;

        // Base results dir: env override > compile-time repo path > /tmp.
        const char* env_out = std::getenv("QUADROTOR_RESULTS_DIR");
        std::string base = env_out ? std::string(env_out)
#ifdef QUADROTOR_RESULTS_DEFAULT
                                    : std::string(QUADROTOR_RESULTS_DEFAULT);
#else
                                    : std::string("/tmp/quadrotor_mpc_results");
#endif

        // Target subfolder: QUADROTOR_TARGET env var
        //   sim_local  (default) → sim/local/{ctrl}/   local workstation + MuJoCo
        //   sim_jetson           → sim/jetson/{ctrl}/  onboard computer + MuJoCo
        //   hw                   → hw/{ctrl}/          onboard computer + vehicle
        const char* env_tgt = std::getenv("QUADROTOR_TARGET");
        std::string target  = env_tgt ? std::string(env_tgt) : "sim_local";
        std::string sub;
        if      (target == "sim_local")  sub = "sim/local/dq_mpcc_ablation";
        else if (target == "sim_jetson") sub = "sim/jetson/dq_mpcc_ablation";
        else if (target == "hw")         sub = "hw/dq_mpcc_ablation";
        else                              sub = "sim/local/dq_mpcc_ablation";  // fallback

        const std::string results_dir = base + "/" + sub;
        { std::error_code ec; std::filesystem::create_directories(results_dir, ec); }

        // Run-id incremental: scan results_dir for dq_mpcc_b_V<N>.csv, pick max+1
        const std::string prefix = "dq_mpcc_ablation_b_V";
        int next_id = 1;
        try {
            for (const auto& e : std::filesystem::directory_iterator(results_dir)) {
                std::string name = e.path().filename().string();
                if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size() + 4) {
                    std::string num = name.substr(prefix.size(),
                                                  name.size() - prefix.size() - 4);
                    try { int n = std::stoi(num); if (n >= next_id) next_id = n + 1; }
                    catch (...) {}
                }
            }
        } catch (...) {}
        const std::string csv_path = results_dir + "/" + prefix
                                     + std::to_string(next_id) + ".csv";

        std::ofstream csv(csv_path);
        if (!csv.is_open()) {
            RCLCPP_ERROR(get_logger(), "Could not create: %s", csv_path.c_str());
            return;
        }
        csv << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
            << "T,wx_cmd,wy_cmd,wz_cmd,solve_ms,loop_ms,s,"
            << "px_ref,py_ref,pz_ref,qw_ref,qx_ref,qy_ref,qz_ref,"
            << "theta,vtheta,a_theta,"
            << "s_nearest,px_ref_sn,py_ref_sn,pz_ref_sn,"
            << "qw_ref_sn,qx_ref_sn,qy_ref_sn,qz_ref_sn,status\n";
        csv << std::setprecision(8);
        for (const auto& e : log_) {
            csv << e.t
                << "," << e.px  << "," << e.py  << "," << e.pz
                << "," << e.vx  << "," << e.vy  << "," << e.vz
                << "," << e.qw  << "," << e.qx  << "," << e.qy  << "," << e.qz
                << "," << e.wx  << "," << e.wy  << "," << e.wz
                << "," << e.T   << "," << e.wx_cmd << "," << e.wy_cmd << "," << e.wz_cmd
                << "," << e.solve_ms << "," << e.loop_ms << "," << e.s
                << "," << e.px_ref  << "," << e.py_ref  << "," << e.pz_ref
                << "," << e.qw_ref  << "," << e.qx_ref  << "," << e.qy_ref  << "," << e.qz_ref
                << "," << e.theta   << "," << e.vtheta  << "," << e.a_theta
                << "," << e.s_nearest
                << "," << e.px_ref_sn << "," << e.py_ref_sn << "," << e.pz_ref_sn
                << "," << e.qw_ref_sn << "," << e.qx_ref_sn << "," << e.qy_ref_sn << "," << e.qz_ref_sn
                << "," << e.status << "\n";
        }
        csv.close();
        RCLCPP_INFO(get_logger(), "CSV saved: %s  (%zu steps)", csv_path.c_str(), log_.size());

        // Figures are generated offline from the CSV logs.
    }

    // Miembros
    ArcLengthPath arc_path_;
    PathWaypoints wp_;
    double s_max_        = 100.0;
    double theta_margin_ = 10.0;
    double mass_g_       = 1.08 * 9.81;

    DqMpccAblationController ctrl_;
    DqMpccWeights    weights_;

    double   theta_current_  = 0.0;
    double   vtheta_current_ = 0.0;
    double   f_current_      = 1.08 * 9.81;  // hover thrust
    double   s_nearest_prev_ = 0.0;
    bool     first_call_     = true;
    OdomSnap init_snap_;

    // Smooth post-path deceleration
    bool   decelerating_    = false;
    double vtheta_nominal_  = 0.0;
    static constexpr double DECEL_VEL_THR = 1.0;  // ‖v‖ threshold for landing [m/s]

    // MPCC-driven landing (bypass mav_manager)
    bool        landing_    = false;
    rclcpp::Time land_start_;

    // Logging (integrated flight stack)
    std::vector<StepLog>                   log_;
    std::chrono::steady_clock::time_point  t_start_;
    std::chrono::steady_clock::time_point  last_loop_time_{};

    static constexpr int N_WARMUP = 20;

    mutable std::mutex  odom_mtx_;
    OdomSnap            odom_;
    std::atomic<bool>   active_{false};
    std::atomic<bool>   motors_on_{false};
    std::atomic<int>    publish_counter_{0};

    rclcpp::Publisher<quadrotor_msgs::msg::TRPYCommand>::SharedPtr     cmd_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr               motors_sub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr                 srv_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr                  pd_cli_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr                  land_cli_;
    rclcpp::Client<mav_manager_srv::srv::Vec4>::SharedPtr              goto_cli_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr                  motors_cli_;
    rclcpp::TimerBase::SharedPtr                                        timer_;
    rclcpp::TimerBase::SharedPtr                                        goto_timer_;
    rclcpp::TimerBase::SharedPtr                                        motors_off_timer_;
    rclcpp::TimerBase::SharedPtr                                        land_timer_;
    rclcpp::CallbackGroup::SharedPtr                                    cb_timer_;
    rclcpp::CallbackGroup::SharedPtr                                    cb_srv_;
};

// main

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DqMpccAblationNode>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
