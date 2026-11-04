/**
 * DQ-MPCC SiL — SE(3) dual-quaternion MPCC with MuJoCo feedback via SilProtocol.
 *
 * Key differences from MPCC SiL:
 *   - State is 16-dim: [dq(8), ω(3), v_b(3), θ, v_θ]
 *   - dq = [qr, qd] where qd = ½ t⊗qr  (unit dual quaternion)
 *   - v_b is body-frame velocity (R⁻¹ · v_inertial)
 *   - SE(3) cost with J_l⁻¹ rotation-translation coupling
 *
 * Protocol: RELOAD → PD HOLD → RUN (DQ-MPCC) → RELOAD
 */

#include "quadrotor_mpc/dq_mpcc/dq_mpcc_controller.hpp"
#include "quadrotor_mpc/mujoco/mujoco_interface.hpp"
#include "quadrotor_mpc/mujoco/sil_protocol.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include "quadrotor_mpc/common/params.hpp"
#include "quadrotor_mpc/trajectory/lissajous.hpp"
#include "quadrotor_mpc/trajectory/arc_length.hpp"
#include "quadrotor_mpc/trajectory/attitude_reference.hpp"
#include "quadrotor_mpc/trajectory/planner_ref.hpp"

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <algorithm>

// Resolve a path: try as-is (absolute or cwd-relative), else fallback to
// $(ament share dir)/<stripped>. Lets SiL bin run from build/ OR ros2 run/launch.
static std::string resolve_path(const std::string& p) {
    if (std::filesystem::exists(p)) return p;
    try {
        auto share = ament_index_cpp::get_package_share_directory("quadrotor_mpc");
        std::string rel = p;
        while (rel.size() > 3 && rel.substr(0, 3) == "../") rel = rel.substr(3);
        std::string cand = share + "/" + rel;
        if (std::filesystem::exists(cand)) return cand;
    } catch (...) {}
    return p;
}
#include <fstream>
#include <thread>
#include <cmath>
#include <random>
#include <chrono>

using namespace quadrotor_mpc;

// Apply weights from JSON trial (prefers params, fallback weights)
static void apply_weights(const nlohmann::json& pt, DqMpccParams& dp) {
    if (pt.contains("params")) {
        const auto& p = pt["params"];
        double qphi=p["Q_phi"]; double qec=p["Q_ec"]; double qel=p["Q_el"];
        double ut=p["U_T"]; double uw=p["U_w"]; double qs=p["Q_s"];
        dp.Q_phi=Vec3(qphi,qphi,qphi); dp.Q_ec=Vec3(qec,qec,qec); dp.Q_el=Vec3(qel,qel,qel);
        dp.U_mat=Eigen::Vector4d(ut,uw,uw,uw); dp.Q_s=qs;
        if (p.contains("W_df")) dp.W_df=p["W_df"];
    } else if (pt.contains("weights")) {
        const auto& w = pt["weights"];
        double qphi=w["Q_phi"][0]; double qec=w["Q_ec"][0]; double qel=w["Q_el"][0];
        double ut=w["U_mat"][0]; double uw=w["U_mat"][1]; double qs=w["Q_s"];
        dp.Q_phi=Vec3(qphi,qphi,qphi); dp.Q_ec=Vec3(qec,qec,qec); dp.Q_el=Vec3(qel,qel,qel);
        dp.U_mat=Eigen::Vector4d(ut,uw,uw,uw); dp.Q_s=qs;
        if (w.contains("W_df")) dp.W_df=w["W_df"];
    }
}

// Load tuned weights from JSON
static bool load_weights_json(const std::string& path,
                              DqMpccParams& dp, QuadParams& qp, double vel) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    apply_weights(j, dp);
    if (j.contains("vtheta_max")) qp.vtheta_max = j["vtheta_max"];
    else qp.vtheta_max = vel;
    return true;
}

// Racing pareto: best trial = min lap_time_min field.
static bool load_pareto_best(const std::string& pareto_path,
                             DqMpccParams& dp, QuadParams& qp,
                             int& trial_id_found) {
    std::ifstream f(pareto_path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    auto& front = j["front"];
    if (front.empty()) return false;
    int best_idx = 0; double best_obj = 1e9;
    for (int i = 0; i < (int)front.size(); ++i) {
        double o = front[i].value("lap_time_min", 1e9);
        if (o < best_obj) { best_obj = o; best_idx = i; }
    }
    auto& pt = front[best_idx];
    apply_weights(pt, dp);
    qp.vtheta_max = pt["params"]["vtheta_max"];
    trial_id_found = pt.value("trial", -1);
    return true;
}

static bool load_pareto_trial(const std::string& pareto_path, int trial_id,
                              DqMpccParams& dp, QuadParams& qp) {
    std::ifstream f(pareto_path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    auto pick = [&](const nlohmann::json& arr) -> const nlohmann::json* {
        for (auto& pt : arr)
            if (pt.value("trial", -1) == trial_id) return &pt;
        return nullptr;
    };
    const nlohmann::json* pt = pick(j["front"]);
    if (!pt && j.contains("history")) pt = pick(j["history"]);
    if (!pt) return false;
    apply_weights(*pt, dp);
    qp.vtheta_max = (*pt)["params"]["vtheta_max"];
    return true;
}

// DQ conversion helpers (inline, no external DQ library needed)

/// Build dual quaternion from rotation quaternion + position.
/// dq = [qr(4), qd(4)] where qd = ½ [0,p] ⊗ qr
static State17 drone_state_to_dq17(const DroneState& ds, double theta, double vtheta, double f)
{
    const auto& q = ds.quat;  // [qw, qx, qy, qz]
    const auto& p = ds.pos;   // [px, py, pz]

    // qr = q (rotation quaternion)
    double qr_w = q(0), qr_x = q(1), qr_y = q(2), qr_z = q(3);

    // t_pure = [0, px, py, pz]
    // qd = ½ t_pure ⊗ qr  (Hamilton product)
    double tw = 0.0, tx = p(0), ty = p(1), tz = p(2);
    double qd_w = 0.5 * (tw*qr_w - tx*qr_x - ty*qr_y - tz*qr_z);
    double qd_x = 0.5 * (tw*qr_x + tx*qr_w + ty*qr_z - tz*qr_y);
    double qd_y = 0.5 * (tw*qr_y - tx*qr_z + ty*qr_w + tz*qr_x);
    double qd_z = 0.5 * (tw*qr_z + tx*qr_y - ty*qr_x + tz*qr_w);

    // Body-frame velocity: v_b = R⁻¹ · v_inertial
    Eigen::Matrix3d R = quat_to_rot(q);
    Eigen::Vector3d v_body = R.transpose() * ds.vel;

    State17 x;
    x(0)  = qr_w;  x(1)  = qr_x;  x(2)  = qr_y;  x(3)  = qr_z;
    x(4)  = qd_w;  x(5)  = qd_x;  x(6)  = qd_y;  x(7)  = qd_z;
    x(8)  = ds.omega(0);  x(9)  = ds.omega(1);  x(10) = ds.omega(2);
    x(11) = v_body(0);    x(12) = v_body(1);     x(13) = v_body(2);
    x(14) = theta;
    x(15) = vtheta;
    x(16) = f;
    return x;
}

// Save SilResult to CSV
static void save_csv(const std::string& path, const SilResult& res) {
    std::ofstream csv(path);
    csv << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
        << "T,wx_cmd,wy_cmd,wz_cmd,solve_ms,loop_ms,s,"
        << "ax,ay,az,"
        << "px_ref,py_ref,pz_ref,qw_ref,qx_ref,qy_ref,qz_ref,"
        << "theta,vtheta,a_theta,"
        << "s_nearest,px_ref_sn,py_ref_sn,pz_ref_sn,"
        << "qw_ref_sn,qx_ref_sn,qy_ref_sn,qz_ref_sn,status\n";

    for (int i = 0; i < res.n_steps; ++i) {
        csv << res.t(i);
        for (int j = 0; j < 13; ++j) csv << "," << res.x(j, i);
        for (int j = 0; j < 4; ++j)  csv << "," << res.u(j, i);
        csv << "," << res.solve_ms(i) << "," << res.loop_ms(i) << "," << res.s(i);
        for (int j = 0; j < 3; ++j) csv << "," << res.a_imu(j, i);
        for (int j = 0; j < 3; ++j) csv << "," << res.p_ref(j, i);
        for (int j = 0; j < 4; ++j) csv << "," << res.q_ref(j, i);
        csv << "," << res.theta(i) << "," << res.vtheta(i) << "," << res.a_theta(i);
        csv << "," << res.s_nearest(i);
        for (int j = 0; j < 3; ++j) csv << "," << res.p_ref_nearest(j, i);
        for (int j = 0; j < 4; ++j) csv << "," << res.q_ref_nearest(j, i);
        csv << "," << res.status(i) << "\n";
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // Parameters
    QuadParams quad;
    DqMpccParams dq_p;

    // CLI modes:
    //   ./dq_mpcc_sil                                → manual weights, vtheta_max=14
    //   ./dq_mpcc_sil 8                              → load tuning/best_weights_dq_mpcc_sil_v8.json
    //   ./dq_mpcc_sil --pareto 10                    → Pareto-optimal gains for v_peak≈10 m/s
    //   ./dq_mpcc_sil Q_phi Q_ec Q_el U_T U_w Q_s [vtheta_max]  → explicit (tuner)
    std::string gain_source = "manual";
    double v_target_label = 0;
    if (argc >= 2 && std::string(argv[1]) == "--pareto") {
        // Racing pareto loader:
        //   --pareto                   → best trial in default file
        //   --pareto t<N>              → specific trial id
        //   --pareto <path>            → best from custom file
        //   --pareto <path> t<N>       → specific trial from custom file
        std::string pareto_path = resolve_path("../tuning/pareto_dq_mpcc.json");
        int selector_arg = 2;
        if (argc >= 3) {
            std::string a2 = argv[2];
            bool is_path = (a2.find('/') != std::string::npos) ||
                           (a2.size() > 5 && a2.substr(a2.size()-5) == ".json");
            if (is_path) { pareto_path = resolve_path(a2); selector_arg = 3; }
        }
        {
            int trial_used = -1;
            bool ok = false;
            if (argc <= selector_arg) {
                ok = load_pareto_best(pareto_path, dq_p, quad, trial_used);
                gain_source = "pareto_best";
            } else {
                std::string sel = argv[selector_arg];
                if (sel.size() > 1 && sel[0] == 't') {
                    int tid = std::atoi(sel.c_str() + 1);
                    ok = load_pareto_trial(pareto_path, tid, dq_p, quad);
                    trial_used = tid;
                    gain_source = "pareto_t" + std::to_string(tid);
                } else {
                    fprintf(stderr,
                        "ERROR: --pareto selector must be 't<N>'. Got: %s\n",
                        sel.c_str());
                    return 1;
                }
            }
            if (!ok) {
                fprintf(stderr, "ERROR: cannot load %s or selector not found\n",
                        pareto_path.c_str());
                return 1;
            }
            fprintf(stderr, "Pareto mode: trial=%d  vtheta_max=%.2f\n",
                    trial_used, quad.vtheta_max);
        }
    } else if (argc == 2) {
        double vel = std::atof(argv[1]);
        v_target_label = vel;
        std::string json_path = "../tuning/best_weights_dq_mpcc_sil_v"
                                + std::to_string(static_cast<int>(vel)) + ".json";
        if (load_weights_json(json_path, dq_p, quad, vel)) {
            gain_source = json_path;
        } else {
            fprintf(stderr, "ERROR: cannot load %s\n", json_path.c_str());
            return 1;
        }
    } else if (argc >= 7) {
        double qphi = std::atof(argv[1]);
        double qec  = std::atof(argv[2]);
        double qel  = std::atof(argv[3]);
        double ut   = std::atof(argv[4]);
        double uw   = std::atof(argv[5]);
        double qs   = std::atof(argv[6]);
        dq_p.Q_phi = Vec3(qphi, qphi, qphi);
        dq_p.Q_ec  = Vec3(qec, qec, qec);
        dq_p.Q_el  = Vec3(qel, qel, qel);
        dq_p.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        dq_p.Q_s   = qs;
        gain_source = "cli";
        if (argc >= 8) quad.vtheta_max = std::atof(argv[7]);
        if (argc >= 9) dq_p.W_df = std::atof(argv[8]);
    }

    // MuJoCo interface
    const char* qn_env = std::getenv("QUAD_NAME");
    std::string quad_name = (qn_env && *qn_env) ? std::string(qn_env) : "quadrotor";
    auto muj = std::make_shared<MujocoInterface>("dq_mpcc_sil_controller", quad_name);
    std::thread spin_thread([&]() { rclcpp::spin(muj); });

    // Load trajectory from the path planner
    RCLCPP_INFO(muj->get_logger(), "Loading planner reference...");
    PlannerRef ref;
    const std::string ref_json = resolve_path("../path_planing/ref_flat_waypoints.json");
    if (!load_planner_ref(ref_json, ref)) {
        RCLCPP_ERROR(muj->get_logger(),
            "Cannot load %s — run: cd ../path_planing && python3 export_ref_json.py",
            ref_json.c_str());
        rclcpp::shutdown(); spin_thread.join(); return 1;
    }
    PathWaypoints& wp = ref.wp;
    const double s_max    = wp.s_max;
    const double s_wp_end = wp.s_max;
    if (gain_source == "manual") {
        quad.vtheta_max = ref.v_peak_G0;   // planner default only when no file/CLI/pareto
    }
    if (const char* vt_env = std::getenv("MPCC_VTHETA_MAX")) {
        double vt = std::atof(vt_env);
        if (vt > 0.0) {
            quad.vtheta_max = vt;
            RCLCPP_INFO(muj->get_logger(),
                "vtheta_max overridden from env: %.2f m/s", vt);
        }
    }
    RCLCPP_INFO(muj->get_logger(),
        "Planner ref: N=%zu, s_max=%.2f m, hover=[%.2f,%.2f,%.2f], "
        "T_lap=%.3fs, v_peak_G0=%.2f → v_θ_max=%.2f",
        wp.pos.size(), s_max,
        ref.hover_pos(0), ref.hover_pos(1), ref.hover_pos(2),
        ref.T_lap, ref.v_peak_G0, quad.vtheta_max);

    // Init DQ-MPCC solver
    RCLCPP_INFO(muj->get_logger(), "Initialising DQ-MPCC solver (N=%d)...",
                DqMpccController::N);
    DqMpccController ctrl;
    if (!ctrl.init()) {
        RCLCPP_ERROR(muj->get_logger(), "DQ-MPCC init failed!");
        rclcpp::shutdown();
        spin_thread.join();
        return 1;
    }

    // Set DQ-MPCC weights
    DqMpccWeights weights;
    weights.Q_phi = dq_p.Q_phi;
    weights.Q_ec  = dq_p.Q_ec;
    weights.Q_el  = dq_p.Q_el;
    weights.U_mat = dq_p.U_mat;
    weights.Q_s   = dq_p.Q_s;
    weights.W_df  = dq_p.W_df;
    weights.vtheta_max = quad.vtheta_max;
    // Env overrides for racing tuning (MPCC_Q_EC_SCALE / MPCC_Q_PHI_SCALE)
    if (const char* e = std::getenv("MPCC_Q_EC_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_ec *= s;
            RCLCPP_INFO(muj->get_logger(),
                "Q_ec scaled ×%.3f → [%.2f, %.2f, %.2f]",
                s, weights.Q_ec(0), weights.Q_ec(1), weights.Q_ec(2));
        }
    }
    if (const char* e = std::getenv("MPCC_Q_PHI_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_phi *= s;
            RCLCPP_INFO(muj->get_logger(),
                "Q_phi scaled ×%.3f → [%.2f, %.2f, %.2f]",
                s, weights.Q_phi(0), weights.Q_phi(1), weights.Q_phi(2));
        }
    }
    ctrl.set_params_all(weights);

    RCLCPP_INFO(muj->get_logger(),
        "Weights: Q_phi=%.1f Q_ec=%.0f Q_el=%.0f U=[%.1f,%.1f] Q_s=%.1f",
        weights.Q_phi(0), weights.Q_ec(0), weights.Q_el(0),
        weights.U_mat(0), weights.U_mat(1), weights.Q_s);

    // Track θ, v_θ, and f locally (NOT from solver predictions)
    double theta_current = 0.0;
    double vtheta_current = 0.0;
    double f_current = quad.mass * quad.g;  // start at hover thrust

    const int N_SQP_WARMUP = 20;

    // Configure SiL protocol
    SilConfig cfg;
    cfg.P0      = ref.hover_pos;
    // Override P0 via environment variable: SIL_P0="x,y,z"
    if (const char* p0_env = std::getenv("SIL_P0")) {
        double px, py, pz;
        if (sscanf(p0_env, "%lf,%lf,%lf", &px, &py, &pz) == 3) {
            cfg.P0 = Vec3(px, py, pz);
            RCLCPP_INFO(muj->get_logger(), "P0 override: [%.2f, %.2f, %.2f]", px, py, pz);
        }
    }
    // Random P0 within Lissajous bounding box: SIL_P0_RANDOM="1"
    // x ∈ [-2.5, 7.5], y ∈ [-1.25, 1.25], z ∈ [0.75, 2.25]
    if (const char* r_env = std::getenv("SIL_P0_RANDOM")) {
        if (std::atoi(r_env) > 0) {
            unsigned seed = 0;
            if (const char* s_env = std::getenv("SIL_P0_SEED"))
                seed = static_cast<unsigned>(std::atoi(s_env));
            else
                seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dx(0.0, 5.0);
            std::uniform_real_distribution<double> dy(-0.625, 0.625);
            std::uniform_real_distribution<double> dz(1.125, 1.875);
            cfg.P0 = Vec3(dx(rng), dy(rng), dz(rng));
            RCLCPP_INFO(muj->get_logger(), "P0 random (seed=%u): [%.2f, %.2f, %.2f]",
                        seed, cfg.P0(0), cfg.P0(1), cfg.P0(2));
        }
    }
    cfg.mass    = quad.mass;
    cfg.gravity = quad.g;
    cfg.s_max   = s_max;
    cfg.t_final = 85.0;
    cfg.abort_on_crash = true;
    cfg.post_run_hold  = (gain_source != "cli");
    if (const char* tl = std::getenv("SIL_TRIAL"))
        cfg.trial_label = std::string(tl);
    else
        cfg.trial_label = "DQ-MPCC";

    SilProtocol proto(muj, cfg);

    // Track s_nearest for geometric reference (paper metrics + termination)
    double s_nearest_prev = 0.0;

    // Progress tracker: use s_nearest (physical drone position on path)
    // NOT theta (virtual, runs ahead). Termination is based on this value.
    proto.set_progress_tracker(
        [&](const Vec3& /*pos*/, double /*s_prev*/) -> double {
            return s_nearest_prev;  // physical progress from previous step
        });

    // Controller callback: DQ-MPCC solve
    bool first_call = true;

    proto.set_controller(
        [&](const DroneState& ds, double t, double s_current) -> ControlOutput {
            // Break the exact identity-quaternion singularity at activation
            // ARM-strict eps in J_l^{-1}/ln_dual blows up when |phi|→0 exactly.
            // Add 1e-3 rad tilt on first call only, normalised quat.
            DroneState ds_safe = ds;
            if (first_call) {
                Eigen::Vector4d qn(ds.quat(0), ds.quat(1) + 5e-4,
                                   ds.quat(2) - 5e-4, ds.quat(3));
                qn /= qn.norm();
                ds_safe.quat = qn;
                if (vtheta_current < 0.5) vtheta_current = 0.5;  // seed nonzero v_theta
            }
            // Convert DroneState to 17-state DQ vector
            State17 x = drone_state_to_dq17(ds_safe, theta_current, vtheta_current, f_current);

            // Set initial condition
            ctrl.set_x0(x);

            // Set weights
            ctrl.set_params_all(weights);

            // First call: seed horizon ALONG REFERENCE PATH (not all=current-pose).
            // Puts initial guess close to optimum → solver linearises around feasible
            // trajectory, avoids first-solve garbage warm-starting subsequent iters.
            if (first_call) {
                const int Nh = DqMpccController::N;
                double v0 = 2.0;
                for (int k = 0; k <= Nh; ++k) {
                    State17 xk = x;
                    double tk = k * 0.01;
                    xk(14) = std::clamp(theta_current + v0 * tk, 0.0, s_max + quad.theta_margin);
                    xk(15) = v0;
                    xk(16) = f_current;
                    ctrl.set_x_init(k, xk);
                    if (k < Nh) {
                        Control5 uk;
                        uk << 0.0, 0.0, 0.0, 0.0, 0.0;
                        ctrl.set_u_init(k, uk);
                    }
                }
                for (int i = 0; i < N_SQP_WARMUP; ++i) {
                    ctrl.solve();
                }
                first_call = false;
            }

            // Solve
            ControlOutput out;
            out.solver_status = ctrl.solve();
            out.solve_time_s  = ctrl.get_solve_time();

            // Get optimal control: [Δf, ωx_cmd, ωy_cmd, ωz_cmd, a_θ]
            Control5 u = ctrl.get_u0();
            out.cmd.thrust    = f_current;         // thrust from state (smooth)
            out.cmd.omega_cmd = u.segment<3>(1);

            // Integrate virtual states locally:
            // θ allowed beyond s_max (B-spline clamps refs). Termination is s_nearest.
            double delta_f = u(0);
            double a_theta = u(4);
            double dt = 0.01;
            double theta_ub = s_max + quad.theta_margin;
            theta_current  = std::clamp(theta_current + vtheta_current * dt, 0.0, theta_ub);
            vtheta_current = std::clamp(vtheta_current + a_theta * dt,
                                        0.0, quad.vtheta_max);
            f_current      = std::clamp(f_current + delta_f * dt,
                                        quad.T_min, quad.T_max);

            // Virtual states for logging
            out.theta   = theta_current;
            out.vtheta  = vtheta_current;
            out.a_theta = a_theta;

            // Reference at current θ (for logging)
            out.p_ref = pos_interp(theta_current, wp);
            out.q_ref = quat_interp(theta_current, wp);

            // Geometrically nearest reference (for paper metrics)
            double sn = find_nearest_s(ds.pos, wp, s_nearest_prev, s_max);
            s_nearest_prev      = sn;
            out.s_nearest       = sn;
            out.p_ref_nearest   = pos_interp(sn, wp);
            out.q_ref_nearest   = quat_interp(sn, wp);

            return out;
        });

    // Execute protocol
    SilResult res = proto.execute();

    // Save CSV
    std::string csv_path = "dq_mpcc_sil_results.csv";
    save_csv(csv_path, res);

    // Persistent copy: <output>/sil/dq_mpcc/dq_mpcc_sil_{source}.csv  (no v prefix)
    {
        const char* env_dir = std::getenv("QUADROTOR_RESULTS_DIR");
        std::string base    = env_dir ? std::string(env_dir)
#ifdef QUADROTOR_RESULTS_DEFAULT
                                      : std::string(QUADROTOR_RESULTS_DEFAULT);
#else
                                      : std::string("/tmp/quadrotor_mpc_results");
#endif
        std::string outdir  = base + "/sil/dq_mpcc";
        { std::error_code ec; std::filesystem::create_directories(outdir, ec); }
        std::string src = gain_source;
        for (auto& c : src) { if (c == '/') c = '_'; }
        std::string persistent = outdir + "/dq_mpcc_sil_" + src + ".csv";
        save_csv(persistent, res);
        RCLCPP_INFO(muj->get_logger(), "CSV saved: %s + %s (%d samples)",
                    csv_path.c_str(), persistent.c_str(), res.n_steps);

    }


    // Timing summary (HW deploy gate)
    if (res.n_steps > 0) {
        std::vector<double> sm(res.solve_ms.data(), res.solve_ms.data() + res.n_steps);
        std::vector<double> lm(res.loop_ms.data(),  res.loop_ms.data()  + res.n_steps);
        std::vector<double> sm_s = sm; std::sort(sm_s.begin(), sm_s.end());
        std::vector<double> lm_s = lm; std::sort(lm_s.begin(), lm_s.end());
        auto p95=[&](const std::vector<double>& s){return s[std::min(s.size()-1, size_t(0.95*(s.size()-1)))];};
        auto mean=[](const std::vector<double>& v){double s=0;for(double x:v)s+=x;return s/v.size();};
        double sm_max=*std::max_element(sm.begin(),sm.end());
        double lm_max=*std::max_element(lm.begin(),lm.end());
        std::fprintf(stderr,
            "\n────────── TIMING SUMMARY ──────────\n"
            "  solve_ms   mean=%.2f  p95=%.2f  max=%.2f\n"
            "  loop_ms    mean=%.2f  p95=%.2f  max=%.2f\n"
            "  100Hz gate: %s  (p95<9 AND max<12 ms)\n"
            "────────────────────────────────────\n",
            mean(sm), p95(sm_s), sm_max,
            mean(lm), p95(lm_s), lm_max,
            (p95(sm_s) < 9.0 && sm_max < 12.0) ? "PASS ✓" : "FAIL ✗");
    }

    rclcpp::shutdown();
    spin_thread.join();
    return 0;
}
