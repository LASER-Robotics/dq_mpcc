/**
 * MPCC SiL — Rate-control MPCC with MuJoCo feedback via SilProtocol.
 *
 * Key difference from NMPC:
 *   - State is 15-dim: [p, v, q, ω, θ, v_θ]
 *   - θ (arc-length progress) is OPTIMISED by the solver
 *   - The solver maximises v_θ while minimising contouring/lag error
 *   - No external progress tracker needed — θ comes from the OCP solution
 *
 * Protocol: RELOAD → PD HOLD → RUN (MPCC) → RELOAD
 */

#include "quadrotor_mpc/mpcc/mpcc_controller.hpp"
#include "quadrotor_mpc/mujoco/mujoco_interface.hpp"
#include "quadrotor_mpc/mujoco/sil_protocol.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include "quadrotor_mpc/common/params.hpp"
#include "quadrotor_mpc/trajectory/attitude_reference.hpp"
#include "quadrotor_mpc/trajectory/planner_ref.hpp"

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <thread>

// Resolve path: try as-is, else prepend $(ament share dir for quadrotor_mpc).
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
#include <cmath>
#include <random>
#include <chrono>

using namespace quadrotor_mpc;

// Apply weights from JSON trial (prefers params, fallback to weights)
// Single source of truth: params (Optuna native flat keys).
static void apply_weights(const nlohmann::json& pt, MpccParams& mp) {
    if (pt.contains("params")) {
        const auto& p = pt["params"];
        double qec = p["Q_ec"]; double qel = p["Q_el"]; double qq = p["Q_q"];
        double ut  = p["U_T"];  double uw  = p["U_w"];  double qs = p["Q_s"];
        mp.Q_ec  = Vec3(qec, qec, qec);
        mp.Q_el  = Vec3(qel, qel, qel);
        mp.Q_q   = Vec3(qq, qq, qq);
        mp.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        mp.Q_s   = qs;
        if (p.contains("W_df")) mp.W_df = p["W_df"];
    } else if (pt.contains("weights")) {
        // Fallback for JSON files without a "params" block
        const auto& w = pt["weights"];
        double qec = w["Q_ec"][0]; double qel = w["Q_el"][0]; double qq = w["Q_q"][0];
        double ut  = w["U_mat"][0]; double uw = w["U_mat"][1]; double qs = w["Q_s"];
        mp.Q_ec  = Vec3(qec, qec, qec);
        mp.Q_el  = Vec3(qel, qel, qel);
        mp.Q_q   = Vec3(qq, qq, qq);
        mp.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        mp.Q_s   = qs;
        if (w.contains("W_df")) mp.W_df = w["W_df"];
    }
}

// Load tuned weights from JSON
static bool load_weights_json(const std::string& path,
                              MpccParams& mp, QuadParams& qp, double vel) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    apply_weights(j, mp);  // flat JSON files are also accepted
    if (j.contains("vtheta_max")) qp.vtheta_max = j["vtheta_max"];
    else qp.vtheta_max = vel;
    return true;
}

// Racing pareto: best trial = min lap_time_min field.
static bool load_pareto_best(const std::string& pareto_path,
                             MpccParams& mp, QuadParams& qp,
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
    apply_weights(pt, mp);
    qp.vtheta_max = pt["params"]["vtheta_max"];
    trial_id_found = pt.value("trial", -1);
    return true;
}

static bool load_pareto_trial(const std::string& pareto_path, int trial_id,
                              MpccParams& mp, QuadParams& qp) {
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
    apply_weights(*pt, mp);
    qp.vtheta_max = (*pt)["params"]["vtheta_max"];
    return true;
}

// Save SilResult to CSV (extended for MPCC: includes θ, v_θ, a_θ)
static void save_csv(const std::string& path, const SilResult& res) {
    std::ofstream csv(path);
    // x: 13 physical states logged (p,v,q,ω from DroneState)
    // u: 4 commands (T, ωx_cmd, ωy_cmd, ωz_cmd) — a_θ is internal to solver
    // extra: theta, v_theta from solver prediction
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

    // Parameters (overrideable via CLI)
    QuadParams quad;
    MpccParams mpcc_p;

    // CLI modes:
    //   ./mpcc_sil                              → manual weights
    //   ./mpcc_sil --pareto                     → best trial from default pareto JSON
    //   ./mpcc_sil --pareto t<N>                → specific trial id
    //   ./mpcc_sil --pareto <path>              → best trial from custom pareto file
    //   ./mpcc_sil --pareto <path> t<N>         → specific trial from custom pareto file
    //   ./mpcc_sil Q_ec Q_el Q_q U_T U_w Q_s [vtheta_max] [W_df] → explicit (tuner)
    std::string gain_source = "manual";
    double v_target_label = 0;  // for CSV naming
    if (argc >= 2 && std::string(argv[1]) == "--pareto") {
        std::string pareto_path = resolve_path("../tuning/pareto_mpcc.json");
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
                ok = load_pareto_best(pareto_path, mpcc_p, quad, trial_used);
                gain_source = "pareto_best";
            } else {
                std::string sel = argv[selector_arg];
                if (sel.size() > 1 && sel[0] == 't') {
                    int tid = std::atoi(sel.c_str() + 1);
                    ok = load_pareto_trial(pareto_path, tid, mpcc_p, quad);
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
        // Production mode: load tuned weights for this velocity
        double vel = std::atof(argv[1]);
        v_target_label = vel;
        std::string json_path = "../tuning/best_weights_mpcc_sil_v"
                                + std::to_string(static_cast<int>(vel)) + ".json";
        if (load_weights_json(json_path, mpcc_p, quad, vel)) {
            gain_source = json_path;
        } else {
            fprintf(stderr, "ERROR: cannot load %s\n", json_path.c_str());
            return 1;
        }
    } else if (argc >= 7) {
        // Tuner mode: explicit weights
        // ./mpcc_sil Q_ec Q_el Q_q U_T U_w Q_s [vtheta_max] [W_df]
        double qec = std::atof(argv[1]);
        double qel = std::atof(argv[2]);
        double qq  = std::atof(argv[3]);
        double ut  = std::atof(argv[4]);
        double uw  = std::atof(argv[5]);
        double qs  = std::atof(argv[6]);
        mpcc_p.Q_ec  = Vec3(qec, qec, qec);
        mpcc_p.Q_el  = Vec3(qel, qel, qel);
        mpcc_p.Q_q   = Vec3(qq, qq, qq);
        mpcc_p.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        mpcc_p.Q_s   = qs;
        gain_source = "cli";
        if (argc >= 8) quad.vtheta_max = std::atof(argv[7]);
        if (argc >= 9) mpcc_p.W_df = std::atof(argv[8]);
    }

    // MuJoCo interface
    const char* qn_env = std::getenv("QUAD_NAME");
    std::string quad_name = (qn_env && *qn_env) ? std::string(qn_env) : "quadrotor";
    auto muj = std::make_shared<MujocoInterface>("mpcc_sil_controller", quad_name);
    std::thread spin_thread([&]() { rclcpp::spin(muj); });

    // Load trajectory from path planner (Phase 2)
    //   ref_flat_waypoints.json is produced by
    //   path_planing/export_ref_json.py  from ref_flat_waypoints.npz.
    //   The OCP itself was also generated against this same reference.
    RCLCPP_INFO(muj->get_logger(), "Loading planner reference...");
    PlannerRef ref;
    const std::string ref_json = resolve_path("../path_planing/ref_flat_waypoints.json");
    if (!load_planner_ref(ref_json, ref)) {
        RCLCPP_ERROR(muj->get_logger(),
            "Cannot load %s — run: cd ../path_planing && python3 export_ref_json.py",
            ref_json.c_str());
        rclcpp::shutdown();
        spin_thread.join();
        return 1;
    }
    PathWaypoints& wp = ref.wp;
    // Full reference length (entry + 3 laps + exit). No theta_margin subtraction:
    // the exit phase natively decelerates the drone back to hover_pos.
    const double s_max    = wp.s_max;
    const double s_wp_end = wp.s_max;
    if (gain_source == "manual") {
        quad.vtheta_max = ref.v_peak_G0;   // planner default only when no file/CLI/pareto
    }
    // Env override for sweeps without rebuild: MPCC_VTHETA_MAX=10 ./mpcc_sil ...
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

    // Init MPCC solver
    RCLCPP_INFO(muj->get_logger(), "Initialising MPCC solver (N=%d)...",
                MpccController::N);
    MpccController ctrl;
    if (!ctrl.init()) {
        RCLCPP_ERROR(muj->get_logger(), "MPCC init failed!");
        rclcpp::shutdown();
        spin_thread.join();
        return 1;
    }

    // Set MPCC weights
    MpccWeights weights;
    weights.Q_ec  = mpcc_p.Q_ec;
    weights.Q_el  = mpcc_p.Q_el;
    weights.Q_q   = mpcc_p.Q_q;
    weights.U_mat = mpcc_p.U_mat;
    weights.Q_s   = mpcc_p.Q_s;
    weights.W_df  = mpcc_p.W_df;
    weights.vtheta_max = quad.vtheta_max;
    // Env override to let the tube replace Q_ec as the dominant contouring
    // mechanism: MPCC_Q_EC_SCALE=0.01 ./mpcc_sil --pareto 10
    if (const char* e = std::getenv("MPCC_Q_EC_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_ec *= s;
            RCLCPP_INFO(muj->get_logger(),
                "Q_ec scaled ×%.3f → [%.2f, %.2f, %.2f]",
                s, weights.Q_ec(0), weights.Q_ec(1), weights.Q_ec(2));
        }
    }
    ctrl.set_params_all(weights);

    RCLCPP_INFO(muj->get_logger(),
        "Weights: Q_ec=%.0f Q_el=%.0f Q_q=%.1f U=[%.1f,%.1f] Q_s=%.1f",
        weights.Q_ec(0), weights.Q_el(0), weights.Q_q(0),
        weights.U_mat(0), weights.U_mat(1), weights.Q_s);

    // Track θ, v_θ, and f from solver (carried between iterations)
    double theta_current = 0.0;
    double vtheta_current = 0.0;
    double f_current = quad.mass * quad.g;  // start at hover thrust

    // Warm-up: run a few SQP-RTI iterations before starting to build up a good initial guess
    const int N_SQP_WARMUP = 20;

    // Configure SiL protocol
    SilConfig cfg;
    cfg.P0      = ref.hover_pos;   // PD HOLD brings drone here before MPCC RUN
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
    // In production mode, keep the drone visible at hover after PATH_COMPLETE.
    // Only "cli" (tuning) mode wants the hard-reset behaviour.
    cfg.post_run_hold = (gain_source != "cli");
    if (const char* tl = std::getenv("SIL_TRIAL"))
        cfg.trial_label = std::string(tl);
    else
        cfg.trial_label = "MPCC";

    SilProtocol proto(muj, cfg);

    // Track s_nearest for geometric reference (paper metrics + termination)
    double s_nearest_prev = 0.0;

    // Progress tracker: use s_nearest (physical drone position on path)
    // NOT theta (virtual, runs ahead). Termination is based on this value.
    proto.set_progress_tracker(
        [&](const Vec3& /*pos*/, double /*s_prev*/) -> double {
            return s_nearest_prev;  // physical progress from previous step
        });

    // Controller callback: MPCC solve
    bool first_call = true;

    proto.set_controller(
        [&](const DroneState& ds, double t, double s_current) -> ControlOutput {
            // Build 16-state vector: [p, v, q, ω, θ, v_θ, f]
            State16 x;
            x.head<3>()       = ds.pos;
            x.segment<3>(3)   = ds.vel;
            x.segment<4>(6)   = ds.quat;
            x.segment<3>(10)  = ds.omega;
            x(13) = theta_current;
            x(14) = vtheta_current;
            x(15) = f_current;

            // Set initial condition
            ctrl.set_x0(x);

            // Set weights
            ctrl.set_params_all(weights);

            // On first call, seed horizon with linearly increasing θ, then warm-start
            if (first_call) {
                const int Nh = MpccController::N;
                double v0 = 2.0;  // initial guess for progress velocity
                for (int k = 0; k <= Nh; ++k) {
                    State16 xk = x;
                    double tk = k * 0.01;
                    xk(13) = std::clamp(theta_current + v0 * tk, 0.0, s_max + quad.theta_margin);  // θ
                    xk(14) = v0;       // v_θ
                    xk(15) = f_current; // f (hover thrust)
                    ctrl.set_x_init(k, xk);
                    if (k < Nh) {
                        Control5 uk;
                        uk << 0.0, 0.0, 0.0, 0.0, 0.0;  // Δf=0, zero rates, zero a_θ
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
            //   θ_{k+1}   = clip(θ_k + v_θ_k * dt, 0, s_max + margin)
            //   v_θ_{k+1} = clip(v_θ_k + a_θ * dt, 0, vtheta_max)
            //   f_{k+1}   = clip(f_k + Δf * dt, T_min, T_max)
            double delta_f = u(0);
            double a_theta = u(4);
            double dt = 0.01;  // control period
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
    std::string csv_path = "mpcc_sil_results.csv";
    save_csv(csv_path, res);

    // Persistent copy: <output>/sil/mpcc/mpcc_sil_{source}.csv  (no v prefix)
    {
        const char* env_dir = std::getenv("QUADROTOR_RESULTS_DIR");
        std::string base    = env_dir ? std::string(env_dir)
#ifdef QUADROTOR_RESULTS_DEFAULT
                                      : std::string(QUADROTOR_RESULTS_DEFAULT);
#else
                                      : std::string("/tmp/quadrotor_mpc_results");
#endif
        std::string outdir  = base + "/sil/mpcc";
        { std::error_code ec; std::filesystem::create_directories(outdir, ec); }
        std::string src = gain_source;
        for (auto& c : src) { if (c == '/') c = '_'; }
        std::string persistent = outdir + "/mpcc_sil_" + src + ".csv";
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
