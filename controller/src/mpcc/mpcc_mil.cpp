/**
 * MPCC MiL — Model-in-the-Loop.
 *
 * Pure C++ (no ROS, no MuJoCo). The "plant" is the same continuous-time
 * rate-controlled quadrotor model the OCP is built on, integrated locally
 * with RK4 at the control period dt. Lets us validate the controller and
 * the reference pipeline in isolation from the simulator.
 *
 * Reference: loaded from path_planing/ref_flat_waypoints.json, produced
 * by the path_planing pipeline (path_lissajous → path_time_3D_PMM_periodic
 * → build_multilap_reference → build_flat_reference → export_ref_json).
 * Already parameterised in arc-length s [m], with unit quaternions and
 * unit tangents; the MiL consumes it directly — no further conversion.
 *
 * Plant state (13): [p(3), v(3), q(4), ω(3)]
 *   ṗ = v
 *   v̇ = (1/m) R(q) [0,0,T]ᵀ + [0,0,-g]ᵀ
 *   q̇ = ½ q ⊗ [0, ω]
 *   ω̇ = (ω_cmd - ω) / τ_rc
 *
 * Virtual states [θ, v_θ, f] integrated with forward-Euler locally,
 * never read back from the solver prediction.
 */

#include "quadrotor_mpc/mpcc/mpcc_controller.hpp"
#include "quadrotor_mpc/common/quaternion_algebra.hpp"
#include "quadrotor_mpc/common/integrators.hpp"
#include "quadrotor_mpc/common/params.hpp"
#include "quadrotor_mpc/trajectory/attitude_reference.hpp"
#include "quadrotor_mpc/trajectory/planner_ref.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

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
#include <chrono>
#include <algorithm>
#include <cstdio>

using namespace quadrotor_mpc;

// Plant state (13): [p, v, q, ω]
using PlantVec = Eigen::Matrix<double, 13, 1>;
using InputVec = Eigen::Matrix<double, 4, 1>;   // [T, ωx_cmd, ωy_cmd, ωz_cmd]

static PlantVec plant_f(const PlantVec& x, const InputVec& u,
                        double mass, double g, double tau_rc)
{
    Vec3 p   = x.segment<3>(0);
    Vec3 v   = x.segment<3>(3);
    Quat4 q  = x.segment<4>(6);
    Vec3 w   = x.segment<3>(10);

    double T = u(0);
    Vec3 w_cmd = u.segment<3>(1);

    // Normalize quaternion inside f to keep RK4 stages consistent
    q = quat_normalize(q);
    Mat3 R = quat_to_rot(q);

    Vec3 pdot = v;
    Vec3 vdot = (1.0 / mass) * (R * Vec3(0.0, 0.0, T)) + Vec3(0.0, 0.0, -g);
    Quat4 qdot = quat_kinematics(q, w);
    Vec3 wdot = (w_cmd - w) / tau_rc;

    PlantVec dx;
    dx.segment<3>(0)  = pdot;
    dx.segment<3>(3)  = vdot;
    dx.segment<4>(6)  = qdot;
    dx.segment<3>(10) = wdot;
    return dx;
}

// Weight loading (prefers params, fallback weights)
static void apply_weights(const nlohmann::json& pt, MpccParams& mp) {
    if (pt.contains("params")) {
        const auto& p = pt["params"];
        double qec=p["Q_ec"]; double qel=p["Q_el"]; double qq=p["Q_q"];
        double ut=p["U_T"]; double uw=p["U_w"]; double qs=p["Q_s"];
        mp.Q_ec=Vec3(qec,qec,qec); mp.Q_el=Vec3(qel,qel,qel); mp.Q_q=Vec3(qq,qq,qq);
        mp.U_mat=Eigen::Vector4d(ut,uw,uw,uw); mp.Q_s=qs;
        if (p.contains("W_df")) mp.W_df=p["W_df"];
    } else if (pt.contains("weights")) {
        const auto& w = pt["weights"];
        double qec=w["Q_ec"][0]; double qel=w["Q_el"][0]; double qq=w["Q_q"][0];
        double ut=w["U_mat"][0]; double uw=w["U_mat"][1]; double qs=w["Q_s"];
        mp.Q_ec=Vec3(qec,qec,qec); mp.Q_el=Vec3(qel,qel,qel); mp.Q_q=Vec3(qq,qq,qq);
        mp.U_mat=Eigen::Vector4d(ut,uw,uw,uw); mp.Q_s=qs;
        if (w.contains("W_df")) mp.W_df=w["W_df"];
    }
}

static bool load_weights_json(const std::string& path,
                              MpccParams& mp, QuadParams& qp, double vel) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    apply_weights(j, mp);
    qp.vtheta_max = j.contains("vtheta_max") ? (double)j["vtheta_max"] : vel;
    return true;
}

// Load the "best" trial from the Pareto front (racing objective):
// minimises lap_time_min field (lap-time of fastest lap, in seconds).
// Falls back to the first front entry if the field is missing.
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

// Load a specific trial id from either the Pareto front or the history.
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

int main(int argc, char** argv)
{
    QuadParams quad;
    MpccParams mpcc_p;

    std::string gain_source = "manual";
    double v_target_label = 0;

    // --pareto                          → best trial in default file
    // --pareto t<N>                     → specific trial by id (default file)
    // --pareto <path>                   → best trial from custom pareto file
    // --pareto <path> t<N>              → specific trial from custom pareto file
    if (argc >= 2 && std::string(argv[1]) == "--pareto") {
        std::string pareto_path = resolve_path("../tuning/pareto_mpcc.json");
        int selector_arg = 2;
        // If argv[2] looks like a path (contains '/' or ends in .json),
        // treat it as a pareto file override.
        if (argc >= 3) {
            std::string a2 = argv[2];
            if (a2.find('/') != std::string::npos ||
                a2.size() > 5 && a2.substr(a2.size()-5) == ".json") {
                pareto_path = resolve_path(a2);
                selector_arg = 3;
            }
        }
        int trial_used = -1;
        bool ok = false;
        // The output CSV name follows the gain file.
        std::string regime_tag = "pareto_best";
        if (pareto_path.find("/matched/") != std::string::npos) regime_tag = "matched";
        else if (pareto_path.find("/free/") != std::string::npos) regime_tag = "free";

        if (argc <= selector_arg) {
            ok = load_pareto_best(pareto_path, mpcc_p, quad, trial_used);
            gain_source = regime_tag;
        } else {
            std::string sel = argv[selector_arg];
            if (sel.size() > 1 && sel[0] == 't') {
                int tid = std::atoi(sel.c_str() + 1);
                ok = load_pareto_trial(pareto_path, tid, mpcc_p, quad);
                trial_used = tid;
                gain_source = regime_tag + "_t" + std::to_string(tid);
            } else {
                std::fprintf(stderr,
                    "ERROR: --pareto argument must be empty or 't<N>' "
                    "(e.g. --pareto t45). Got: %s\n", sel.c_str());
                return 1;
            }
        }
        if (!ok) {
            std::fprintf(stderr,
                "ERROR: cannot load %s or selector not found\n",
                pareto_path.c_str());
            return 1;
        }
        std::fprintf(stderr, "Pareto mode: trial=%d  vtheta_max=%.2f\n",
                     trial_used, quad.vtheta_max);
    } else if (argc == 2) {
        double vel = std::atof(argv[1]);
        v_target_label = vel;
        std::string path = "../tuning/best_weights_mpcc_sil_v"
                           + std::to_string((int)vel) + ".json";
        if (!load_weights_json(path, mpcc_p, quad, vel)) {
            std::fprintf(stderr, "ERROR: cannot load %s\n", path.c_str());
            return 1;
        }
        gain_source = path;
    } else if (argc >= 7) {
        // Tuner mode: ./mpcc_mil Q_ec Q_el Q_q U_T U_w Q_s [vtheta_max] [W_df]
        double qec = std::atof(argv[1]), qel = std::atof(argv[2]);
        double qq  = std::atof(argv[3]), ut  = std::atof(argv[4]);
        double uw  = std::atof(argv[5]), qs  = std::atof(argv[6]);
        mpcc_p.Q_ec  = Vec3(qec, qec, qec);
        mpcc_p.Q_el  = Vec3(qel, qel, qel);
        mpcc_p.Q_q   = Vec3(qq, qq, qq);
        mpcc_p.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        mpcc_p.Q_s   = qs;
        gain_source = "cli";
        if (argc >= 8) quad.vtheta_max = std::atof(argv[7]);
        if (argc >= 9) mpcc_p.W_df    = std::atof(argv[8]);
    }

    // Load reference from the planner
    //   ref_flat_waypoints.json is produced by
    //   path_planing/export_ref_json.py from ref_flat_waypoints.npz
    PlannerRef ref;
    const std::string ref_json = resolve_path("../path_planing/ref_flat_waypoints.json");
    if (!load_planner_ref(ref_json, ref)) {
        std::fprintf(stderr,
            "ERROR: cannot load %s\n"
            "  Run:  cd ../path_planing && python3 export_ref_json.py\n",
            ref_json.c_str());
        return 1;
    }
    PathWaypoints& wp = ref.wp;
    // The planner reference includes entry + 3 laps + exit. The exit phase
    // decelerates the drone geometrically back to hover, so we terminate when
    // the drone reaches the end of the reference — no theta_margin subtraction.
    const double s_max    = wp.s_max;
    const double s_wp_end = wp.s_max;
    // Override Pareto vtheta_max with the planner-prescribed value.
    // This is the single physically-meaningful choice: v_θ_max must match
    // the PMM reference's peak gate-crossing speed so MPCC can track it.
    if (gain_source == "manual") {
        quad.vtheta_max = ref.v_peak_G0;   // planner default only if no file/CLI/pareto
    }
    // Env override for sweeps: MPCC_VTHETA_MAX=14 ./mpcc_mil --pareto 10
    if (const char* vt_env = std::getenv("MPCC_VTHETA_MAX")) {
        double vt = std::atof(vt_env);
        if (vt > 0.0) {
            quad.vtheta_max = vt;
            std::fprintf(stderr,
                "vtheta_max overridden from env: %.2f m/s\n", vt);
        }
    }
    std::fprintf(stderr,
        "Planner reference: N=%zu, s_max=%.2f (margin=%.1f)\n"
        "  hover_pos=[%.2f, %.2f, %.2f]  s_analysis=[%.2f, %.2f]  T_lap=%.3f s\n"
        "  v_lap_avg=%.2f m/s  v_peak_G0=%.2f m/s  → v_θ_max=%.2f\n",
        wp.pos.size(), s_max, quad.theta_margin,
        ref.hover_pos(0), ref.hover_pos(1), ref.hover_pos(2),
        ref.s_analysis_start, ref.s_analysis_end, ref.T_lap,
        ref.v_lap_avg, ref.v_peak_G0, quad.vtheta_max);

    // Init MPCC solver
    MpccController ctrl;
    if (!ctrl.init()) {
        std::fprintf(stderr, "MPCC init failed\n");
        return 1;
    }
    MpccWeights weights;
    weights.Q_ec = mpcc_p.Q_ec; weights.Q_el = mpcc_p.Q_el;
    weights.Q_q  = mpcc_p.Q_q;  weights.U_mat = mpcc_p.U_mat;
    weights.Q_s  = mpcc_p.Q_s;  weights.W_df = mpcc_p.W_df;
    weights.vtheta_max = quad.vtheta_max;
    // Env scale on Q_ec so the tube constraint becomes dominant: MPCC_Q_EC_SCALE=0.01
    if (const char* e = std::getenv("MPCC_Q_EC_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_ec *= s;
            std::fprintf(stderr, "Q_ec scaled ×%.3f → [%.2f, %.2f, %.2f]\n",
                         s, weights.Q_ec(0), weights.Q_ec(1), weights.Q_ec(2));
        }
    }
    ctrl.set_params_all(weights);

    std::fprintf(stderr,
        "Weights: Q_ec=%.0f Q_el=%.0f Q_q=%.1f U=[%.1f,%.1f] Q_s=%.1f vθmax=%.1f\n",
        weights.Q_ec(0), weights.Q_el(0), weights.Q_q(0),
        weights.U_mat(0), weights.U_mat(1), weights.Q_s, weights.vtheta_max);

    // Initialize plant at planner's hover_pos, identity attitude
    PlantVec x_plant = PlantVec::Zero();
    x_plant.segment<3>(0) = ref.hover_pos;                 // p = (0,0,1.07)
    x_plant.segment<4>(6) = Quat4(1.0, 0.0, 0.0, 0.0);     // q = identity
    // v, ω = 0

    double theta_current  = 0.0;
    double vtheta_current = 0.0;
    double f_current      = quad.mass * quad.g;            // hover thrust

    // Sim parameters
    const double dt       = 0.01;
    const double t_final  = 30.0;
    const double z_min    = 0.15;
    const double z_max    = 5.0;
    const int max_steps   = (int)(t_final / dt) + 1;

    // Log buffers (same columns as SiL CSV)
    std::vector<double> log_t, log_solve_ms, log_loop_ms, log_s;
    std::vector<double> log_theta, log_vtheta, log_atheta, log_s_nearest;
    std::vector<Eigen::Matrix<double,13,1>> log_x;
    std::vector<Eigen::Vector4d> log_u;
    std::vector<Vec3> log_p_ref, log_p_ref_sn;
    std::vector<Quat4> log_q_ref, log_q_ref_sn;
    std::vector<int> log_status;
    log_t.reserve(max_steps);

    // Warm-up: seed horizon + a few SQP iterations
    const int N_SQP_WARMUP = 20;
    {
        State16 x0;
        x0.head<3>()      = x_plant.segment<3>(0);
        x0.segment<3>(3)  = x_plant.segment<3>(3);
        x0.segment<4>(6)  = x_plant.segment<4>(6);
        x0.segment<3>(10) = x_plant.segment<3>(10);
        x0(13) = theta_current; x0(14) = vtheta_current; x0(15) = f_current;
        ctrl.set_x0(x0);
        const int Nh = MpccController::N;
        double v0 = 2.0;
        for (int k = 0; k <= Nh; ++k) {
            State16 xk = x0;
            xk(13) = std::clamp(theta_current + v0 * k * 0.01, 0.0,
                                s_max + quad.theta_margin);
            xk(14) = v0;
            xk(15) = f_current;
            ctrl.set_x_init(k, xk);
            if (k < Nh) {
                Control5 uk; uk.setZero();
                ctrl.set_u_init(k, uk);
            }
        }
        for (int i = 0; i < N_SQP_WARMUP; ++i) ctrl.solve();
    }

    // Main loop
    double s_nearest_prev = 0.0;
    std::string stop_reason = "timeout";
    int solve_fails = 0;

    for (int step = 0; step < max_steps; ++step) {
        double t = step * dt;
        auto t_loop_start = std::chrono::steady_clock::now();

        // Build x16 for solver
        State16 x;
        x.head<3>()       = x_plant.segment<3>(0);
        x.segment<3>(3)   = x_plant.segment<3>(3);
        x.segment<4>(6)   = quat_normalize(x_plant.segment<4>(6));
        x.segment<3>(10)  = x_plant.segment<3>(10);
        x(13) = theta_current;
        x(14) = vtheta_current;
        x(15) = f_current;

        ctrl.set_x0(x);
        int status = ctrl.solve();
        double solve_s = ctrl.get_solve_time();
        if (status != 0) solve_fails++;

        Control5 u = ctrl.get_u0();
        double delta_f = u(0);
        Vec3   w_cmd   = u.segment<3>(1);
        double a_theta = u(4);

        // Safety clamps
        w_cmd = w_cmd.cwiseMax(-quad.W_max).cwiseMin(quad.W_max);

        // Virtual-state Euler integration (same policy as SiL)
        double theta_ub = s_max + quad.theta_margin;
        theta_current  = std::clamp(theta_current + vtheta_current * dt, 0.0, theta_ub);
        vtheta_current = std::clamp(vtheta_current + a_theta * dt, 0.0, quad.vtheta_max);
        f_current      = std::clamp(f_current + delta_f * dt, quad.T_min, quad.T_max);

        // Integrate plant with RK4 using commanded thrust f and ω_cmd
        InputVec u_plant;
        u_plant << f_current, w_cmd(0), w_cmd(1), w_cmd(2);
        auto f_fn = [&](const PlantVec& xs, const InputVec& us) {
            return plant_f(xs, us, quad.mass, quad.g, quad.tau_rc);
        };
        x_plant = rk4_step<13, 4>(f_fn, x_plant, u_plant, dt);
        x_plant.segment<4>(6) = quat_normalize(x_plant.segment<4>(6));

        // References for logging
        Vec3  p_ref = pos_interp(theta_current, wp);
        Quat4 q_ref = quat_interp(theta_current, wp);
        double sn = find_nearest_s(x_plant.segment<3>(0), wp, s_nearest_prev, s_max);
        s_nearest_prev = sn;
        Vec3  p_ref_sn = pos_interp(sn, wp);
        Quat4 q_ref_sn = quat_interp(sn, wp);

        auto t_loop_end = std::chrono::steady_clock::now();
        double loop_ms =
            std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();

        // Log
        log_t.push_back(t);
        Eigen::Matrix<double,13,1> xl = x_plant;
        xl.segment<4>(6) = quat_normalize(xl.segment<4>(6));
        log_x.push_back(xl);
        Eigen::Vector4d ul; ul << f_current, w_cmd(0), w_cmd(1), w_cmd(2);
        log_u.push_back(ul);
        log_solve_ms.push_back(solve_s * 1000.0);
        log_loop_ms.push_back(loop_ms);
        log_s.push_back(sn);
        log_theta.push_back(theta_current);
        log_vtheta.push_back(vtheta_current);
        log_atheta.push_back(a_theta);
        log_s_nearest.push_back(sn);
        log_p_ref.push_back(p_ref);
        log_q_ref.push_back(q_ref);
        log_p_ref_sn.push_back(p_ref_sn);
        log_q_ref_sn.push_back(q_ref_sn);
        log_status.push_back(status);

        // Termination
        double z = x_plant(2);
        if (z < z_min) { stop_reason = "floor"; break; }
        if (z > z_max) { stop_reason = "ceiling"; break; }
        if (sn >= s_max - 0.1) { stop_reason = "completed"; break; }
    }

    // Save CSV (schema identical to mpcc_sil)
    auto save_csv = [&](const std::string& path) {
        std::ofstream csv(path);
        csv << "t,px,py,pz,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
            << "T,wx_cmd,wy_cmd,wz_cmd,solve_ms,loop_ms,s,"
            << "px_ref,py_ref,pz_ref,qw_ref,qx_ref,qy_ref,qz_ref,"
            << "theta,vtheta,a_theta,"
            << "s_nearest,px_ref_sn,py_ref_sn,pz_ref_sn,"
            << "qw_ref_sn,qx_ref_sn,qy_ref_sn,qz_ref_sn,status\n";
        for (size_t i = 0; i < log_t.size(); ++i) {
            csv << log_t[i];
            for (int j = 0; j < 13; ++j) csv << "," << log_x[i](j);
            for (int j = 0; j < 4;  ++j) csv << "," << log_u[i](j);
            csv << "," << log_solve_ms[i] << "," << log_loop_ms[i] << "," << log_s[i];
            for (int j = 0; j < 3; ++j) csv << "," << log_p_ref[i](j);
            for (int j = 0; j < 4; ++j) csv << "," << log_q_ref[i](j);
            csv << "," << log_theta[i] << "," << log_vtheta[i] << "," << log_atheta[i];
            csv << "," << log_s_nearest[i];
            for (int j = 0; j < 3; ++j) csv << "," << log_p_ref_sn[i](j);
            for (int j = 0; j < 4; ++j) csv << "," << log_q_ref_sn[i](j);
            csv << "," << log_status[i] << "\n";
        }
    };

    save_csv("mpcc_mil_results.csv");
    {
        // Resolve writable output dir: env QUADROTOR_RESULTS_DIR > $HOME/quadrotor_mpc_results > /tmp.
        // Output directory: QUADROTOR_RESULTS_DIR if set, otherwise the build-time default.
        const char* env_dir = std::getenv("QUADROTOR_RESULTS_DIR");
        std::string base    = env_dir ? std::string(env_dir)
#ifdef QUADROTOR_RESULTS_DEFAULT
                                      : std::string(QUADROTOR_RESULTS_DEFAULT);
#else
                                      : std::string("/tmp/quadrotor_mpc_results");
#endif
        std::string outdir  = base + "/mil/mpcc";
        { std::error_code ec; std::filesystem::create_directories(outdir, ec); }
        std::string src = gain_source;
        for (auto& c : src) if (c == '/') c = '_';
        std::string persistent = outdir + "/mpcc_mil_" + src + ".csv";
        save_csv(persistent);
        std::fprintf(stderr, "CSV saved: mpcc_mil_results.csv + %s (%zu samples)\n",
                     persistent.c_str(), log_t.size());

    }

    // Summary stats — solve + loop timing for HW deploy gate.
    auto pct = [](std::vector<double>& v, double q) {
        if (v.empty()) return 0.0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        size_t i = std::min(s.size()-1, size_t(q * (s.size()-1)));
        return s[i];
    };
    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0; for (double x : v) s += x; return s / v.size();
    };
    auto vmax = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double m = 0; for (double x : v) m = std::max(m, x); return m;
    };
    std::fprintf(stderr,
        "\n────────── TIMING SUMMARY ──────────\n"
        "  solve_ms   mean=%.2f  p95=%.2f  max=%.2f\n"
        "  loop_ms    mean=%.2f  p95=%.2f  max=%.2f\n"
        "  100Hz gate: %s  (p95<9 AND max<12 ms)\n"
        "────────────────────────────────────\n",
        mean(log_solve_ms), pct(log_solve_ms, 0.95), vmax(log_solve_ms),
        mean(log_loop_ms),  pct(log_loop_ms,  0.95), vmax(log_loop_ms),
        (pct(log_solve_ms, 0.95) < 9.0 && vmax(log_solve_ms) < 12.0) ? "PASS ✓" : "FAIL ✗");

    std::fprintf(stderr,
        "Done: stop=%s  steps=%zu  t_final=%.2fs  s_final=%.2fm  solve_fails=%d\n",
        stop_reason.c_str(), log_t.size(),
        log_t.empty() ? 0.0 : log_t.back(),
        log_s_nearest.empty() ? 0.0 : log_s_nearest.back(),
        solve_fails);

    return 0;
}
