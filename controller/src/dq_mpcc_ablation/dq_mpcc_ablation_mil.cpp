/**
 * DQ-MPCC MiL — Model-in-the-Loop for the SE(3) dual-quaternion MPCC.
 *
 * Pure C++ (no ROS, no MuJoCo). The "plant" is the same continuous-time
 * rate-controlled quadrotor model the OCP is built on (13 physical states),
 * integrated locally with RK4 at the control period dt.
 *
 * Reference: loaded from path_planing/ref_flat_waypoints.json (centered PMM).
 *
 * Plant state (13): [p(3), v(3), q(4), ω(3)]
 *   ṗ = v
 *   v̇ = (1/m) R(q) [0,0,T]ᵀ + [0,0,-g]ᵀ
 *   q̇ = ½ q ⊗ [0, ω]
 *   ω̇ = (ω_cmd - ω) / τ_rc
 *
 * DQ state (17) is built per-step from the plant (p, v, q, ω) by the
 * drone_state_to_dq17 helper (same as dq_mpcc_sil.cpp):
 *     qr = q
 *     qd = ½ [0,p] ⊗ qr
 *     v_b = R(q)⁻¹ · v
 *   plus theta, v_theta, f (integrated with forward-Euler locally).
 *
 * Virtual states [θ, v_θ, f] never read back from the solver prediction.
 */

#include "quadrotor_mpc/dq_mpcc_ablation/dq_mpcc_ablation_controller.hpp"
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
    Vec3  p = x.segment<3>(0);
    Vec3  v = x.segment<3>(3);
    Quat4 q = x.segment<4>(6);
    Vec3  w = x.segment<3>(10);

    double T = u(0);
    Vec3 w_cmd = u.segment<3>(1);

    q = quat_normalize(q);
    Mat3 R = quat_to_rot(q);

    Vec3  pdot = v;
    Vec3  vdot = (1.0 / mass) * (R * Vec3(0.0, 0.0, T)) + Vec3(0.0, 0.0, -g);
    Quat4 qdot = quat_kinematics(q, w);
    Vec3  wdot = (w_cmd - w) / tau_rc;

    PlantVec dx;
    dx.segment<3>(0)  = pdot;
    dx.segment<3>(3)  = vdot;
    dx.segment<4>(6)  = qdot;
    dx.segment<3>(10) = wdot;
    return dx;
}

// DQ state construction (17D) from plant
static State17 plant_to_dq17(const PlantVec& x, double theta, double vtheta, double f)
{
    Vec3  p = x.segment<3>(0);
    Vec3  v = x.segment<3>(3);
    Quat4 q = quat_normalize(x.segment<4>(6));
    Vec3  w = x.segment<3>(10);

    double qr_w = q(0), qr_x = q(1), qr_y = q(2), qr_z = q(3);
    double tw = 0.0, tx = p(0), ty = p(1), tz = p(2);
    double qd_w = 0.5 * (tw*qr_w - tx*qr_x - ty*qr_y - tz*qr_z);
    double qd_x = 0.5 * (tw*qr_x + tx*qr_w + ty*qr_z - tz*qr_y);
    double qd_y = 0.5 * (tw*qr_y - tx*qr_z + ty*qr_w + tz*qr_x);
    double qd_z = 0.5 * (tw*qr_z + tx*qr_y - ty*qr_x + tz*qr_w);

    Mat3 R = quat_to_rot(q);
    Vec3 v_body = R.transpose() * v;

    State17 xdq;
    xdq(0)  = qr_w;  xdq(1)  = qr_x;  xdq(2)  = qr_y;  xdq(3)  = qr_z;
    xdq(4)  = qd_w;  xdq(5)  = qd_x;  xdq(6)  = qd_y;  xdq(7)  = qd_z;
    xdq(8)  = w(0);  xdq(9)  = w(1);  xdq(10) = w(2);
    xdq(11) = v_body(0);  xdq(12) = v_body(1);  xdq(13) = v_body(2);
    xdq(14) = theta;  xdq(15) = vtheta;  xdq(16) = f;
    // Note: State17 is declared as Matrix<double,16,1> in the header (typo)
    // but acados uses 17 positions — the last entry f is read via a deeper
    // buffer path in the solver. For MiL this works because we pass via
    // set_x0 which forwards the array as-is.
    return xdq;
}

// Weight loading (prefers params, fallback weights)
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

static bool load_weights_json(const std::string& path,
                              DqMpccParams& dp, QuadParams& qp, double vel) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    apply_weights(j, dp);
    qp.vtheta_max = j.contains("vtheta_max") ? (double)j["vtheta_max"] : vel;
    return true;
}

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

int main(int argc, char** argv)
{
    QuadParams   quad;
    DqMpccParams dq_p;

    std::string gain_source = "manual";
    double v_target_label = 0;

    if (argc >= 2 && std::string(argv[1]) == "--pareto") {
        std::string pareto_path = resolve_path("../tuning/sil/matched/pareto_dq_mpcc_ablation.json");
        int selector_arg = 2;
        if (argc >= 3) {
            std::string a2 = argv[2];
            if (a2.find('/') != std::string::npos ||
                a2.size() > 5 && a2.substr(a2.size()-5) == ".json") {
                pareto_path = resolve_path(a2);
                selector_arg = 3;
            }
        }
        std::string regime_tag = "pareto_best";
        if (pareto_path.find("/matched/") != std::string::npos) regime_tag = "matched";
        else if (pareto_path.find("/free/") != std::string::npos) regime_tag = "free";

        int trial_used = -1;
        bool ok = false;
        if (argc <= selector_arg) {
            ok = load_pareto_best(pareto_path, dq_p, quad, trial_used);
            gain_source = regime_tag;
        } else {
            std::string sel = argv[selector_arg];
            if (sel.size() > 1 && sel[0] == 't') {
                int tid = std::atoi(sel.c_str() + 1);
                ok = load_pareto_trial(pareto_path, tid, dq_p, quad);
                trial_used = tid;
                gain_source = regime_tag + "_t" + std::to_string(tid);
            } else {
                std::fprintf(stderr,
                    "ERROR: --pareto argument must be empty or 't<N>'. "
                    "Got: %s\n", sel.c_str());
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
        std::string path = "../tuning/best_weights_dq_mpcc_sil_v"
                           + std::to_string((int)vel) + ".json";
        if (!load_weights_json(path, dq_p, quad, vel)) {
            std::fprintf(stderr, "ERROR: cannot load %s\n", path.c_str());
            return 1;
        }
        gain_source = path;
    } else if (argc >= 7) {
        // Tuner mode: ./dq_mpcc_mil Q_phi Q_ec Q_el U_T U_w Q_s [vtheta_max] [W_df]
        double qphi = std::atof(argv[1]), qec = std::atof(argv[2]);
        double qel  = std::atof(argv[3]), ut  = std::atof(argv[4]);
        double uw   = std::atof(argv[5]), qs  = std::atof(argv[6]);
        dq_p.Q_phi = Vec3(qphi, qphi, qphi);
        dq_p.Q_ec  = Vec3(qec, qec, qec);
        dq_p.Q_el  = Vec3(qel, qel, qel);
        dq_p.U_mat = Eigen::Vector4d(ut, uw, uw, uw);
        dq_p.Q_s   = qs;
        gain_source = "cli";
        if (argc >= 8) quad.vtheta_max = std::atof(argv[7]);
        if (argc >= 9) dq_p.W_df       = std::atof(argv[8]);
    }

    // Load reference from planner
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
    const double s_max    = wp.s_max;
    const double s_wp_end = wp.s_max;
    if (gain_source == "manual") {
        quad.vtheta_max = ref.v_peak_G0;   // planner default only if no file/CLI/pareto
    }
    if (const char* vt_env = std::getenv("MPCC_VTHETA_MAX")) {
        double vt = std::atof(vt_env);
        if (vt > 0.0) {
            quad.vtheta_max = vt;
            std::fprintf(stderr,
                "vtheta_max overridden from env: %.2f m/s\n", vt);
        }
    }
    std::fprintf(stderr,
        "Planner ref: N=%zu, s_max=%.2f m, hover=[%.2f,%.2f,%.2f], T_lap=%.3fs, "
        "v_peak_G0=%.2f → v_θ_max=%.2f\n",
        wp.pos.size(), s_max,
        ref.hover_pos(0), ref.hover_pos(1), ref.hover_pos(2),
        ref.T_lap, ref.v_peak_G0, quad.vtheta_max);

    // Init DQ-MPCC solver
    DqMpccAblationController ctrl;
    if (!ctrl.init()) {
        std::fprintf(stderr, "DQ-MPCC init failed\n");
        return 1;
    }
    DqMpccWeights weights;
    weights.Q_phi = dq_p.Q_phi;  weights.Q_ec = dq_p.Q_ec;
    weights.Q_el  = dq_p.Q_el;   weights.U_mat = dq_p.U_mat;
    weights.Q_s   = dq_p.Q_s;    weights.W_df = dq_p.W_df;
    weights.vtheta_max = quad.vtheta_max;
    // Env scale on Q_ec (tube racing mode)
    if (const char* e = std::getenv("MPCC_Q_EC_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_ec *= s;
            std::fprintf(stderr, "Q_ec scaled ×%.3f → [%.2f, %.2f, %.2f]\n",
                         s, weights.Q_ec(0), weights.Q_ec(1), weights.Q_ec(2));
        }
    }
    // Env scale on Q_phi (DQ orientation cost) — diagnostic for attitude tracking
    if (const char* e = std::getenv("MPCC_Q_PHI_SCALE")) {
        double s = std::atof(e);
        if (s > 0.0) {
            weights.Q_phi *= s;
            std::fprintf(stderr, "Q_phi scaled ×%.3f → [%.2f, %.2f, %.2f]\n",
                         s, weights.Q_phi(0), weights.Q_phi(1), weights.Q_phi(2));
        }
    }
    ctrl.set_params_all(weights);

    // Initialize plant at hover
    PlantVec x_plant = PlantVec::Zero();
    x_plant.segment<3>(0) = ref.hover_pos;
    x_plant.segment<4>(6) = Quat4(1.0, 0.0, 0.0, 0.0);

    double theta_current  = 0.0;
    double vtheta_current = 0.0;
    double f_current      = quad.mass * quad.g;

    const double dt       = 0.01;
    const double t_final  = 30.0;
    const double z_min    = 0.15;
    const double z_max    = 5.0;
    const int    max_steps = (int)(t_final / dt) + 1;

    // Log buffers
    std::vector<double> log_t, log_solve_ms, log_loop_ms, log_s;
    std::vector<double> log_theta, log_vtheta, log_atheta, log_s_nearest;
    std::vector<Eigen::Matrix<double,13,1>> log_x;
    std::vector<Eigen::Vector4d> log_u;
    std::vector<Vec3> log_p_ref, log_p_ref_sn;
    std::vector<Quat4> log_q_ref, log_q_ref_sn;
    std::vector<int> log_status;
    log_t.reserve(max_steps);

    // Warm-up
    const int N_SQP_WARMUP = 20;
    {
        State17 x0 = plant_to_dq17(x_plant, theta_current, vtheta_current, f_current);
        ctrl.set_x0(x0);
        const int Nh = DqMpccAblationController::N;
        double v0 = 2.0;
        for (int k = 0; k <= Nh; ++k) {
            State17 xk = x0;
            xk(14) = std::clamp(theta_current + v0 * k * dt, 0.0,
                                s_max + quad.theta_margin);
            xk(15) = v0;
            xk(16) = f_current;
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

        // Build DQ state from physical plant
        State17 x = plant_to_dq17(x_plant, theta_current, vtheta_current, f_current);
        ctrl.set_x0(x);
        int status = ctrl.solve();
        double solve_s = ctrl.get_solve_time();
        if (status != 0) solve_fails++;

        Control5 u = ctrl.get_u0();
        double delta_f = u(0);
        Vec3   w_cmd   = u.segment<3>(1);
        double a_theta = u(4);

        w_cmd = w_cmd.cwiseMax(-quad.W_max).cwiseMin(quad.W_max);

        // Virtual-state Euler integration
        double theta_ub = s_max + quad.theta_margin;
        theta_current  = std::clamp(theta_current + vtheta_current * dt, 0.0, theta_ub);
        vtheta_current = std::clamp(vtheta_current + a_theta * dt, 0.0, quad.vtheta_max);
        f_current      = std::clamp(f_current + delta_f * dt, quad.T_min, quad.T_max);

        // Plant RK4
        InputVec u_plant;
        u_plant << f_current, w_cmd(0), w_cmd(1), w_cmd(2);
        auto f_fn = [&](const PlantVec& xs, const InputVec& us) {
            return plant_f(xs, us, quad.mass, quad.g, quad.tau_rc);
        };
        x_plant = rk4_step<13, 4>(f_fn, x_plant, u_plant, dt);
        x_plant.segment<4>(6) = quat_normalize(x_plant.segment<4>(6));

        // References / logging
        Vec3  p_ref = pos_interp(theta_current, wp);
        Quat4 q_ref = quat_interp(theta_current, wp);
        double sn = find_nearest_s(x_plant.segment<3>(0), wp, s_nearest_prev, s_max);
        s_nearest_prev = sn;
        Vec3  p_ref_sn = pos_interp(sn, wp);
        Quat4 q_ref_sn = quat_interp(sn, wp);

        auto t_loop_end = std::chrono::steady_clock::now();
        double loop_ms = std::chrono::duration<double, std::milli>(
            t_loop_end - t_loop_start).count();

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

        double z = x_plant(2);
        if (z < z_min) { stop_reason = "floor"; break; }
        if (z > z_max) { stop_reason = "ceiling"; break; }
        if (sn >= s_max - 0.1) { stop_reason = "completed"; break; }
    }

    // Save CSV (schema identical to mpcc_sil / dq_mpcc_sil)
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

    save_csv("dq_mpcc_ablation_mil_results.csv");
    {
        const char* env_dir = std::getenv("QUADROTOR_RESULTS_DIR");
        std::string base    = env_dir ? std::string(env_dir)
#ifdef QUADROTOR_RESULTS_DEFAULT
                                      : std::string(QUADROTOR_RESULTS_DEFAULT);
#else
                                      : std::string("/tmp/quadrotor_mpc_results");
#endif
        std::string outdir  = base + "/mil/dq_mpcc_ablation";
        { std::error_code ec; std::filesystem::create_directories(outdir, ec); }
        std::string src = gain_source;
        for (auto& c : src) if (c == '/') c = '_';
        std::string persistent = outdir + "/dq_mpcc_ablation_mil_" + src + ".csv";
        save_csv(persistent);
        std::fprintf(stderr, "CSV saved: dq_mpcc_mil_results.csv + %s (%zu samples)\n",
                     persistent.c_str(), log_t.size());

    }

    auto pct = [](std::vector<double>& v, double q) {
        if (v.empty()) return 0.0;
        std::vector<double> s = v; std::sort(s.begin(), s.end());
        return s[std::min(s.size()-1, size_t(q * (s.size()-1)))];
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
