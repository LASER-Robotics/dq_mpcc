#include "quadrotor_mpc/trajectory/planner_ref.hpp"

#include <nlohmann/json.hpp>
#include <fstream>

namespace quadrotor_mpc {

bool load_planner_ref(const std::string& path, PlannerRef& r) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    auto j = nlohmann::json::parse(f);
    int n = j["n"];
    r.wp.s.resize(n); r.wp.pos.resize(n); r.wp.tang.resize(n); r.wp.quat.resize(n);
    r.wp.s_max = j["s_max"];
    for (int i = 0; i < n; ++i) {
        r.wp.s[i] = j["s"][i];
        r.wp.pos[i]  = Vec3(j["pos"][i][0], j["pos"][i][1], j["pos"][i][2]);
        r.wp.tang[i] = Vec3(j["tang"][i][0], j["tang"][i][1], j["tang"][i][2]);
        r.wp.quat[i] = Quat4(j["quat"][i][0], j["quat"][i][1],
                             j["quat"][i][2], j["quat"][i][3]);
    }
    r.hover_pos        = Vec3(j["hover_pos"][0], j["hover_pos"][1], j["hover_pos"][2]);
    r.s_analysis_start = j["s_analysis_start"];
    r.s_analysis_end   = j["s_analysis_end"];
    r.v_lap_avg        = j["v_lap_avg"];
    r.v_peak_G0        = j["v_peak_G0"];
    r.T_lap            = j["T_lap"];
    return true;
}

}  // namespace quadrotor_mpc
