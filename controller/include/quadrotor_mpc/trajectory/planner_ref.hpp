#pragma once
/**
 * planner_ref.hpp — Shared loader for the path planner's reference JSON
 * (path_planing/ref_flat_waypoints.json, produced by export_ref_json.py).
 *
 * Used by every executable that consumes the multilap PMM reference:
 *   mpcc_mil, mpcc_sil, mpcc_node, and their dq_mpcc counterparts.
 *
 * The JSON is the single source of truth for:
 *   - geometric waypoints (arc-length parameterised)
 *   - hover_pos               (start/end point of entry/exit)
 *   - analysis window         (lap 2+3 bounds in arc-length)
 *   - v_peak_G0 / v_lap_avg   (natural speed of the periodic lap → v_θ_max)
 */

#include "quadrotor_mpc/trajectory/attitude_reference.hpp"
#include <string>

namespace quadrotor_mpc {

struct PlannerRef {
    PathWaypoints wp;
    Vec3   hover_pos        = Vec3::Zero();
    double s_analysis_start = 0.0;
    double s_analysis_end   = 0.0;
    double v_lap_avg        = 0.0;
    double v_peak_G0        = 0.0;   // → recommended v_θ_max
    double T_lap            = 0.0;
};

/// Load the planner reference (geometry + metadata) from JSON.
/// Returns false if the file is missing or malformed.
bool load_planner_ref(const std::string& path, PlannerRef& out);

}  // namespace quadrotor_mpc
