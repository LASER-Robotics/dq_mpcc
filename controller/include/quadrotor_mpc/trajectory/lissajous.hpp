#pragma once

#include "quadrotor_mpc/common/types.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace quadrotor_mpc {

/// 3D Lissajous trajectory with frequency ratio 1:2:3.
///   x(t) = Ax*sin(1*w*t) + Cx
///   y(t) = Ay*sin(2*w*t)
///   z(t) = Az*sin(3*w*t) + Cz
///
/// Parameters loaded from config/trajectory.json (single source of truth).
class LissajousTrajectory {
public:
    double w  = 2.0;
    double Ax = 4.5, Ay = 1.25, Az = 0.75;
    double Cx = 2.5, Cz = 1.5;
    double t_final = 63.0;

    /// Load parameters from JSON file. Throws on missing file or keys.
    void load_from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("LissajousTrajectory: cannot open " + path);
        nlohmann::json j = nlohmann::json::parse(f);
        Ax      = j.at("Ax").get<double>();
        Ay      = j.at("Ay").get<double>();
        Az      = j.at("Az").get<double>();
        Cx      = j.at("Cx").get<double>();
        Cz      = j.at("Cz").get<double>();
        w       = j.at("w").get<double>();
        t_final = j.at("t_final").get<double>();
    }

    /// Load config/trajectory.json relative to the build directory (SiL binaries). Throws if not found.
    void load_default_json() {
        const std::string rel = "../config/trajectory.json";
        std::ifstream f(rel);
        if (!f.is_open())
            throw std::runtime_error("LissajousTrajectory: cannot find " + rel);
        f.close();
        load_from_json(rel);
    }

    /// Position at parameter t
    Vec3 position(double t) const;

    /// Velocity dp/dt at parameter t
    Vec3 velocity(double t) const;

    /// Sample N_samples points over [0, t_final]
    void sample(int N_samples,
                std::vector<Vec3>& positions,
                std::vector<Vec3>& velocities,
                std::vector<double>& t_values) const;
};

}  // namespace quadrotor_mpc
