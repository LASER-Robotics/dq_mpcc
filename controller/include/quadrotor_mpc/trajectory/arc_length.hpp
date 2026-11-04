#pragma once

#include "quadrotor_mpc/common/types.hpp"
#include <vector>

namespace quadrotor_mpc {

/// Arc-length parameterised path with piecewise-linear interpolation.
///
/// Built from dense time samples via trapezoidal integration of ||dp/dt||.
/// After construction, query position/tangent/quaternion at any arc-length s.
class ArcLengthPath {
public:
    /// Build from sampled positions and velocities (dp/dt).
    /// Computes cumulative arc-length via trapezoidal integration.
    void build(const std::vector<Vec3>& positions,
               const std::vector<Vec3>& velocities,
               const std::vector<double>& t_values);

    /// Total arc-length of the path
    double s_max() const { return arc_lengths_.back(); }

    /// Position at arc-length s (piecewise-linear interpolation)
    Vec3 position(double s) const;

    /// Unit tangent at arc-length s
    Vec3 tangent(double s) const;

    /// Raw arc-length values (for building waypoints)
    const std::vector<double>& arc_lengths() const { return arc_lengths_; }
    const std::vector<Vec3>&   positions()   const { return positions_; }

private:
    std::vector<double> arc_lengths_;
    std::vector<Vec3>   positions_;
    std::vector<Vec3>   tangents_;  // unit tangents at each sample

    /// Find segment index for arc-length s (binary search)
    int find_segment(double s) const;

    /// Lerp between two values
    static Vec3 lerp(const Vec3& a, const Vec3& b, double alpha) {
        return (1.0 - alpha) * a + alpha * b;
    }
};

}  // namespace quadrotor_mpc
