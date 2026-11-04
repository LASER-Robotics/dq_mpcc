#include "quadrotor_mpc/trajectory/arc_length.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>

namespace quadrotor_mpc {

void ArcLengthPath::build(const std::vector<Vec3>& positions,
                           const std::vector<Vec3>& velocities,
                           const std::vector<double>& t_values) {
    const int N = static_cast<int>(positions.size());
    assert(N >= 2);

    // Cumulative arc-length via trapezoidal integration of ||dp/dt||
    arc_lengths_.resize(N);
    arc_lengths_[0] = 0.0;
    for (int i = 1; i < N; ++i) {
        double dt = t_values[i] - t_values[i-1];
        double speed_prev = velocities[i-1].norm();
        double speed_curr = velocities[i].norm();
        arc_lengths_[i] = arc_lengths_[i-1] + 0.5 * (speed_prev + speed_curr) * dt;
    }

    // Store positions
    positions_ = positions;

    // Compute unit tangents via finite differences on positions w.r.t. arc-length
    tangents_.resize(N);
    for (int i = 0; i < N; ++i) {
        Vec3 tang;
        if (i == 0) {
            tang = positions[1] - positions[0];
        } else if (i == N - 1) {
            tang = positions[N-1] - positions[N-2];
        } else {
            tang = positions[i+1] - positions[i-1];
        }
        double n = tang.norm();
        tangents_[i] = (n > 1e-12) ? tang / n : Vec3(1.0, 0.0, 0.0);
    }
}

int ArcLengthPath::find_segment(double s) const {
    // Binary search for the segment containing s
    auto it = std::upper_bound(arc_lengths_.begin(), arc_lengths_.end(), s);
    int idx = static_cast<int>(it - arc_lengths_.begin()) - 1;
    return std::clamp(idx, 0, static_cast<int>(arc_lengths_.size()) - 2);
}

Vec3 ArcLengthPath::position(double s) const {
    s = std::clamp(s, arc_lengths_.front(), arc_lengths_.back());
    int i = find_segment(s);
    double ds = arc_lengths_[i+1] - arc_lengths_[i];
    double alpha = (ds > 1e-12) ? (s - arc_lengths_[i]) / ds : 0.0;
    return lerp(positions_[i], positions_[i+1], alpha);
}

Vec3 ArcLengthPath::tangent(double s) const {
    s = std::clamp(s, arc_lengths_.front(), arc_lengths_.back());
    int i = find_segment(s);
    double ds = arc_lengths_[i+1] - arc_lengths_[i];
    double alpha = (ds > 1e-12) ? (s - arc_lengths_[i]) / ds : 0.0;
    Vec3 t = lerp(tangents_[i], tangents_[i+1], alpha);
    double n = t.norm();
    return (n > 1e-12) ? t / n : Vec3(1.0, 0.0, 0.0);
}

}  // namespace quadrotor_mpc
