#!/usr/bin/env python3
"""play_multilap.py — Publish /quadrotor/position_cmd from multilap reference.

Replays a slice of ref_flat_waypoints.npz (arc-length s ∈ [s_start, s_end])
over time at 100 Hz. Drives dq_cpp NMPC for the entry+warmup phase (hover→G0)
or warmup_out+exit phase (decel→hover) in Mode B handoff architecture.

Usage:
  ros2 run quadrotor_mpc play_multilap entry            # s∈[0, ~26m] hover → G0 @ v_full
  ros2 run quadrotor_mpc play_multilap exit             # s∈[~92m, end] decel → hover
  ros2 run quadrotor_mpc play_multilap entry quad=quadrotor

Stops when:
  - All ref samples published, OR
  - Ctrl+C (gracefully).
"""

import os
import sys
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from quadrotor_msgs.msg import PositionCommand


def _path_planing_dir():
    """Reference files installed with the package (share/quadrotor_mpc/path_planing)."""
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("quadrotor_mpc"), "path_planing")


REF_NPZ_DEFAULT = os.path.join(_path_planing_dir(), "ref_flat_waypoints.npz")
META_NPZ_DEFAULT = os.path.join(_path_planing_dir(), "multilap_metadata.npz")
DT_PUB = 0.01   # 100 Hz


def quat_to_yaw(q_wxyz):
    w, x, y, z = q_wxyz
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class PlayMultilap(Node):
    def __init__(self, quad: str, phase: str, ref_npz: str, meta_npz: str):
        super().__init__("play_multilap")
        self.quad = quad
        self.phase = phase

        d = np.load(ref_npz)
        # Use time-grid arrays (continuous) instead of waypoint sampling
        self.t_ref = d["t"]                # (N,)
        self.p_t  = d["p_t"]               # (3, N)
        self.v_t  = d["v_t"]               # (3, N)
        self.a_s  = d["a_smooth_t"]        # (3, N)
        self.q_t  = d["q_t"]               # (4, N)  [wxyz]
        self.s_t  = d["s_t"]               # (N,)
        N_total = len(self.t_ref)

        m = np.load(meta_npz, allow_pickle=True)
        ps = dict(m["phase_starts"])  # {phase: idx}
        pe = dict(m["phase_ends"])

        if phase == "entry":
            k_lo = 0
            # End at lap1 START (= warmup end). Drone reaches G0 at v_full there.
            k_hi = int(ps.get("lap1", N_total))
        elif phase == "exit":
            # Start at warmup_out (= lap3 end). Drone decel from v_full → hover.
            k_lo = int(ps.get("warmup_out", int(ps.get("exit", N_total // 2))))
            k_hi = N_total
        else:
            raise ValueError(f"phase must be 'entry' or 'exit', got: {phase}")

        self.k_lo, self.k_hi = k_lo, k_hi
        self.k_cur = k_lo
        self.t0 = None

        self.pub = self.create_publisher(
            PositionCommand, f"/{quad}/position_cmd", 10
        )
        # Tick at 100 Hz
        self.timer = self.create_timer(DT_PUB, self._tick)

        self.get_logger().info(
            f"[play_multilap] phase={phase}  range k=[{k_lo}..{k_hi}]  "
            f"s=[{self.s_t[k_lo]:.2f}..{self.s_t[k_hi-1]:.2f}]m  "
            f"duration={self.t_ref[k_hi-1]-self.t_ref[k_lo]:.2f}s"
        )

    def _tick(self):
        if self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds * 1e-9

        # Time-based index advance: real_t since start, plus k_lo's t_ref offset
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        t_target = self.t_ref[self.k_lo] + elapsed
        # Find first k whose t_ref >= t_target
        k = self.k_lo + int(np.searchsorted(self.t_ref[self.k_lo:self.k_hi], t_target))

        if k >= self.k_hi:
            self.get_logger().info(
                f"[play_multilap] phase={self.phase} complete (k={k})"
            )
            self.timer.cancel()
            return

        self.k_cur = k

        msg = PositionCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.position.x, msg.position.y, msg.position.z = (
            float(self.p_t[0, k]), float(self.p_t[1, k]), float(self.p_t[2, k])
        )
        msg.velocity.x, msg.velocity.y, msg.velocity.z = (
            float(self.v_t[0, k]), float(self.v_t[1, k]), float(self.v_t[2, k])
        )
        msg.acceleration.x, msg.acceleration.y, msg.acceleration.z = (
            float(self.a_s[0, k]), float(self.a_s[1, k]), float(self.a_s[2, k])
        )
        msg.jerk.x = msg.jerk.y = msg.jerk.z = 0.0
        msg.yaw = quat_to_yaw(self.q_t[:, k])
        msg.yaw_dot = 0.0

        self.pub.publish(msg)


def main():
    rclpy.init()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    phase = args[0]
    quad = "quadrotor"
    for a in args[1:]:
        if a.startswith("quad="):
            quad = a.split("=", 1)[1]
    node = PlayMultilap(quad, phase, REF_NPZ_DEFAULT, META_NPZ_DEFAULT)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
