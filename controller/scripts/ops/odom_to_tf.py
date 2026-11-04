#!/usr/bin/env python3
"""
odom_to_tf.py — Broadcast TF from an Odometry topic so Foxglove/RViz get a
frame tree (the headless sim publishes no /tf). Non-invasive: run alongside
the foxglove_bridge when you want to visualize, kill it when done.

Reads <ns>/odom, broadcasts transform header.frame_id -> child_frame_id.
Falls back to world -> base_link if the message frames are empty.

Usage:
  python3 odom_to_tf.py                          # /quadrotor/odom
  python3 odom_to_tf.py --odom /quadrotor/odom
  python3 odom_to_tf.py --parent world --child base_link   # force frame names
"""
import argparse
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomToTF(Node):
    def __init__(self, odom_topic, parent, child):
        super().__init__("odom_to_tf")
        self.parent = parent
        self.child = child
        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, odom_topic, self.cb, 50)
        self.get_logger().info(
            f"odom_to_tf: {odom_topic} -> TF "
            f"({parent or 'msg.frame_id|world'} -> {child or 'msg.child_frame_id|base_link'})")

    def cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent or msg.header.frame_id or "world"
        t.child_frame_id = self.child or msg.child_frame_id or "base_link"
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom", default="/quadrotor/odom")
    ap.add_argument("--parent", default="", help="force parent frame (else msg.frame_id|world)")
    ap.add_argument("--child", default="", help="force child frame (else msg.child_frame_id|base_link)")
    args = ap.parse_args()
    rclpy.init()
    node = OdomToTF(args.odom, args.parent, args.child)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
