#!/usr/bin/env python3
"""
mpcc_on.py
==========
Activate MPCC and wait.
  - Ctrl+C runs the safe abort: goTo origin, then land
  - after the automatic landing, Ctrl+C exits cleanly

Usage:
  ros2 run flight_bringup mpcc_on
  ros2 run flight_bringup mpcc_on quadrotor   # quad_name as positional argument
"""

import sys
import time
import signal
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from mav_manager_srv.srv import Vec4


class MpccOnNode(Node):

    def __init__(self, quad_name: str):
        super().__init__('mpcc_on_node')
        self.quad = quad_name

    def call_bool(self, srv_name: str, data: bool):
        cli = self.create_client(SetBool, srv_name)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'Service not available: {srv_name}')
        future = cli.call_async(SetBool.Request(data=data))
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'Service call failed: {srv_name}')
        return result

    def call_trigger(self, srv_name: str):
        cli = self.create_client(Trigger, srv_name)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'Service not available: {srv_name}')
        future = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def call_goto(self, srv_name: str, x, y, z, yaw):
        cli = self.create_client(Vec4, srv_name)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'Service not available: {srv_name}')
        req = Vec4.Request()
        req.goal = [x, y, z, yaw]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def safe_shutdown(self):
        """Ctrl+C: exit the script without touching MPCC; the vehicle stays controlled."""
        print('\n[MPCC ON] Script closed; MPCC keeps running.')
        print(f'  To land: ros2 run quadrotor_mpc mpcc_off')

    def run(self):
        ns = f'/{self.quad}'
        self.get_logger().info(f'[MPCC ON] quad: {self.quad}')

        # Activate MPCC first (it waits for ≥2 publish ticks before returning),
        # THEN deactivate PD. This eliminates the no-command gap that lets
        # the drone fall and pre-degrades MPCC's initial state.
        self.get_logger().info('1/2 Activating MPCC (waiting for the controller)...')
        self.call_bool(f'{ns}/activate_mpcc', True)

        self.get_logger().info('2/2 Deactivating the flight controller (activate_pd)...')
        self.call_bool(f'{ns}/activate_pd', False)

        self.get_logger().info('MPCC active; Ctrl+C for a safe abort')

        # Flag set by the signal handler; rclpy is not called from the handler
        self._interrupt_flag = False
        self._z = 999.0
        self._was_airborne = False

        def _set_flag(sig, frame):
            self._interrupt_flag = True  # flag only, no rclpy calls here

        signal.signal(signal.SIGINT,  _set_flag)
        signal.signal(signal.SIGTSTP, _set_flag)

        from nav_msgs.msg import Odometry
        self.create_subscription(Odometry, f'/{self.quad}/odom',
            lambda msg: setattr(self, '_z', msg.pose.pose.position.z), 10)

        while True:
            try:
                rclpy.spin_once(self, timeout_sec=0.2)
            except Exception:
                pass

            # Manual interrupt (Ctrl+C / Ctrl+Z)
            if self._interrupt_flag:
                self.safe_shutdown()
                break

            # Automatic landing complete: exit
            if self._z > 0.5:
                self._was_airborne = True
            if self._was_airborne and self._z < 0.12:
                self.get_logger().info('Automatic landing complete, exiting')
                break

            if not rclpy.ok() and not self._interrupt_flag:
                break


def main(args=None):
    rclpy.init(args=args)
    quad = sys.argv[1] if len(sys.argv) > 1 else 'quadrotor'
    node = MpccOnNode(quad)
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
