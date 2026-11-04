#!/usr/bin/env python3
"""
dq_mpcc_on.py
=============
Activate DQ-MPCC and wait.
  - Ctrl+C exits the script; DQ-MPCC keeps running
  - after the automatic landing, the script exits

Usage:
  ros2 run flight_bringup dq_mpcc_on
  ros2 run flight_bringup dq_mpcc_on quadrotor   # quad_name as positional argument
"""

import sys
import time
import signal
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from mav_manager_srv.srv import Vec4


class DqMpccOnNode(Node):

    def __init__(self, quad_name: str):
        super().__init__('dq_mpcc_on_node')
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

    def safe_shutdown(self):
        """Ctrl+C: exit the script without touching DQ-MPCC; the vehicle stays controlled."""
        print('\n[DQ-MPCC ON] Script closed; DQ-MPCC keeps running.')
        print(f'  To land: ros2 run quadrotor_mpc dq_mpcc_off')

    def run(self):
        ns = f'/{self.quad}'
        self.get_logger().info(f'[DQ-MPCC ON] quad: {self.quad}')

        # Activate DQ-MPCC first (waits for ≥2 publish ticks); THEN deactivate PD.
        # Eliminates the no-command gap.
        self.get_logger().info('1/2 Activating DQ-MPCC (waiting for the controller)...')
        self.call_bool(f'{ns}/activate_dq_mpcc', True)

        self.get_logger().info('2/2 Deactivating the flight controller (activate_pd)...')
        self.call_bool(f'{ns}/activate_pd', False)

        self.get_logger().info('DQ-MPCC active; Ctrl+C for a safe abort')

        self._interrupt_flag = False
        self._z = 999.0
        self._was_airborne = False

        def _set_flag(sig, frame):
            self._interrupt_flag = True

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

            if self._interrupt_flag:
                self.safe_shutdown()
                break

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
    node = DqMpccOnNode(quad)
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
