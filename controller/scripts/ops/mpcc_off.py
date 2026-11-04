#!/usr/bin/env python3
"""
mpcc_off.py
===========
Safe manual abort of MPCC: return to the origin, then land.
  1. Deactivate MPCC
  2. Activate the flight controller (4 s to settle, the vehicle arrives with path velocity)
  3. goTo (0, 0, 1.07)  return to the origin
  4. land

Usage:
  ros2 run flight_bringup mpcc_off
  ros2 run flight_bringup mpcc_off quadrotor   # quad_name as positional argument
"""

import sys
import time
import signal
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger
from mav_manager_srv.srv import Vec4


class MpccOffNode(Node):

    def __init__(self, quad_name: str):
        super().__init__('mpcc_off_node')
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
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'Service call failed: {srv_name}')
        return result

    def call_goto(self, srv_name: str, x: float, y: float, z: float, yaw: float):
        cli = self.create_client(Vec4, srv_name)
        if not cli.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f'Service not available: {srv_name}')
        req = Vec4.Request()
        req.goal = [x, y, z, yaw]
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('Service call failed: goTo')
        return result

    def safe_shutdown(self):
        """Safe sequence: land in place where the racing exit left the vehicle."""
        mav = f'/{self.quad}/mav_services'
        try:
            self.get_logger().info('→ Landing in place (the vehicle is already at the exit point, about [2, 0, 1.5])...')
            self.call_trigger(f'{mav}/land')
            self.get_logger().info('Land OK')
        except Exception as e:
            self.get_logger().error(f'safe_shutdown failed: {e}')

    def run(self):
        ns  = f'/{self.quad}'
        mav = f'/{self.quad}/mav_services'
        self.get_logger().info(f'[MPCC OFF] quad: {self.quad}')

        # Ctrl+C and Ctrl+Z both run the safe sequence
        def _signal_handler(sig, frame):
            self.get_logger().warn('Interrupt received, running the safe sequence...')
            self.safe_shutdown()
            rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTSTP, _signal_handler)  # Ctrl+Z

        self.get_logger().info('1/2 Deactivating MPCC, activating the flight controller...')
        self.call_bool(f'{ns}/activate_mpcc', False)
        self.call_bool(f'{ns}/activate_pd', True)

        self.get_logger().info('2/2 Safe sequence: returning to the origin...')
        self.safe_shutdown()


def main(args=None):
    rclpy.init(args=args)
    quad = sys.argv[1] if len(sys.argv) > 1 else 'quadrotor'
    node = MpccOffNode(quad)
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
