"""Software-in-the-loop run of DQ-MPCC with the exact J_l^-1.

Usage:
  ros2 launch quadrotor_mpc sil_dq_mpcc.launch.py                          # deployed DQ-MPCC gains, trial 46
  ros2 launch quadrotor_mpc sil_dq_mpcc.launch.py pareto_path:=/abs/gains.json
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('quadrotor_mpc')
    default_path = os.path.join(share, 'tuning', 'sil', 'manual', 'pareto_dq_mpcc_ablation_manual.json')

    return LaunchDescription([
        DeclareLaunchArgument('pareto_path', default_value=default_path,
                              description='Gain file (default: deployed DQ-MPCC gains, trial 46)'),
        DeclareLaunchArgument('quad_name', default_value='quadrotor',
                              description='Namespace of the simulator topics and services'),
        ExecuteProcess(
            cmd=['ros2', 'run', 'quadrotor_mpc', 'dq_mpcc_sil',
                 '--pareto', LaunchConfiguration('pareto_path')],
            additional_env={'QUAD_NAME': LaunchConfiguration('quad_name')},
            output='screen'
        )
    ])
