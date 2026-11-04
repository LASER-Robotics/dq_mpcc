"""
experiment.launch.py
====================
Controller layer for the integrated flight stack (real vehicle or MuJoCo).

  dq_nmpc_control_nodelet (package dq_cpp) -> flight controller for takeoff,
       hover and landing, following `position_cmd`. Exposes the `activate_pd`
       service used for the handoff with the racing controllers.
  mpcc_node / dq_mpcc_node / dq_mpcc_ablation_node -> racing controllers
       (standby until activate_*).
  pi_bridge_node (package pi_ros2_interface) -> serial bridge to the flight
       controller board. Only with with_pi:=true (default); use with_pi:=false in simulation.

Arguments:
  ctrl:=mpcc|dq_mpcc|dq_ablation|all   racing controller(s) to start (default all)
  weights_file_*:=<path>               override a gain file (defaults: deployed gains)
  quad_name:=<name>                    namespace
  with_pi:=false                       do not start the serial bridge (simulation)
  platform_type:=<name>                platform configuration of the flight stack (mixer, dq_cpp parameters)

Usage:
  T1:  ros2 launch flight_bringup flight_stack.launch.py [quad_name:=...]
  T2:  ros2 launch quadrotor_mpc experiment.launch.py ctrl:=mpcc [quad_name:=...] [with_pi:=false]
  T3:  ros2 run flight_bringup takeoff   # activates dq_cpp (activate_pd) and the motors
       ros2 run quadrotor_mpc mpcc_on    # hands over to MPCC; landing is automatic
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Deployed gain files (MPCC trial 76, DQ-MPCC trial 46). The exact-J_l^-1
    # variant (dq_mpcc) runs at the deployed DQ-MPCC gains.
    share = get_package_share_directory('quadrotor_mpc')
    manual = os.path.join(share, 'tuning', 'sil', 'manual')
    default_mpcc = os.path.join(manual, 'pareto_mpcc_manual.json')
    default_abl  = os.path.join(manual, 'pareto_dq_mpcc_ablation_manual.json')
    default_dq   = default_abl

    launch_args = [
        DeclareLaunchArgument('quad_name',    default_value='quadrotor',
                              description='Vehicle namespace'),
        DeclareLaunchArgument('ctrl', default_value='all',
                              description='Racing controller to start: mpcc | dq_mpcc | dq_ablation | all'),
        DeclareLaunchArgument('weights_file_mpcc', default_value=default_mpcc,
                              description='MPCC gain file'),
        DeclareLaunchArgument('weights_file_dq',   default_value=default_dq,
                              description='Gain file for the exact-J_l^-1 variant'),
        DeclareLaunchArgument('weights_file_abl',  default_value=default_abl,
                              description='DQ-MPCC gain file'),
        DeclareLaunchArgument("use_nmpc", default_value="true"),
        DeclareLaunchArgument("platform_type", default_value="quadrotor"),
        DeclareLaunchArgument("pi_serial_device", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("pi_baud", default_value="921600"),
        DeclareLaunchArgument("pi_rx_rate_hz", default_value="1000.0"),
        DeclareLaunchArgument("pi_tx_rate_hz", default_value="500.0"),
        DeclareLaunchArgument("pi_beta_states_rate_hz", default_value="500.0"),
        DeclareLaunchArgument("pi_onboard_control_rate_hz", default_value="500.0"),
        DeclareLaunchArgument("with_pi", default_value="true",
                              description="Launch pi_bridge_node (serial→Betaflight). Set to false in simulation."),
        DeclareLaunchArgument("with_flight_stack", default_value="true",
                              description="Include flight_stack.launch.py (mav_manager and trackers). false: start it separately."),
    ]

    # Include flight_stack (T1) automatically — one-shot T1+T2 launch
    flight_stack_share = get_package_share_directory('flight_bringup')
    flight_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(flight_stack_share, 'launch', 'flight_stack.launch.py')
        ),
        launch_arguments={'quad_name': LaunchConfiguration('quad_name')}.items(),
        condition=IfCondition(LaunchConfiguration('with_flight_stack')),
    )

    quad_name         = LaunchConfiguration('quad_name')
    platform_type = LaunchConfiguration("platform_type")
    ctrl              = LaunchConfiguration('ctrl')
    weights_file_mpcc = LaunchConfiguration('weights_file_mpcc')
    weights_file_dq   = LaunchConfiguration('weights_file_dq')
    weights_file_abl  = LaunchConfiguration('weights_file_abl')
    pi_serial_device = LaunchConfiguration("pi_serial_device")
    pi_baud = LaunchConfiguration("pi_baud")
    pi_rx_rate_hz = LaunchConfiguration("pi_rx_rate_hz")
    pi_tx_rate_hz = LaunchConfiguration("pi_tx_rate_hz")
    pi_beta_states_rate_hz = LaunchConfiguration("pi_beta_states_rate_hz")
    pi_onboard_control_rate_hz = LaunchConfiguration("pi_onboard_control_rate_hz")
    use_nmpc = LaunchConfiguration("use_nmpc")
    with_pi = LaunchConfiguration("with_pi")

    control_config = PathJoinSubstitution( [get_package_share_directory("acp_autonomy"),
            "config",
            platform_type,
            "default",
            "dq_control.yaml",
        ]
    )



    def _ctrl_in(*names):
        # True if ctrl is in names, or ctrl is 'all'.
        opts = '","'.join(names)
        return IfCondition(PythonExpression(['"', ctrl, '" in ("all","', opts, '")']))

    def _ctrl_exact(*names):
        # True only if ctrl is in names (not included in 'all').
        # Used for dq_mpcc (exact J_l^-1): 'all' starts only mpcc and dq_ablation.
        opts = '","'.join(names)
        return IfCondition(PythonExpression(['"', ctrl, '" in ("', opts, '")']))

    # Flight controller: dual-quaternion NMPC (dq_cpp)
    # Follows `position_cmd` for takeoff, hover and landing.
    # Exposes the `activate_pd` service used for the handoff with the racing controllers.
    flight_ctrl_container = ComposableNodeContainer(
        name='flight_ctrl_container',
        namespace=quad_name,
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='dq_cpp',
                plugin='dq_nmpc_control_nodelet::NMPCControlNodelet',
                namespace=quad_name,
                name='dq_nmpc_control_nodelet',
                parameters=[control_config, {'platform_type': platform_type}],
            ),
        ],
        output='screen',
    )

    # MPCC node
    mpcc_node = Node(
        package='quadrotor_mpc',
        executable='mpcc_node',
        name='mpcc_node',
        namespace='',           # the node sets its own namespace through a parameter
        parameters=[{
            'quad_name':     quad_name,
            'weights_file':  weights_file_mpcc,
        }],
        condition=_ctrl_in('mpcc'),
        output='screen',
    )

    # DQ-MPCC (exact J_l^-1) node (optional)
    dq_mpcc_node = Node(
        package='quadrotor_mpc',
        executable='dq_mpcc_node',
        name='dq_mpcc_node',
        namespace='',
        parameters=[{
            'quad_name':     quad_name,
            'weights_file':  weights_file_dq,
        }],
        output='screen',
        condition=_ctrl_exact('dq_mpcc'),
    )

    # DQ-MPCC node (optional)
    dq_ablation_node = Node(
        package='quadrotor_mpc',
        executable='dq_mpcc_ablation_node',
        name='dq_mpcc_ablation_node',
        namespace='',
        parameters=[{
            'quad_name':     quad_name,
            'weights_file':  weights_file_abl,
        }],
        output='screen',
        condition=_ctrl_in('dq_ablation'),
    )

    pi_bridge_node = Node(
        package="pi_ros2_interface",
        executable="pi_bridge_node_exec",
        namespace=quad_name,
        name="pi_bridge_node",
        parameters=[{
            "serial_device": pi_serial_device,
            "baud": pi_baud,
            "rx_rate_hz": pi_rx_rate_hz,
            "tx_rate_hz": pi_tx_rate_hz,
            "beta_states_rate_hz": pi_beta_states_rate_hz,
            "onboard_control_rate_hz": pi_onboard_control_rate_hz,
            "control_type": use_nmpc,
            }, control_config],
        output="screen",
        condition=IfCondition(with_pi),
        )


    ld = LaunchDescription(launch_args)
    ld.add_action(flight_stack_launch)
    ld.add_action(flight_ctrl_container)
    ld.add_action(mpcc_node)
    ld.add_action(dq_mpcc_node)
    ld.add_action(dq_ablation_node)
    ld.add_action(pi_bridge_node)
    return ld
