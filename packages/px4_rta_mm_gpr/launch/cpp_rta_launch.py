from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """C++ fast loop (PX4 I/O, 100 Hz control, watchdog) + Python planner (rollouts, GP, wind, LQR)."""
    fast_loop = Node(
        package='px4_rta_mm_gpr_cpp',
        executable='rta_fast_loop',
        output='screen',
    )
    planner = Node(
        package='px4_rta_mm_gpr',
        executable='px4_rta_mm_gpr',
        output='screen',
        arguments=['--sim', '--log-file', 'log.log', '--cpp-control',
                   '--executor', 'events', '--rollout-backend', 'process'],
    )
    return LaunchDescription([fast_loop, planner])  # both read PX4's topics directly (no relay)
