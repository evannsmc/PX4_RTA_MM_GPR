from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Relay + C++ fast loop (PX4 I/O, 100 Hz control, watchdog) + Python planner (rollouts, GP, wind, LQR)."""
    full_state_relay = Node(
        package='mocap_px4_relays',
        executable='full_state_relay',
        output='screen',
        # PX4 >= 1.16 publishes the versioned topic name; the relay still subscribes to the old one
        remappings=[('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position_v1')],
    )
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
    return LaunchDescription([full_state_relay, fast_loop, planner])
