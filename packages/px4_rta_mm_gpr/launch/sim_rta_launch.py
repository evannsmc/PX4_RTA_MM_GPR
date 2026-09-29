from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # The node reads PX4's vehicle_odometry / vehicle_local_position directly (no mocap_px4_relays relay).
    rta_mm_gpr = Node(
        package='px4_rta_mm_gpr',
        executable='px4_rta_mm_gpr',
        output='screen',
        arguments=['--sim', '--log-file', 'log.log']
    )
    return LaunchDescription([rta_mm_gpr])
