from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()


    full_state_relay = Node(
        package='mocap_px4_relay',
        executable='full_state_relay',
        output='screen',
    )

    rta_mm_gpr = Node(
        package='px4_rta_mm_gpr',
        executable='px4_rta_mm_gpr',
        output='screen',
        arguments=['--sim', '--log-file', 'log.log']
    )

    ld.add_action(full_state_relay)
    ld.add_action(rta_mm_gpr)
    return ld
