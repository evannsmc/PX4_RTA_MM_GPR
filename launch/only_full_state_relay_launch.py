from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()


    full_state_relay = Node(
        package='mocap_px4_relay',
        executable='full_state_relay',
        output='screen',
    )

    ld.add_action(full_state_relay)
    return ld
