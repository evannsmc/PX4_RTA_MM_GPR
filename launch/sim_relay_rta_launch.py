from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()


    full_state_relay = Node(
        package='mocap_px4_relays',
        executable='full_state_relay',
        output='screen',
        # PX4 >= 1.16 publishes the versioned topic name; the relay still subscribes to the old one
        remappings=[('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position_v1')],
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
