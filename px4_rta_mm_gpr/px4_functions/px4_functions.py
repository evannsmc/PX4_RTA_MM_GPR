from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
)


def arm(self) -> None:
    """Send an arm command to the vehicle."""
    publish_vehicle_command(self,
        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
    self.get_logger().info('Arm command sent')

def disarm(self) -> None:
    """Send a disarm command to the vehicle."""
    publish_vehicle_command(self,
        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
    self.get_logger().info('Disarm command sent')

def engage_offboard_mode(self) -> None:
    """Switch to offboard mode."""
    publish_vehicle_command(self,
        VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
    self.get_logger().info("Switching to offboard mode")

def land(self) -> None:
    """Switch to land mode."""
    publish_vehicle_command(self,
        VehicleCommand.VEHICLE_CMD_NAV_LAND)
    self.get_logger().info("Switching to land mode")

def publish_offboard_control_heartbeat_signal_position(self) -> None:
    """Publish the offboard control mode heartbeat for position-only setpoints."""
    msg = OffboardControlMode()
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    msg.position = True
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    msg.body_rate = False
    msg.thrust_and_torque = False
    msg.direct_actuator = False
    self.offboard_control_mode_publisher.publish(msg)
    # self.get_logger().info("Switching to position control mode")

def publish_offboard_control_heartbeat_signal_body_rate(self) -> None:
    """Publish the offboard control mode heartbeat for body rate setpoints."""
    msg = OffboardControlMode()
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    msg.position = False
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    msg.body_rate = True
    msg.thrust_and_torque = False
    msg.direct_actuator = False
    self.offboard_control_mode_publisher.publish(msg)
    # self.get_logger().info("Switching to body rate control mode")

def publish_position_setpoint(self, x: float = 0.0, y: float = 0.0, z: float = -3.0, yaw: float = 0.0) -> None:
    """Publish the trajectory setpoint.

    Args:
        x (float, optional): Desired x position in meters. Defaults to 0.0.
        y (float, optional): Desired y position in meters_. Defaults to 0.0.
        z (float, optional): Desired z position in meters. Defaults to -3.0.
        yaw (float, optional): Desired yaw position in radians. Defaults to 0.0.

    Returns:
        None

    Raises:
        TypeError: If x, y, z, or yaw are not of type float.
    Raises:
        ValueError: If x, y, z are not within the expected range.
    """
    for name, val in zip(("x","y","z","yaw"), (x,y,z,yaw)):
        if not isinstance(val, float):
            raise TypeError(
                            f"\n{'=' * 60}"
                            f"\nInvalid input type for {name}\n"
                            f"Expected float\n"
                            f"Received {type(val).__name__}\n"
                            f"{'=' * 60}"
                            )
            
    # if not (-2.0 <= x <= 2.0) or not (-2.0 <= y <= 2.0) or not (-3.0 <= z <= -0.2):
    #     raise ValueError("x must be between -2.0 and 2.0, y must be between -2.0 and 2.0, z must be between -0.2 and -3.0")
    
    msg = TrajectorySetpoint()
    msg.position = [x, y, z] # position in meters
    msg.yaw = yaw # yaw in radians
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.trajectory_setpoint_publisher.publish(msg)
    self.get_logger().info(f"Publishing position setpoints {[x, y, z, yaw]}")

def publish_body_rate_setpoint(self, throttle: float = 0.0, p: float = 0.0, q: float = 0.0, r: float = 0.0) -> None:
    """Publish the body rate setpoint.
    
    Args:
        p (float): Desired roll rate in radians per second.
        q (float): Desired pitch rate in radians per second.
        r (float): Desired yaw rate in radians per second.
        throttle (float): Desired throttle in normalized from [-1,1] in NED body frame

    Returns:
        None
    
    Raises:
        ValueError: If p, q, r, or throttle are not within expected ranges.
    """

    
    # print(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, throttle={throttle}")
    msg = VehicleRatesSetpoint()
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    msg.roll = p
    msg.pitch = q
    msg.yaw = r
    msg.thrust_body[0] = 0.0
    msg.thrust_body[1] = 0.0
    msg.thrust_body[2] = -1 * float(throttle)
    self.vehicle_rates_setpoint_publisher.publish(msg)
    self.get_logger().info(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, thrust_body={throttle}")

def publish_vehicle_command(self, command, **params) -> None:
    """Publish a vehicle command."""
    msg = VehicleCommand()
    msg.command = command
    msg.param1 = params.get("param1", 0.0)
    msg.param2 = params.get("param2", 0.0)
    msg.param3 = params.get("param3", 0.0)
    msg.param4 = params.get("param4", 0.0)
    msg.param5 = params.get("param5", 0.0)
    msg.param6 = params.get("param6", 0.0)
    msg.param7 = params.get("param7", 0.0)
    msg.target_system = 1
    msg.target_component = 1
    msg.source_system = 1
    msg.source_component = 1
    msg.from_external = True
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.vehicle_command_publisher.publish(msg)

