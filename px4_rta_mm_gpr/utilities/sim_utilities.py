import math as m

MASS: float = 1.92 # (kg) mass of the multirotor

THRUST_CONSTANT: float = 8.54858e-06 #x500 gazebo simulation motor thrust constant
MOTOR_VELOCITY_ARMED: float = 10.0 #x500 gazebo motor velocity when armed
MAX_ROTOR_SPEED: float = 1000.0 #x500 gazebo simulation max rotor speed
MOTOR_INPUT_SCALING: float = MAX_ROTOR_SPEED #x500 gazebo simulation motor input scaling



def get_throttle_command_from_force(collective_thrust) -> float: #Converts force to throttle command
    """ Convert the positive collective thrust force to a positive throttle command. """
    print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
    try:
        motor_speed = m.sqrt(collective_thrust / (4.0 * THRUST_CONSTANT))
        throttle_command = (motor_speed - MOTOR_VELOCITY_ARMED) / MOTOR_INPUT_SCALING
        return throttle_command
    except Exception as e:
        print(f"Error in sim throttle conversion: {e}")
        raise  # Raise the exception to ensure the error is handled properly
