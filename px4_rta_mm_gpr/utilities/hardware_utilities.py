import math as m

MASS: float = 1.75 # (kg) mass of the multirotor

def get_throttle_command_from_force(collective_thrust) -> float: #Converts force to throttle command
    """ Convert the positive collective thrust force to a positive throttle command. """
    print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
    try:
        a = 0.00705385408507030
        b = 0.0807474474438391
        c = 0.0252575818743285
        throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c  # equation form is a*x + b*sqrt(x) + c = y
        return throttle_command

    except Exception as e:
        print(f"Error in hardware throttle conversion (non-sim mode): {e}")
        raise  # Raise the exception to ensure the error is handled properly