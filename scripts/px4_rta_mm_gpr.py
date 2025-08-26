import os
import sys
import traceback

import rclpy # Import ROS2 Python client library
from .rta_mm_gpr_node import OffboardControl
from Logger import Logger, install_shutdown_logging # pyright: ignore[reportMissingImports]


# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):

    f"{65 * '='}\n"
    f"Initializing ROS 2 node: '{__name__}' for offboard control\n"
    f"{65 * '='}\n"

    # Figure out if in simulation or hardware mode to set important variables to the appropriate values
    sim_val = int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: "))
    if sim_val not in (0, 1):
        raise ValueError(
                        f"\n{65 * '='}\n"
                        f"Invalid input for sim: {sim_val}, expected 0 or 1\n"
                        f"{65 * '='}\n")
    sim = bool(sim_val)
    print(f"{'SIMULATION' if sim else 'HARDWARE'}")

    rclpy.init(args=args)
    offboard_control = OffboardControl(sim)

    filename = sys.argv[1] if len(sys.argv) > 1 else "log.log"
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    base_dir = sys.argv[2] if len(sys.argv) > 2 else base_path
    log = Logger(filename, base_dir)

    install_shutdown_logging(log, offboard_control)#, also_shutdown=rclpy.shutdown)# Ensure logs are flushed on Ctrl+C / SIGTERM / normal exit
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print(
              f"\n{65 * '='}\n"
              f"Keyboard interrupt detected (Ctrl+C), exiting...\n"
              )
    except Exception as e:
        traceback.print_exc()
    finally:
        # belt & suspenders
        pass

if __name__ == '__main__':
    main()