import os
import sys
import argparse
import traceback

import rclpy # Import ROS2 Python client library
from .rta_mm_gpr_node import OffboardControl

BANNER = "=" * 65
print(f"{BANNER}\nInitializing ROS 2 node\n{BANNER}")

# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",
                        action=argparse.BooleanOptionalAction,
                        required=True)
    parser.add_argument("--log-file",
                        required=True)
    args, unknown = parser.parse_known_args(sys.argv[1:])
    print(f"Arguments: {args}, Unknown: {unknown}")
    sim = args.sim  # already a bool
    filename = args.log_file

    print(f"{sim=}, {filename=}")
    print(f"{'SIMULATION' if sim else 'HARDWARE'}")

    rclpy.init()
    offboard_control = OffboardControl(sim)

    # base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    # log = Logger(filename, base_path)

    # install_shutdown_logging(log, offboard_control)#, also_shutdown=rclpy.shutdown)# Ensure logs are flushed on Ctrl+C / SIGTERM / normal exit
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print(
              f"{BANNER}\nKeyboard interrupt detected (Ctrl+C), exiting...{BANNER}")
    except Exception as e:
        traceback.print_exc()
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()