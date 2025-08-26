import os
import sys
import argparse
import traceback

import rclpy # Import ROS2 Python client library
from .rta_mm_gpr_node import OffboardControl
from Logger import Logger, install_shutdown_logging # pyright: ignore[reportMissingImports]


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

    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    log = Logger(filename, base_path)

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