import os
import sys
import inspect
import argparse
import traceback

import rclpy # Import ROS2 Python client library
from .rta_mm_gpr_node import OffboardControl
# from .test_node import TestNode as OffboardControl
from Logger import Logger # type: ignore

BANNER = "=" * 65

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
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    print(f"{sim=}, {filename=}, {base_path=}")
    print(f"{'SIMULATION' if sim else 'HARDWARE'}")


    rclpy.init()
    offboard_control = OffboardControl(sim)
    logger = None


    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")

        try:
            if logger:
                logger.log(offboard_control)
            offboard_control.destroy_node()
        except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "<unknown>"
            print(f"\nError in {__name__}:{func_name}: {e}")
            traceback.print_exc()


        # Guard shutdown so it's called at most once
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"\nError in {__name__}: {e}")
            traceback.print_exc()


    try:
        print(f"{BANNER}\nInitializing ROS 2 node\n{BANNER}")
        logger = Logger(filename, base_path)
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected (Ctrl+C), exiting...")
    except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "<unknown>"
            print(f"\nError in {__name__}:{func_name}: {e}")
            traceback.print_exc()
    finally:
        shutdown_logging()
        print("\nNode has shut down.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "<unknown>"
            print(f"\nError in {__name__}:{func_name}: {e}")
            traceback.print_exc()
