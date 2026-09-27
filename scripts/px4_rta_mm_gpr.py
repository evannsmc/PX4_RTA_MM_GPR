import os
import sys
import inspect
import argparse
import traceback

import rclpy # Import ROS2 Python client library
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from .rta_mm_gpr_node import OffboardControl, RuntimeOptions
# from .test_node import TestNode as OffboardControl
try:
    from ros2_logger import Logger # ROS2Logger >= Mar 2026 (package renamed)
except ImportError:
    from Logger import Logger # type: ignore

BANNER = "=" * 65


def parse_cpu_list(text):
    """'2,3' or '2-5' -> frozenset({2, 3}) / frozenset({2, 3, 4, 5})"""
    if not text:
        return None
    cpus = set()
    for part in text.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return frozenset(cpus)


def make_executor(kind: str, num_threads: int):
    """Build the executor that decides how (and whether in parallel) the node's callbacks run."""
    if kind == 'multi':
        # One thread per callback group is enough: callbacks in the same mutually-exclusive group
        # never run concurrently, so extra threads would just sit idle.
        return MultiThreadedExecutor(num_threads=num_threads)
    if kind == 'single':
        return SingleThreadedExecutor() # the original behaviour of rclpy.spin(node)
    if kind == 'events':
        # Single-threaded, but the wait/dispatch loop is implemented in C++ (much lower overhead than
        # the pure-Python executors). Pairs well with --rollout-backend process.
        from rclpy.experimental import EventsExecutor
        return EventsExecutor()
    raise ValueError(f"unknown executor '{kind}'")


# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",
                        action=argparse.BooleanOptionalAction,
                        required=True)
    parser.add_argument("--log-file",
                        required=True)
    parser.add_argument("--executor", choices=['multi', 'single', 'events'], default='multi',
                        help="multi: MultiThreadedExecutor with one thread per callback group (default); "
                             "single: SingleThreadedExecutor (original behaviour); "
                             "events: rclpy.experimental.EventsExecutor (C++ single-threaded)")
    parser.add_argument("--threads", type=int, default=None,
                        help="worker threads for --executor multi (default: number of callback groups)")
    parser.add_argument("--rollout-backend", choices=['thread', 'process'], default='thread',
                        help="thread: rollout runs in its callback group's executor thread (default); "
                             "process: rollout runs in a separate worker process")
    parser.add_argument("--rollout-cpus", type=parse_cpu_list, default=None,
                        help="CPU affinity for the rollout worker process, e.g. '2,3' or '4-7'")
    parser.add_argument("--gp-learn", action=argparse.BooleanOptionalAction, default=True,
                        help="feed wind estimates into the GP data buffers (--no-gp-learn reproduces the "
                             "original behaviour, where the buffers were never updated)")
    parser.add_argument("--tube-horizon", type=float, default=30.0,
                        help="rollout horizon in seconds (default 30). Compute time scales linearly with it")
    parser.add_argument("--gc-freeze", action=argparse.BooleanOptionalAction, default=True,
                        help="gc.freeze() after initialisation so garbage collection pauses stay short")
    parser.add_argument("--gc-no-full", action=argparse.BooleanOptionalAction, default=True,
                        help="disable automatic full (gen-2) garbage collections during flight (they stall "
                             "every thread for ~300 ms); a full collection still runs at shutdown")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                        help="print from every callback (slow; for debugging only)")
    args, unknown = parser.parse_known_args(sys.argv[1:])
    print(f"Arguments: {args}, Unknown: {unknown}")
    sim = args.sim  # already a bool
    filename = args.log_file
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    print(f"{sim=}, {filename=}, {base_path=}")
    print(f"{'SIMULATION' if sim else 'HARDWARE'}")

    options = RuntimeOptions(executor=args.executor,
                             rollout_backend=args.rollout_backend,
                             rollout_cpus=args.rollout_cpus,
                             gp_learn=args.gp_learn,
                             gc_freeze=args.gc_freeze,
                             gc_no_full=args.gc_no_full,
                             tube_horizon=args.tube_horizon,
                             verbose=args.verbose)

    rclpy.init()
    offboard_control = OffboardControl(sim, options)
    executor = make_executor(args.executor, args.threads or offboard_control.num_callback_groups)
    executor.add_node(offboard_control)
    logger = None


    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")

        try:
            offboard_control.close() # stop the rollout worker process (if any), print timing summary
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
        print(f"{BANNER}\nInitializing ROS 2 node ({type(executor).__name__})\n{BANNER}")
        logger = Logger(filename, base_path)
        executor.spin()
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
