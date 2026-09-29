import os
import sys
import inspect
import argparse
import traceback

import rclpy # Import ROS2 Python client library
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from .rta_mm_gpr_node import OffboardControl, RuntimeOptions

BANNER = "=" * 65


def default_log_path(filename):
    """<workspace>/src/data_analysis/log_files/px4_rta_mm_gpr/<filename> for a node run from a colcon workspace
    (build/ or install/), else ./flight_logs/<filename>. The directory is created."""
    parts = os.path.abspath(__file__).split(os.sep)
    marks = [i for i, p in enumerate(parts) if p in ('build', 'install')]
    if marks:
        base = os.path.join(os.sep.join(parts[:marks[0]]) or os.sep, 'src', 'data_analysis', 'log_files',
                            'px4_rta_mm_gpr')
    else:
        base = os.path.join(os.getcwd(), 'flight_logs')
    path = os.path.join(base, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


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
    parser.add_argument("--tube-early-exit", action=argparse.BooleanOptionalAction, default=True,
                        help="stop each rollout at its first certification violation + --tube-margin "
                             "(identical plan rows, ~20x less compute)")
    parser.add_argument("--tube-margin", type=float, default=1.0,
                        help="seconds of plan kept past the violation, as a fallback if a replan is late")
    parser.add_argument("--replan-lead", type=float, default=None,
                        help="start the next rollout this many seconds before the plan expires "
                             "(default: auto = 2x recent worst-case rollout latency)")
    parser.add_argument("--nr-ref-from-plan", action=argparse.BooleanOptionalAction, default=True,
                        help="NR tracker's y/z reference = RTA plan at t + T_lookahead (instead of y=0, z=-12.5+0.1t)")
    parser.add_argument("--nr-anti-windup", action=argparse.BooleanOptionalAction, default=True,
                        help="clip the NR pitch/yaw-rate channels to the CBF limits (+-0.8 rad/s)")
    parser.add_argument("--gp-feedforward", action=argparse.BooleanOptionalAction, default=True,
                        help="reference thrust cancels the GP mean disturbance (removes the steady altitude offset)")
    parser.add_argument("--min-altitude", type=float, default=0.3,
                        help="certified tubes must stay this many metres above the ground (<= 0 disables)")
    parser.add_argument("--tube-threshold", type=float, default=0.25,
                        help="certified tubes keep their position bounds (y, altitude) within this many metres of "
                             "the reference position")
    parser.add_argument("--position-uncertainty", default='auto',
                        help="delta added to the tube threshold (and the minimum initial-box half-width) in y and z: "
                             "'auto' (sim: 0, hardware: ekf2), 'ekf2' (EKF2 position standard deviation x "
                             "--uncertainty-sigmas, per axis), or a fixed value in metres")
    parser.add_argument("--uncertainty-sigmas", type=float, default=3.0,
                        help="standard deviations of EKF2's position estimate that make up delta")
    parser.add_argument("--gp", choices=['tv', 'static'], default='tv',
                        help="wind model: tv = time-varying GP (forgetting 0.25), static = time-invariant GP")
    parser.add_argument("--embedding", choices=['uw', 'u', 'none'], default='uw',
                        help="embedding system (paper Appendix A): uw = first order in u and w (68)-(69); "
                             "u = first order in u only (66)-(67); none = no first-order terms (64)-(65)")
    parser.add_argument("--backup", choices=['land', 'none'], default='land',
                        help="what to do when no certified plan exists: PX4 LAND, or keep flying the expired plan")
    parser.add_argument("--backup-grace", type=float, default=0.02,
                        help="seconds a plan may be expired before the backup engages")
    parser.add_argument("--thrust-limits-mass-scaled", action=argparse.BooleanOptionalAction, default=True,
                        help="scale the RTA thrust limits [13, 21] N (tuned at 1.75 kg) with the vehicle mass")
    parser.add_argument("--entry-ramp", action=argparse.BooleanOptionalAction, default=False,
                        help="ramp the RTA goal from the entry position to GOAL_STATE at bounded speed "
                             "(off by default: it increased upsets in the SITL ablation)")
    parser.add_argument("--ramp-speed-y", type=float, default=0.5, help="lateral goal ramp speed (m/s)")
    parser.add_argument("--ramp-speed-z", type=float, default=1.0, help="vertical goal ramp speed (m/s)")
    parser.add_argument("--gc-freeze", action=argparse.BooleanOptionalAction, default=True,
                        help="gc.freeze() after initialisation so garbage collection pauses stay short")
    parser.add_argument("--gc-no-full", action=argparse.BooleanOptionalAction, default=True,
                        help="disable automatic full (gen-2) garbage collections during flight (they stall "
                             "every thread for ~300 ms); a full collection still runs at shutdown")
    parser.add_argument("--log-autosave", type=float, default=5.0,
                        help="flush the flight log to disk every N seconds during flight (0: only at shutdown)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                        help="print from every callback (slow; for debugging only)")
    args, unknown = parser.parse_known_args(sys.argv[1:])
    print(f"Arguments: {args}, Unknown: {unknown}")
    ros_args = unknown[unknown.index('--ros-args'):] if '--ros-args' in unknown else []
    ignored = unknown[:len(unknown) - len(ros_args)]
    if ignored:  # a typo or an option from another branch would otherwise be dropped silently
        print(f"{BANNER}\nWARNING: ignoring unknown arguments {ignored}\n{BANNER}")
    sim = args.sim  # already a bool
    log_path = default_log_path(args.log_file)
    print(f"{sim=}, log: {log_path}")
    print(f"{'SIMULATION' if sim else 'HARDWARE'}")

    options = RuntimeOptions(executor=args.executor,
                             rollout_backend=args.rollout_backend,
                             rollout_cpus=args.rollout_cpus,
                             gp_learn=args.gp_learn,
                             gc_freeze=args.gc_freeze,
                             gc_no_full=args.gc_no_full,
                             tube_horizon=args.tube_horizon,
                             tube_early_exit=args.tube_early_exit,
                             tube_margin=args.tube_margin,
                             replan_lead=args.replan_lead,
                             nr_anti_windup=args.nr_anti_windup,
                             nr_ref_from_plan=args.nr_ref_from_plan,
                             entry_ramp=args.entry_ramp,
                             thrust_limits_mass_scaled=args.thrust_limits_mass_scaled,
                             gp_feedforward=args.gp_feedforward,
                             gp=args.gp, embedding=args.embedding,
                             min_altitude=args.min_altitude,
                             tube_threshold=args.tube_threshold,
                             position_uncertainty=args.position_uncertainty,
                             uncertainty_sigmas=args.uncertainty_sigmas,
                             backup=args.backup,
                             backup_grace=args.backup_grace,
                             ramp_speed_y=args.ramp_speed_y,
                             ramp_speed_z=args.ramp_speed_z,
                             verbose=args.verbose)

    rclpy.init()
    offboard_control = OffboardControl(sim, options)
    executor = make_executor(args.executor, args.threads or offboard_control.num_callback_groups)
    executor.add_node(offboard_control)


    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")

        try:
            offboard_control.close() # stop the rollout worker process (if any), print timing summary
            offboard_control.save_flight_log(log_path) # <name>.h5 + legacy <name>.csv
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
        if args.log_autosave > 0: # crash tolerance: a crash loses at most this many seconds of the flight log
            offboard_control.recorder.start_autosave(log_path, args.log_autosave)
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
