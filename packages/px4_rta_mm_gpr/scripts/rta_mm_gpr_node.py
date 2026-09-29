from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup # Callback groups decide which callbacks may run concurrently
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy) # Import ROS2 QoS policy modules
from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
    VehicleStatus, #Import PX4 ROS2-API messages for receiving vehicle state information
    VehicleOdometry, # EKF2's position variance (position-uncertainty delta of the certificate)
    RcChannels
)
from mocap_msgs.msg import FullState


import gc
import os
import time
import threading
import control
import numpy as np
import inspect
import traceback
from dataclasses import dataclass
from typing import Optional
from scipy.spatial.transform import Rotation as R

from px4_rta_mm_gpr.utilities.jax_setup import jit
from px4_rta_mm_gpr.jax_mm_rta import *
from px4_rta_mm_gpr.px4_functions import *
from px4_rta_mm_gpr.jax_nr import NR_tracker_original, dynamics
from px4_rta_mm_gpr.utilities import test_function, adjust_yaw
from px4_rta_mm_gpr.concurrency import (
    RolloutPlan, LoopStats, RolloutConfig, RolloutRequest, RolloutResult,
    ThreadRolloutBackend, ProcessRolloutBackend)

import immrax as irx
import jax.numpy as jnp
from px4_rta_mm_gpr.flight_log import FlightRecorder
from px4_rta_mm_gpr.control_kernels import ControlKernels

BANNER = '\n' + "==" * 30 + '\n'


@dataclass(frozen=True)
class RuntimeOptions:
    """How the node is scheduled. Set from the command line in scripts/px4_rta_mm_gpr.py."""
    executor: str = 'multi'              # 'multi' | 'single' | 'events'
    rollout_backend: str = 'thread'      # 'thread' | 'process'
    rollout_cpus: Optional[frozenset] = None  # CPU affinity for the rollout worker process
    gp_learn: bool = True                # feed wind estimates into the GP data buffers
    tube_horizon: float = 30.0           # (s) rollout horizon; the scan is causal, so a shorter horizon gives the
                                         # same plan whenever the safety violation happens inside it
    gc_freeze: bool = True               # move everything allocated during init out of the GC's reach
    gc_no_full: bool = True              # no automatic full (generation-2) collections during flight
    tube_early_exit: bool = True         # stop each rollout at its first certification violation + margin
    tube_margin: float = 1.0             # (s) rows kept past the violation (fallback + NR look-ahead)
    replan_lead: Optional[float] = None  # (s) start the next rollout this long before the plan expires; None = auto
    nr_ref_from_plan: bool = True        # NR's y/z reference = the RTA plan at t + T_lookahead (no conflicting goals)
    nr_anti_windup: bool = True          # clip the NR pitch/yaw-rate channels to the CBF limits (+-0.8 rad/s)
    gp_feedforward: bool = True          # reference thrust cancels the GP mean disturbance (no altitude offset)
    min_altitude: float = 0.3            # (m) certified tubes must stay this far above the ground (<= 0: no floor)
    tube_threshold: float = 0.25         # (m) certified tubes: position bounds (py, pz) within this (+ delta) of the reference
    position_uncertainty: str = 'auto'   # delta: 'auto' (sim 0, hardware 'ekf2'), 'ekf2', or a fixed value in metres
    uncertainty_sigmas: float = 3.0      # delta = this many EKF2 standard deviations (per axis)
    backup: str = 'land'                 # 'land': PX4 LAND when no certified plan exists; 'none': keep flying
    backup_grace: float = 0.02           # (s) how long a plan may be expired before the backup engages
    thrust_limits_mass_scaled: bool = True # RTA thrust limits as fractions of hover thrust (hardware ratios)
    entry_ramp: bool = False             # move the RTA goal from the entry position to GOAL_STATE at bounded speed
    ramp_speed_y: float = 0.5            # (m/s) lateral speed of the ramped goal
    ramp_speed_z: float = 1.0            # (m/s) vertical speed of the ramped goal
    verbose: bool = False                # per-callback debug printing (slow: ~100s of prints/s)


@dataclass(frozen=True)
class VehicleState:
    """One odometry sample. Built completely, then published with a single assignment, so
    readers in other threads never see a mix of two samples (e.g. new position, old velocity)."""
    stamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    roll: float
    pitch: float
    yaw: float
    nr_state_vector: np.ndarray                 # (x, y, z, vx, vy, vz, roll, pitch, yaw)
    rta_mm_gpr_state_vector_planar: np.ndarray  # (py, pz, h, v, theta)

class WindEKF:
    def __init__(self, mass, Q=None, R=None):
        """
        EKF/KF for estimating wind disturbance forces in y, z [N].

        State: x = [w_y, w_z]^T
        Measurement: z = [ay_meas, az_meas]^T
          z = a_model + (1/m) * x + noise
        """
        self.m = mass

        # State: start with zero wind
        self.x = np.zeros((2, 1))  # [w_y; w_z]
        self.P = np.eye(2) * 1.0   # initial covariance, tune as needed

        # Process noise covariance (how fast wind can change)
        if Q is None:
            # e.g. 0.1 N^2 per step on each axis
            self.Q = np.eye(2) * 0.1
        else:
            self.Q = np.array(Q, dtype=float)

        # Measurement noise covariance (accel noise + model error)
        if R is None:
            # e.g. (0.1 m/s^2)^2 on each axis
            sigma_a = 0.1
            self.R = np.eye(2) * (sigma_a ** 2)
        else:
            self.R = np.array(R, dtype=float)

        # Constant matrices
        self.F = np.eye(2)
        self.H = (1.0 / self.m) * np.eye(2)  # maps [w_y, w_z] -> accel contribution

    def reset(self):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0

    def predict(self, dt=None):
        """
        Time update (no input; wind is modeled as random walk).

        dt=None: one step of the original 10 Hz filter (process noise Q per step).
        dt given: Q is interpreted as noise per 0.1 s and scaled by dt / 0.1, so the filter behaves the
        same at any sample rate (the node now updates at the odometry rate, 40 Hz).
        """
        Q = self.Q if dt is None else self.Q * (dt / 0.1)
        # x_{k+1|k} = F x_{k|k}
        self.x = self.F @ self.x
        # P_{k+1|k} = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, ay_meas, az_meas, ay_model, az_model):
        """
        Measurement update.

        ay_meas, az_meas: measured accelerations [m/s^2]
        ay_model, az_model: model-predicted accel from your dynamics [m/s^2]

        Returns:
            wy, wz: updated estimates of wind forces [N]
        """
        # Measurement vector
        z = np.array([[ay_meas],
                      [az_meas]])

        # Model accel vector (acts as known offset)
        a_model = np.array([[ay_model],
                            [az_model]])

        # Predicted measurement: z_hat = a_model + H x_{k+1|k}
        z_hat = a_model + self.H @ self.x

        # Innovation
        y = z - z_hat

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        # Return wind forces
        wy, wz = self.x.flatten()
        return wy, wz


class OffboardControl(Node):
    def __init__(self, sim: bool, options: RuntimeOptions = RuntimeOptions()) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        # Initialize essential variables
        self.sim: bool = sim
        self.options = options
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration
        self.debug = print if options.verbose else (lambda *args, **kwargs: None) # hot-path printing is opt-in

        if self.sim:
            print("Using simulator constants and functions")
            from px4_rta_mm_gpr.utilities import sim_utilities # Import simulation constants
            self.MASS = sim_utilities.MASS
            self.get_throttle_command_from_force = sim_utilities.get_throttle_command_from_force
        else:
            print("Using hardware constants and functions")
            from px4_rta_mm_gpr.utilities import hardware_utilities # Import hardware constants
            self.MASS = hardware_utilities.MASS
            self.get_throttle_command_from_force = hardware_utilities.get_throttle_command_from_force

        self.wind_ekf = WindEKF(mass=self.MASS)
        self.USE_EKF = True
        self.wy, self.wz = 0.0, 0.0 # latest wind force estimates (written by the state thread)
        self.wind_ekf_active = False    # the EKF only runs while OUR body-rate commands fly the vehicle
        self.wind_ekf_last_t = None


        self.tube_pos_indices = [0, 1, 5, 6]  # Indices for x, y, z, yaw in the rollout reference trajectory

        self.tube_start = 9
        self.tube_extent = 23
        self.tube_skip = 2



##########################################################################################
        # Callback groups: callbacks in the SAME mutually-exclusive group never run at the same time;
        # callbacks in DIFFERENT groups may run in parallel on a MultiThreadedExecutor.
        # (With a single-threaded executor the groups are harmless: everything is serialized anyway.)
        self.px4_io_group = MutuallyExclusiveCallbackGroup()     # offboard heartbeat, vehicle status, RC
        self.state_group = MutuallyExclusiveCallbackGroup()      # odometry
        self.control_group = MutuallyExclusiveCallbackGroup()    # 100 Hz control law
        self.rollout_group = MutuallyExclusiveCallbackGroup()    # reachable tube + reference rollouts
        self.estimation_group = MutuallyExclusiveCallbackGroup() # wind EKF/GP data + LQR gain updates
        self.num_callback_groups = 5

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)

        # Create subscribers
        self.state: Optional[VehicleState] = None # latest odometry sample (None until the first message)
        self.vehicle_odometry_subscriber = self.create_subscription(
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_odometry_subscriber_callback, qos_profile,
            callback_group=self.state_group)


        # Position-uncertainty delta for the certificate: EKF2's position standard deviation (x uncertainty_sigmas) on
        # hardware, where the estimate fuses mocap; 0 by default in SITL, whose EKF2 fuses simulated GPS (its sigma
        # describes GPS, not the mocap setup this certificate is meant for; see docs/04).
        mode = options.position_uncertainty
        self.delta_mode = ('0' if self.sim else 'ekf2') if mode == 'auto' else mode
        self.position_sigma: Optional[np.ndarray] = None   # (sigma_y, sigma_z) from EKF2, one assignment per message
        if self.delta_mode == 'ekf2':
            self.vehicle_odometry_subscriber = self.create_subscription(
                VehicleOdometry, '/fmu/out/vehicle_odometry', self.ekf2_odometry_callback, qos_profile,
                callback_group=self.state_group)
        else:
            float(self.delta_mode)  # a fixed value in metres (raises on a typo)
        print(f"Certificate: position bounds within {options.tube_threshold} m + delta of the reference; "
              f"delta = {self.delta_mode if self.delta_mode != 'ekf2' else f'{options.uncertainty_sigmas} x EKF2 sigma'}")

        self.in_offboard_mode: bool = False
        self.armed: bool = False
        self.in_land_mode: bool = False
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile,
            callback_group=self.px4_io_group)

        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        print(f"RC switch mode: {'On' if self.offboard_mode_rc_switch_on else 'Off'}")
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile,
            callback_group=self.px4_io_group)

        # MoCap related variables
        self.mocap_initialized: bool = False
        self.full_rotations: int = 0
        self.max_yaw_stray = 5 * np.pi / 180

        # PX4 variables
        self.offboard_heartbeat_counter: int = 0
        self.vehicle_status = VehicleStatus()

        # Callback function time constants
        self.heartbeat_period: float = 0.1 # (s) We want 10Hz for offboard heartbeat signal
        self.control_period: float = 0.01 # (s) We want 100Hz for direct control algorithm
        self.wind_estimate_period: float = 0.1 # (s) We want 10Hz for wind estimation update
        self.lqr_period: float = 0.05 # (s) how often the LQR re-linearization condition is checked

        self.OBS_DYN = jnp.array([
                                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # Timing diagnostics, printed at shutdown
        self.loop_stats = {
            'control': LoopStats('control', self.control_period),
            'heartbeat': LoopStats('heartbeat', self.heartbeat_period),
            'wind': LoopStats('wind', self.wind_estimate_period),
        }
        self.rollout_compute_times: list = []
        self.rollout_latencies: list = []
        self.gc_pauses = {0: [], 1: [], 2: []} # (s) stop-the-world garbage collection pauses per generation
        self._gc_start = None

        # Shared state between threads. Each of these is only ever REPLACED (one reference
        # assignment, atomic in CPython), never mutated in place.
        self.plan: Optional[RolloutPlan] = None          # written by rollout thread, read by control
        self.gains: Optional[tuple] = None               # (K_feedback, K_reference); written by estimation thread
        self.plan_lock = threading.Lock()                # guards plan installation bookkeeping
        self.plan_seq: int = 0
        self.warmup_rollout_requested = threading.Event() # wind thread asks the rollout thread for a warm-up rollout
        self.collection_time: float = 0.0  # Time at which the collection starts (rollout thread only)
        self.certification_gaps: int = 0   # control ticks that ran on an expired plan (should stay 0)
        self.backup_engaged: Optional[str] = None # reason, once the backup (LAND) has been commanded
        self.uncertified_since: Optional[float] = None # start of the current stretch without a certified plan
        self.ramp_origin = None            # (t, y, z) where the RTA goal ramp starts

        self.init_jit_compile_nr_rta() # Initialize JIT compilation for NR tracker and RTA pipeline

        self.last_lqr_update_time: float = 0.0  # Initialize last LQR update time
        self.first_LQR: bool = True  # Flag to indicate if this is the first LQR update

        # Time variables
        self.T0 = time.time() # (s) initial time of program (reset after JIT compilation)
        self.begin_actuator_control = 15 # (s) time after which we start sending actuator control commands
        self.land_time = self.begin_actuator_control + 20 # (s) time after which we start sending landing commands
        if self.sim:
            self.max_height = -12.5
            self.max_y = 4.0
        else:
            self.max_height = -3.75
            self.max_y = -2.5

        # Flight recorder: preallocated buffers, written to HDF5 (+ legacy CSV) at shutdown
        self.recorder = FlightRecorder(metadata=dict(
            platform='sim' if self.sim else 'hardware', mass=float(self.MASS),
            executor=options.executor, rollout_backend=options.rollout_backend,
            gp_learn=options.gp_learn, gc_freeze=options.gc_freeze, gc_no_full=options.gc_no_full,
            tube_horizon=self.tube_horizon, tube_timestep=self.tube_timestep,
            collection_threshold=self.collection_threshold, control_period=self.control_period,
            begin_actuator_control=float(self.begin_actuator_control), land_time=float(self.land_time),
            max_height=self.max_height, max_y=self.max_y,
            goal_state=np.asarray(self.GOAL_STATE), x_pert=np.asarray(self.x_pert),
            Q_planar=np.diag(np.asarray(self.Q_planar)), R_planar=np.diag(np.asarray(self.R_planar)),
            Q_ref_planar=np.diag(np.asarray(self.Q_ref_planar)), R_ref_planar=np.diag(np.asarray(self.R_ref_planar)),
            ulim_lower=np.asarray(self.rollout_config.ulim_lower), ulim_upper=np.asarray(self.rollout_config.ulim_upper),
            T_lookahead=self.T_LOOKAHEAD, wind_ekf_Q=np.diag(self.wind_ekf.Q), wind_ekf_R=np.diag(self.wind_ekf.R),
            tube_early_exit=options.tube_early_exit, tube_margin=options.tube_margin,
            replan_lead='auto' if options.replan_lead is None else options.replan_lead,
            nr_anti_windup=options.nr_anti_windup, nr_ref_from_plan=options.nr_ref_from_plan,
            entry_ramp=options.entry_ramp,
            thrust_limits_mass_scaled=options.thrust_limits_mass_scaled,
            gp_feedforward=options.gp_feedforward, min_altitude=options.min_altitude,
            position_uncertainty=self.delta_mode, uncertainty_sigmas=options.uncertainty_sigmas,
            backup=options.backup, backup_grace=options.backup_grace,
            ramp_speed_y=options.ramp_speed_y, ramp_speed_z=options.ramp_speed_z))

        if self.options.gc_freeze:
            # Everything allocated so far (JAX/XLA caches, compiled functions, immrax objects, ROS entities)
            # lives for the whole flight. gc.freeze() moves it to a permanent generation that the collector
            # never scans again, so each full collection only walks objects created during flight.
            gc.collect()
            gc.freeze()
            print(f"gc.freeze(): {gc.get_freeze_count()} objects excluded from garbage collection")
        if self.options.gc_no_full:
            # A full collection walks EVERY tracked object while holding the GIL (measured: ~300 ms, i.e. 30
            # missed control ticks). Young generations are still collected (~1 ms); gen-2 garbage is left for
            # the explicit gc.collect() at shutdown. For a flight of a few minutes the extra memory is small.
            g0, g1, _ = gc.get_threshold()
            gc.set_threshold(g0, g1, 1_000_000_000)

        gc.callbacks.append(self._gc_callback) # registered after init: only in-flight collections are reported

        # Timers for my callback functions (created last so no callback runs before init finishes)
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback,
                                                callback_group=self.px4_io_group) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback,
                                               callback_group=self.control_group) #My control algorithm needs to execute at >= 100Hz
        self.rollout_timer = self.create_timer(self.control_period,
                                               self.rollout_callback,
                                               callback_group=self.rollout_group) #Checks at 100Hz whether a new rollout is needed
        self.wind_estimator = self.create_timer(self.wind_estimate_period,
                                                self.wind_estimator_callback,
                                                callback_group=self.estimation_group)
        self.lqr_timer = self.create_timer(self.lqr_period,
                                           self.lqr_update_callback,
                                           callback_group=self.estimation_group)

    def _gc_callback(self, phase, info):
        # Runs in whichever thread triggered the collection, with the GIL held: every other Python thread
        # (including the control loop) is frozen for this whole duration.
        if phase == 'start':
            self._gc_start = time.perf_counter()
        elif self._gc_start is not None:
            self.gc_pauses[info['generation']].append(time.perf_counter() - self._gc_start)
            self._gc_start = None

    def now(self) -> float:
        """Seconds since the node's T0. Each callback computes its own time instead of sharing
        one ``self.time_from_start`` attribute that several threads would race to overwrite."""
        return time.time() - self.T0

    def init_jit_compile_nr_rta(self):
        """
        Set up the NR tracker / RTA parameters and compile every in-flight computation AHEAD OF TIME.

        Each kernel is compiled once for the exact argument shapes/dtypes used in flight
        (see px4_rta_mm_gpr/control_kernels.py). A compiled executable cannot recompile, so nothing
        can stall a flight with JIT compilation; a signature mismatch would raise immediately instead.
        Compiled executables are also cached on disk, so later runs start much faster.
        """
        print(f"{BANNER}Compiling (ahead of time) the NR tracker, RTA feedback, wind model, LQR linearization and rollout.")

        # Initialize NR algorithm parameters
        # last input to the controller: always a float64 NumPy array (canonical type for the compiled kernels)
        self.last_input = np.array([self.MASS * self.GRAVITY, 0.01, 0.01, 0.01])
        self.hover_input_planar = np.array([self.MASS * self.GRAVITY, 0.]) # hover input to the controller
        self.T_LOOKAHEAD: float = 0.8 # (s) lookahead time for the controller in seconds
        self.T_LOOKAHEAD_PRED_STEP: float = 0.1 # (s) we do state prediction for T_LOOKAHEAD seconds ahead in intervals of T_LOOKAHEAD_PRED_STEP seconds
        self.INTEGRATION_TIME: float = self.control_period # integration time constant for the controller in seconds

        # Initialize rta_mm_gpr variables
        # Goal 1.0 m above the take-off point (was 0.6 m, which relied on the removed sim "+0.5 m" z adjustment and sat
        # only 0.3 m above the ground floor; see docs/04).
        self.GOAL_STATE = jnp.array([0., -1.0, 0., 0., 0.])
        x0 = jnp.array([0.1, 0.1, 0.1, 0.02, 0.03])  # representative planar state (only for start-up printing)

        np.random.seed(0)

        initialization_values_GP = jnp.array([[-2, np.random.rand()], #make the second column all zeros
                                        [0, np.random.rand()],
                                        [2, np.random.rand()],
                                        [4, np.random.rand()],
                                        [6, np.random.rand()],
                                        [8, np.random.rand()],
                                        [10, np.random.rand()],
                                        [12, np.random.rand()]]) # at heights of y in the first column, disturbance to the values in the second column
        # add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
        initialization_GP = TVGPR(jnp.hstack((jnp.zeros((initialization_values_GP.shape[0], 1)), initialization_values_GP)),
                                            sigma_f = 5.0,
                                            l=2.0,
                                            sigma_n = 0.01,
                                            epsilon=0.1,
                                            discrete=False
                                            )


        self.x_pert = 5e-4 * jnp.array([1., 1., 1., 1., 1.]) #


        # Initialize rollout parameters
        self.n_obs = 9
        obs = np.tile(np.array([[0, float(x0[1]), float(get_gp_mean(initialization_GP, 0.0, x0)[0])]]),(self.n_obs,1))
        self.obs = obs

        self.wind_count = 0
        self.gp_obs_count = 0
        # GP data buffers: rows of (time, input, wind force). NumPy + copy-on-write (see wind callback).
        self.gz_wind_obs_in_y = obs.copy()   # y-wind observations at various times & altitudes z
        self.gy_wind_obs_in_z = obs.copy()   # z-wind observations at various times & lateral positions y

        self.quad_sys_planar = PlanarMultirotorTransformed(mass=self.MASS)
        # RTA input limits (thrust N, roll rate rad/s). [13, 21] N was tuned on the 1.75 kg vehicle: hover 17.2 N,
        # i.e. 0.757..1.224 x hover thrust. Used unchanged on the 2.0 kg sim model it leaves only +1.4 N (0.7 m/s^2)
        # to brake a descent, so the rollouts plan (and the vehicle flies) descents it cannot stop. Scaling the limits
        # with hover thrust keeps the hardware numbers exactly and gives the sim the same authority.
        HW_MASS, HW_THRUST_LIMITS = 1.75, (13.0, 21.0)
        if self.options.thrust_limits_mass_scaled:
            scale = self.MASS / HW_MASS
            thrust_lo, thrust_hi = HW_THRUST_LIMITS[0] * scale, HW_THRUST_LIMITS[1] * scale
        else:
            thrust_lo, thrust_hi = HW_THRUST_LIMITS
        self.ulim_lower, self.ulim_upper = (thrust_lo, -1.0), (thrust_hi, 1.0)
        self.ulim_planar = irx.interval(list(self.ulim_lower), list(self.ulim_upper)) # type: ignore
        print(f"RTA input limits: thrust [{thrust_lo:.2f}, {thrust_hi:.2f}] N (hover {self.MASS * self.GRAVITY:.2f} N), roll rate [-1, 1] rad/s")
        self.Q_planar = jnp.array([20, 5, 10, 1, 1]) * jnp.eye(self.quad_sys_planar.xlen) # weights that prioritize overall tracking of the reference (defined below)
        self.R_planar = jnp.array([50, 20]) * jnp.eye(2)


        #(py,pz,h,v,theta)
        # Reference weights (py, pz, h, v, theta). Was [40, 5, 40, 5, 3]: vertically under-damped -- from 2.8 m sinking at
        # 1.5 m/s the reference overshot a 0.6 m goal down to ~0 m. pz 10 / v 40 bottoms out at ~0.66 m for a 1.0 m goal.
        self.Q_ref_planar =jnp.array([40, 10, 40, 40, 3]) * jnp.eye(self.quad_sys_planar.xlen)
        self.R_ref_planar = jnp.array([50, 20]) * jnp.eye(2)


        self.tube_timestep = 0.01  # Time step
        self.tube_horizon = self.options.tube_horizon   # Reachable tube horizon (default 30.0 s); an upper bound with early exit
        self.collection_threshold = self.options.tube_threshold # tube/reference POSITION deviation (m, py and pz) that ends the safety horizon

        # Everything static about the rollout, as plain values (picklable for the worker process)
        self.rollout_config = RolloutConfig(
            mass=float(self.MASS), horizon=self.tube_horizon, timestep=self.tube_timestep,
            ulim_lower=self.ulim_lower, ulim_upper=self.ulim_upper,
            goal_state=tuple(float(v) for v in self.GOAL_STATE),
            x_pert=tuple(float(v) for v in self.x_pert),
            collection_threshold=self.collection_threshold,
            n_obs=self.n_obs,
            early_exit=self.options.tube_early_exit,
            margin_steps=int(round(self.options.tube_margin / self.tube_timestep)),
            min_altitude=self.options.min_altitude if self.options.min_altitude > 0 else None,
            gp_feedforward=self.options.gp_feedforward)

        # The process backend starts FIRST so the worker's rollout compilation overlaps with ours.
        rollout_backend = None
        t_compile = time.perf_counter()
        if self.options.rollout_backend == 'process':
            print("Starting rollout worker process (it compiles the rollout in parallel)...")
            rollout_backend = ProcessRolloutBackend(self.rollout_config, self._warmup_request(x0),
                                                    cpu_affinity=self.options.rollout_cpus)

        self.kernels = ControlKernels(self.MASS, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME,
                                      self.ulim_lower, self.ulim_upper, self.quad_sys_planar,
                                      anti_windup=self.options.nr_anti_windup)
        print(f"Control kernels compiled in {self.kernels.compile_time:.2f} s "
              f"(NR anti-windup: {self.options.nr_anti_windup})")
        K_feedback, K_reference = self.compute_lqr_gains(x0, self.hover_input_planar)

        if rollout_backend is None:
            rollout_backend = ThreadRolloutBackend(self.rollout_config)
            print(f"Rollout compiled in {rollout_backend.engine.compile_time:.2f} s; "
                  f"one rollout takes {rollout_backend.warmup(self._warmup_request(x0, K_feedback, K_reference)) * 1e3:.1f} ms")
        else:
            warm = rollout_backend.wait_ready(timeout=600.0)
            print(f"Rollout worker ready (compile + first rollout: {warm:.2f} s)")
        self.rollout_backend = rollout_backend
        print(f"All compilation done in {time.perf_counter() - t_compile:.2f} s "
              f"(early-exit rollouts: {self.options.tube_early_exit}, margin {self.options.tube_margin:.2f} s)")

        print(f"{K_feedback=}\n{K_reference=}")
        print(f"Executor: {self.options.executor} | rollout backend: {self.rollout_backend.name} | "
              f"GP learning: {self.options.gp_learn} | verbose: {self.options.verbose}")

        # Pause for 3 seconds to give myself time to read the print statements above
        print(f"\nPausing for 3 seconds to read the compilation times above.\nContinuing...\n")
        time.sleep(3)

    def _warmup_request(self, x0, K_feedback=None, K_reference=None) -> RolloutRequest:
        """A representative request used only to trigger JIT compilation of the rollout."""
        K_fb = np.zeros((2, 5)) if K_feedback is None else np.asarray(K_feedback)
        K_ref = np.zeros((2, 5)) if K_reference is None else np.asarray(K_reference)
        return RolloutRequest(t_start=0.0, state=np.asarray(x0), K_feedback=K_fb, K_reference=K_ref,
                              obs_wy=self.gz_wind_obs_in_y, obs_wz=self.gy_wind_obs_in_z, warmup=True)

    def close(self) -> None:
        """Stop helper processes and print the timing summary. Called on shutdown."""
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
        backend = getattr(self, 'rollout_backend', None)
        if backend is not None:
            backend.close()
        print(self.timing_summary())
        print(self.heap_summary())

    def heap_summary(self, top: int = 6) -> str:
        """Which object types the garbage collector has to scan (i.e. what makes full collections slow)."""
        from collections import Counter
        counts = Counter(type(o).__name__ for o in gc.get_objects())
        return (f"gc-tracked objects: {sum(counts.values())} (+{gc.get_freeze_count()} frozen); top: "
                + ', '.join(f"{name}={n}" for name, n in counts.most_common(top)))

    def save_flight_log(self, path: str) -> str:
        """Write the HDF5 flight log (and the legacy ROS2Logger CSV next to it)."""
        timing = {f'{name}_period': st.periods() for name, st in self.loop_stats.items()}
        timing.update({f'{name}_exec': st.exec_times() for name, st in self.loop_stats.items()})
        timing.update(rollout_compute=self.rollout_compute_times, rollout_latency=self.rollout_latencies,
                      **{f'gc_gen{g}_pause': v for g, v in self.gc_pauses.items()})
        h5_path = self.recorder.save(path, timing=timing)
        from px4_rta_mm_gpr.flight_log import FlightLog
        with FlightLog(h5_path) as log:
            csv_path = log.to_legacy_csv(os.path.splitext(h5_path)[0] + '.csv')
        print(f"[flight_log] {h5_path} ({os.path.getsize(h5_path) / 1e6:.1f} MB, {len(self.recorder.ticks)} ticks, "
              f"{self.recorder.n_plans} plans)\n[flight_log] legacy CSV: {csv_path}")
        return h5_path

    def timing_summary(self) -> str:
        lines = [f"{BANNER}Timing summary (executor={self.options.executor}, "
                 f"rollout backend={self.options.rollout_backend})"]
        lines += [s.summary() for s in self.loop_stats.values()]
        for name, values in (('rollout compute', self.rollout_compute_times),
                             ('rollout latency', self.rollout_latencies)):
            v = np.asarray(values)
            if v.size:
                lines.append(f"{name:>16}: n={v.size}  mean={1e3 * v.mean():.1f} ms  "
                             f"p50={1e3 * np.median(v):.1f} ms  max={1e3 * v.max():.1f} ms")
        for gen, pauses in self.gc_pauses.items():
            v = np.asarray(pauses)
            if v.size:
                lines.append(f"{'gc gen ' + str(gen):>16}: n={v.size}  total={v.sum():.3f} s  "
                             f"p50={1e3 * np.median(v):.2f} ms  max={1e3 * v.max():.1f} ms")
        ticks = len(self.recorder.ticks)
        if ticks:
            lines.append(f"{'cert. gaps':>16}: {self.certification_gaps} of {ticks} RTA ticks ran on an expired plan "
                         f"({100.0 * self.certification_gaps / ticks:.2f}%)")
        return '\n'.join(lines) + BANNER


    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        self.debug(f"{BANNER}In RC Channel Callback")
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False

    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        self.debug(f"{BANNER}Received odometry data: {msg=}")

        x = msg.position[0]
        y = msg.position[1]
        # PX4's local z (NED, 0 at the take-off point). The earlier sim-only "+0.5 m when |z| < 1.2 m" adjustment
        # is gone: the ground is now handled explicitly (min_altitude in the certificate + LAND backup).
        z = msg.position[2]

        vx, vy, vz = msg.velocity
        ax, ay, az = msg.acceleration

        roll, pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        yaw = adjust_yaw(self, yaw)  # Adjust yaw to account for full rotations

        # Build the whole sample, then publish it with ONE assignment (atomic for readers)
        self.state = VehicleState(
            stamp=self.now(), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, ax=ax, ay=ay, az=az,
            roll=roll, pitch=pitch, yaw=yaw,
            nr_state_vector=np.array([x, y, z, vx, vy, vz, roll, pitch, yaw]),
            rta_mm_gpr_state_vector_planar=np.array([y, z, vy, vz, roll]), # px, py, h, v, theta = x
        )
        self.debug(f"in odom, flat output: {[x, y, z, yaw]}")
        self.update_wind_estimate(self.state)

    def ekf2_odometry_callback(self, msg) -> None:
        """EKF2's position variance (NED): keep the standard deviations of y and z (one assignment)."""
        var = np.asarray(msg.position_variance, dtype=float)
        self.position_sigma = np.sqrt(np.maximum(var[1:3], 0.0))

    def position_delta(self) -> np.ndarray:
        """delta (m, py and pz) for the next rollout: the initial box is at least delta wide, the threshold + delta."""
        if self.delta_mode != 'ekf2':
            return np.full(2, float(self.delta_mode))
        sigma = self.position_sigma
        if sigma is None:   # no EKF2 variance yet: nothing can be certified (safe; it arrives long before the RTA phase)
            return np.full(2, np.inf)
        return self.options.uncertainty_sigmas * sigma

    def our_commands_fly(self, t: float) -> bool:
        """True while the vehicle is flown by this node's body-rate commands (RTA phase, offboard, plan ready)."""
        return (self.in_offboard_mode and self.begin_actuator_control <= t < self.land_time
                and self.plan is not None and self.gains is not None)

    def update_wind_estimate(self, s: VehicleState) -> None:
        """Wind EKF measurement update at the odometry rate (40 Hz), in the state thread.

        Only runs while our commands fly: before the RTA phase PX4's position controller flies the vehicle, and
        the model's input (self.last_input) is not what the vehicle actually receives, so the residual would be
        model error, not wind. The filter is reset when the RTA phase begins.
        """
        t = s.stamp
        if not self.our_commands_fly(t):
            self.wind_ekf_active = False
            return
        if not self.wind_ekf_active:
            self.wind_ekf.reset()
            self.wind_ekf_active, self.wind_ekf_last_t = True, t
        dt = min(max(t - self.wind_ekf_last_t, 0.0), 0.1)
        self.wind_ekf_last_t = t

        ay_hat, az_hat = self.kernels.wind_model(s.nr_state_vector, self.last_input) # AOT-compiled
        if self.USE_EKF:
            self.wind_ekf.predict(dt)
            wy, wz = self.wind_ekf.update(ay_meas=s.ay, az_meas=s.az, ay_model=ay_hat, az_model=az_hat)
        else: # raw residual (no filtering)
            wy, wz = self.MASS * (s.ay - ay_hat), self.MASS * (s.az - az_hat)
        self.wy, self.wz = float(wy), float(wz)
        self.recorder.wind_sample(t, self.wy, self.wz, s.ay, s.az, ay_hat, az_hat, s.z, s.y)


    def lqr_update_callback(self) -> None:
        """Re-linearize the planar model and recompute both LQR gains when needed.

        Previously this ran inline in the odometry callback (before the first RTA update) and in the
        control callback (every 1.8 s, or EVERY tick while |yaw| > max_yaw_stray). It now runs in the
        estimation callback group so its ~4 ms never delays a control tick.
        """
        s = self.state
        if s is None:
            return
        t = self.now()
        in_rta_window = self.begin_actuator_control <= t < self.land_time

        if self.first_LQR and not in_rta_window:
            # Before the RTA phase: linearize about hover at the current state (as the odometry callback did)
            self.gains = self.compute_lqr_gains(s.rta_mm_gpr_state_vector_planar, self.hover_input_planar)
            self.last_lqr_update_time = t
            self.recorder.gain_update(t, True, *self.gains)
        elif in_rta_window and ((t - self.last_lqr_update_time) >= 1.8 or abs(self.yaw_error(t, s.yaw)) > self.max_yaw_stray):
            self.update_lqr_feedback(self.quad_sys_planar, s.rta_mm_gpr_state_vector_planar, self.last_input, t)

    def yaw_error(self, t: float, yaw: float) -> float:
        """Yaw tracking error wrapped to [-pi, pi].

        The re-linearisation trigger used to test |yaw| > max_yaw_stray, which is permanently true whenever
        the desired yaw is not 0 (or after a full turn, since adjust_yaw() unwraps yaw to an absolute angle).
        """
        yaw_des = float(self.get_ref(t)[3])
        return float(np.arctan2(np.sin(yaw - yaw_des), np.cos(yaw - yaw_des)))

    def compute_lqr_gains(self, state, input) -> tuple:
        A, B = self.kernels.linearize(state, np.asarray(input)[0:2]) # AOT-compiled Jacobians of the planar model
        K_feedback, _, _ = control.lqr(A, B, np.asarray(self.Q_planar), np.asarray(self.R_planar))
        K_reference, _, _ = control.lqr(A, B, np.asarray(self.Q_ref_planar), np.asarray(self.R_ref_planar))
        return (np.asarray(K_feedback), np.asarray(K_reference))


    def wind_estimator_callback(self):
        """10 Hz: push the latest wind estimate into the GP data buffers (ring of n_obs rows).

        The EKF itself runs at the odometry rate (update_wind_estimate); this timer only samples it.
        """
        with self.loop_stats['wind'].measure():
            if not self.in_offboard_mode:
                return
            s = self.state
            if s is None: # offboard can engage before the first odometry message arrives
                return
            wind_estimate_time = self.now()
            self.wind_count += 1

            if self.options.gp_learn and self.wind_ekf_active and self.our_commands_fly(wind_estimate_time):
                wind_idx = self.gp_obs_count % self.n_obs
                self.gp_obs_count += 1
                # BUG FIX: the original `self.gz_wind_obs_in_y.at[i, :].set(...)` built a new JAX array and
                # threw it away (JAX arrays are immutable), so the GP never saw a single wind estimate.
                # Copy-on-write: modify a copy, then swap the reference so the rollout thread, which may be
                # reading the old buffer right now, always sees a complete buffer.
                new_obs_y = self.gz_wind_obs_in_y.copy()
                new_obs_y[wind_idx, :] = (wind_estimate_time, s.z, self.wy) # y-wind as a function of altitude
                self.gz_wind_obs_in_y = new_obs_y

                new_obs_z = self.gy_wind_obs_in_z.copy()
                new_obs_z[wind_idx, :] = (wind_estimate_time, s.y, self.wz) # z-wind as a function of lateral position
                self.gy_wind_obs_in_z = new_obs_z

            if self.wind_count % 50 == 0 and wind_estimate_time < self.begin_actuator_control:
                # Ask the rollout thread for a warm-up rollout instead of blocking this thread with it
                self.debug("Requesting rollout after wind update")
                self.warmup_rollout_requested.set()


    def rollout_callback(self):
        """Rollout timer: installs finished rollouts and starts a new one when the plan expires.

        Thread backend: ``submit`` blocks THIS callback group's thread for the whole rollout, which
        is fine on a MultiThreadedExecutor (control keeps running in its own thread).
        Process backend: ``submit``/``poll`` return immediately; the rollout runs in another process.
        """
        try:
            result = self.rollout_backend.poll() # a result finished since the last tick (process backend)
            if result is not None:
                self.install_plan(result)
            if self.rollout_backend.busy:
                return

            current_time = self.now()
            warmup = self.warmup_rollout_requested.is_set()
            in_window = self.begin_actuator_control - 1.0 <= current_time <= self.land_time
            # Pre-emptive replanning: the next plan must be INSTALLED before the current one's safety horizon ends,
            # so the rollout starts `lead` seconds early (lead ~ 2x the recent worst-case rollout latency).
            safety_due = in_window and current_time >= self.collection_time - self.replan_lead()
            if not (warmup or safety_due):
                return # You're safe!
            s, gains = self.state, self.gains
            if s is None or gains is None:
                return
            if safety_due:
                warmup = False # a real (safety) rollout supersedes a pending warm-up request
            self.warmup_rollout_requested.clear()

            self.debug(f"{BANNER}Unsafe region begins now at {current_time:.2f}. Recomputing reachable tube and reference trajectory.")
            request = RolloutRequest(t_start=current_time,
                                     state=s.rta_mm_gpr_state_vector_planar,
                                     K_feedback=gains[0], K_reference=gains[1],
                                     obs_wy=self.gz_wind_obs_in_y, obs_wz=self.gy_wind_obs_in_z,
                                     warmup=warmup, submitted_at=time.time(),
                                     goal=self.rollout_goal(current_time, s, warmup),
                                     delta=self.position_delta())
            self.rollout_backend.submit(request)

            result = self.rollout_backend.poll() # thread backend: already done
            if result is not None:
                self.install_plan(result)

        except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "<unknown>"
            print(f"\nError in {__name__}:{func_name}: {e}")
            traceback.print_exc()
            raise

    def replan_lead(self) -> float:
        """How early (s) to start the next rollout before the current plan's safety horizon ends."""
        if self.options.replan_lead is not None:
            return self.options.replan_lead
        recent = self.rollout_latencies[-50:]
        worst = max(recent) if recent else 0.05
        return max(2.0 * worst, 2.0 * self.control_period) # 2x margin + the rollout timer's own granularity

    def rollout_goal(self, t: float, s: VehicleState, warmup: bool) -> np.ndarray:
        """Goal state for the rollout's reference.

        With the entry ramp, the goal starts at the vehicle's position when the RTA phase begins and moves toward
        GOAL_STATE at bounded speed, instead of jumping from ~(3.9, -12.5) to (0, -0.6) at once (which saturated the
        roll rate and produced a ~2 m/s descent).
        """
        final = np.asarray(self.GOAL_STATE, dtype=float)
        if not self.options.entry_ramp:
            return final
        if warmup:
            return np.array([s.y, s.z, 0.0, 0.0, 0.0]) # pre-RTA: hold position (these plans are never flown)
        if self.ramp_origin is None:
            self.ramp_origin = (t, s.y, s.z)
        t0, y0, z0 = self.ramp_origin
        elapsed = max(t - t0, 0.0)
        dy = np.clip(final[0] - y0, -self.options.ramp_speed_y * elapsed, self.options.ramp_speed_y * elapsed)
        dz = np.clip(final[1] - z0, -self.options.ramp_speed_z * elapsed, self.options.ramp_speed_z * elapsed)
        return np.array([y0 + dy, z0 + dz, 0.0, 0.0, 0.0])

    def install_plan(self, result: RolloutResult) -> None:
        """Turn a finished rollout into an immutable RolloutPlan and publish it for the control thread."""
        req = result.request
        latency = time.time() - req.submitted_at

        collection_time = self.collection_time
        if not req.warmup:
            t_index = result.violation_idx
            safety_horizon = t_index * self.tube_timestep
            collection_time = req.t_start + safety_horizon  # Update the collection time based on the rollout start time and index
            self.debug(f"{collection_time=}\n{safety_horizon=}")

        # Stored tube slice for the log: starts where "now" is inside the plan once it is installed
        tube_start = int(latency // self.control_period) + 1
        tube_time_indices = slice(tube_start, tube_start + self.tube_extent, self.tube_skip)
        save_tube = result.reachable_tube[tube_time_indices][:, self.tube_pos_indices]

        with self.plan_lock:
            self.plan_seq += 1
            plan = RolloutPlan(t_start=req.t_start, dt=self.tube_timestep,
                               reachable_tube=result.reachable_tube,
                               rollout_ref=result.rollout_ref,
                               feedfwd_input=result.feedfwd_input,
                               collection_time=collection_time,
                               compute_time=result.compute_time,
                               latency=latency,
                               save_tube=save_tube,
                               tube_start=tube_start,
                               seq=self.plan_seq,
                               state0=req.state, K_feedback=req.K_feedback, K_reference=req.K_reference,
                               obs_wy=req.obs_wy, obs_wz=req.obs_wz,
                               violation_idx=result.violation_idx, warmup=req.warmup,
                               goal=req.goal, delta=req.delta)
            self.collection_time = collection_time
            self.plan = plan # <- the single atomic "publish"

        self.recorder.add_plan(plan)
        self.rollout_compute_times.append(result.compute_time)
        self.rollout_latencies.append(latency)
        self.debug(f"rollout #{plan.seq} ({'warm-up' if req.warmup else 'safety'}): "
                   f"compute {result.compute_time:.3f} s, latency {latency:.3f} s")


    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        self.in_offboard_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_land_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND)

        if not self.in_offboard_mode:
            self.debug(f"{BANNER}"
                  f"Not in offboard mode yet!"
                  f"Current vehicle status: {vehicle_status.nav_state}\n"
                  f"{VehicleStatus.NAVIGATION_STATE_OFFBOARD = }\n"
                  f"{self.armed=}\n"
                  f"{self.in_land_mode=}\n"
                  f"{BANNER}")
            return
        self.debug(f"{BANNER}In Offboard Mode!{BANNER}")

    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        with self.loop_stats['heartbeat'].measure():
            t = self.now()
            self.debug(f"{BANNER}In offboard callback at {t:.2f} seconds")

            if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
                self.debug(f"Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
                self.offboard_heartbeat_counter = 0
                return # skip the rest of this function if RC switch is not set to offboard

            if t < self.begin_actuator_control:
                publish_offboard_control_heartbeat_signal_position(self)
            elif t < self.land_time:
                publish_offboard_control_heartbeat_signal_body_rate(self)
            else:
                publish_offboard_control_heartbeat_signal_position(self)


            if self.offboard_heartbeat_counter <= 10:
                if self.offboard_heartbeat_counter == 10:
                    engage_offboard_mode(self)
                    arm(self)
                self.offboard_heartbeat_counter += 1

            if int(t) != int(t - self.heartbeat_period): # ~1 Hz status line (cheap, always on)
                plan = self.plan
                self.get_logger().info(
                    f"t={t:5.1f}s offboard={self.in_offboard_mode} armed={self.armed} "
                    f"plan#={plan.seq if plan else '-'} "
                    f"rollout={1e3 * plan.compute_time if plan else float('nan'):.0f}ms")

    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""
        with self.loop_stats['control'].measure():
            t = self.now()
            self.debug(f"{BANNER}In control callback at time: ", t)
            if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
                self.debug(f"Not in offboard mode.\n"
                      f"Current nav_state number: {self.vehicle_status.nav_state}\n"
                      f"nav_state number for offboard: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                      f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
                return  # skip the rest of this function if not in offboard mode
            s = self.state
            if s is None or self.backup_engaged is not None:
                return

            if t < self.begin_actuator_control:
                publish_position_setpoint(self, 0., self.max_y, self.max_height, 0.0)
            elif t < self.land_time:
                if self.plan is None or self.gains is None:
                    # First safety rollout not installed yet: hold the pre-RTA setpoint instead of crashing
                    publish_position_setpoint(self, 0., self.max_y, self.max_height, 0.0)
                    return
                self.control_administrator(t, s)
            elif t > self.land_time or (abs(s.z) <= 1.0 and t > 15):
                self.debug("Landing...")
                publish_position_setpoint(self, 0.0, 0.0, -0.83, 0.0)
                if abs(s.x) < 0.25 and abs(s.y) < 0.25 and abs(s.z) <= 0.90:
                    print("Vehicle is close to the ground, preparing to land.")
                    land(self)
                    disarm(self)
                    exit(0)
            else:
                raise ValueError("Unexpected time_from_start value or unexpected termination conditions")

    def get_ref(self, time_from_start: float) -> np.ndarray:
        """Get the reference trajectory for the LQR and NR tracker.

        Args:
            time_from_start (float): Time from the start of the program in seconds.

        Returns:
            tuple: Reference trajectory for LQR and NR tracker.
        """
        # Define your reference trajectories here based on time_from_start
        x_des = 0.0
        y_des = 0.0
        z_des = np.clip(self.max_height + 0.1 * time_from_start, self.max_height, -0.55)  # Clip z_des to be between self.max_height and -0.55

        yaw_des = 0.0

        ref_nr = np.array([x_des, y_des, z_des, yaw_des])  # Reference position setpoint for NR tracker (x, y, z, yaw)
        return ref_nr

    def control_administrator(self, t: float, s: VehicleState) -> None:
        self.debug(f"{BANNER}In control administrator at {t:.2f} seconds")
        ref_nr = self.get_ref(t)
        plan = self.plan # one read: the rollout thread may swap in a new plan at any moment
        if self.options.nr_ref_from_plan:
            # The RTA flies thrust and roll along the plan; the NR tracker only keeps pitch and yaw. If the NR's own
            # y/z reference differs from the plan (it was y=0, z=-12.5+0.1t), its coupled Newton step answers the
            # y/z error with pitch/yaw rates, which fights the RTA (logged: pitch swinging -37..+52 deg). Give it the
            # plan's y/z at its own prediction horizon, so only x and yaw are left for it to correct.
            ahead = plan.rollout_ref[plan.index_at(t + self.T_LOOKAHEAD)]
            ref_nr = np.array([ref_nr[0], ahead[0], ahead[1], ref_nr[3]])
        feedback_K = self.gains[0]
        if t > plan.collection_time:
            self.certification_gaps += 1 # running on an uncertified part of the plan (no certified successor yet)
            # Measure how long we have been CONTINUOUSLY uncertified -- not the age of this particular plan: near the
            # floor, fresh plans certified for 0 s arrive every ~10 ms, so no single plan is ever "old".
            if self.uncertified_since is None:
                self.uncertified_since = t
            if self.options.backup == 'land' and t - self.uncertified_since > self.options.backup_grace:
                self.engage_backup(t, f"no certified plan for {t - self.uncertified_since:.3f} s (latest plan #{plan.seq}: "
                                      f"violation row {plan.violation_idx}, lowest tube altitude "
                                      f"{-plan.reachable_tube[:plan.violation_idx + 1, 6].max():.2f} m)")
                return
        else:
            self.uncertified_since = None

        # Time-indexed reference: the row of the plan that corresponds to *now*, regardless of how many
        # control ticks actually ran since the plan was computed (replaces the per-tick traj_idx counter).
        traj_idx = plan.index_at(t)

        ctrl_T0 = time.perf_counter()
        # ONE compiled call: NR tracker (pitch/yaw rates) + RTA feedback (thrust/roll rate) + anti-windup clip
        new_u = self.kernels.control_step(s.nr_state_vector, s.rta_mm_gpr_state_vector_planar, self.last_input,
                                          ref_nr, plan.rollout_ref[traj_idx], plan.feedfwd_input[traj_idx],
                                          feedback_K)
        control_comp_time = time.perf_counter() - ctrl_T0
        self.debug(f"control step: {control_comp_time * 1e3:.3f} ms, {traj_idx=}, {new_u=}")

        self.last_input = new_u  # Update the last input for the next iteration (float64 NumPy array)
        new_force = new_u[0]
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])
        new_pitch_rate = float(new_u[2])
        new_yaw_rate = float(new_u[3])
        publish_body_rate_setpoint(self, new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)

        # One preallocated row per tick (plans themselves are stored once, when installed)
        self.recorder.tick(t, s.x, s.y, s.z, s.yaw, s.vx, s.vy, s.vz, s.roll, s.pitch,
                           float(new_force), new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate,
                           control_comp_time, self.wy, self.wz, plan, traj_idx)

    def engage_backup(self, t: float, reason: str) -> None:
        """Runtime-assurance fallback: hand the vehicle to PX4's LAND mode and stop issuing RTA commands."""
        if self.backup_engaged is not None:
            return
        self.backup_engaged = reason
        land(self)
        self.recorder.event(t, 'backup_land', reason)
        self.get_logger().warn(f"t={t:.2f} s: BACKUP -> LAND ({reason})")

    def update_lqr_feedback(self, sys, state, input, t: float): # sys kept for the original signature
            self.debug(f"{BANNER}UPDATING LQR")
            t0 = time.time()
            self.gains = self.compute_lqr_gains(state, input) # (K_feedback, K_reference), swapped in atomically
            self.recorder.gain_update(t, False, *self.gains)
            self.debug(f"LQR Update time: {time.time()-t0}")

            self.last_lqr_update_time = t  # Update the last LQR update time
            if self.first_LQR:
                self.first_LQR = False  # Set first_LQR to False after the first update
