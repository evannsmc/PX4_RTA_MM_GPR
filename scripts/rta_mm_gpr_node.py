from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy) # Import ROS2 QoS policy modules
from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
    VehicleStatus, FullState, #Import PX4 ROS2-API messages for receiving vehicle state information
    RcChannels
)


import time
import control
import numpy as np
import inspect
import traceback
from typing import Optional
from scipy.spatial.transform import Rotation as R

from px4_rta_mm_gpr.utilities.jax_setup import jit
from px4_rta_mm_gpr.jax_mm_rta import *
from px4_rta_mm_gpr.px4_functions import *
from px4_rta_mm_gpr.jax_nr import NR_tracker_original, dynamics
from px4_rta_mm_gpr.utilities import test_function, adjust_yaw

import immrax as irx
import jax.numpy as jnp
from Logger import LogType, VectorLogType # pyright: ignore[reportMissingImports]

BANNER = '\n' + "==" * 30 + '\n'

class OffboardControl(Node):
    def __init__(self, sim: bool) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

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



        # Logging related variables
        self.time_log = LogType("time", 0)

        self.x_log = LogType("x", 1)
        self.y_log = LogType("y", 2)
        self.z_log = LogType("z", 3)
        self.yaw_log = LogType("yaw", 4)

        self.ctrl_comp_time_log = LogType("ctrl_comp_time", 5)
        self.rollout_comptime_log = LogType("rollout_comptime", 6)

        self.y_ref_log = LogType("y_ref", 7)
        self.z_ref_log = LogType("z_ref", 8)
        self.yaw_ref_log = LogType("yaw_ref", 9)

        self.throttle_log = LogType("throttle", 10)
        self.roll_rate_log = LogType("roll_rate", 11)
        self.pitch_rate_log = LogType("pitch_rate", 12)
        self.yaw_rate_log = LogType("yaw_rate", 13)

        self.save_tube_log = VectorLogType("save_tube", 14, ['pyL', 'pzL', 'pyH', 'pzH'])
        self.wy_log = LogType("wy", 15)
        self.wz_log = LogType("wz", 16)

        self.tube_pos_indices = [0, 1, 5, 6]  # Indices for x, y, z, yaw in the rollout reference trajectory

        self.tube_start = 10
        self.tube_extent = 35
        self.tube_skip = 4

        tube_start = self.tube_start
        tube_end = self.tube_start + self.tube_extent
        self.tube_time_indices = slice(tube_start, tube_end, self.tube_skip)

        # self.num_save = len(range(*self.tube_time_indices.indices(26)))
        # print(f"{self.tube_time_indices=}, {self.tube_pos_indices=}, {self.num_save=}")
        # exit(0)



        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                ])

##########################################################################################
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
        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_odometry_subscriber_callback, qos_profile)
        

        self.in_offboard_mode: bool = False       
        self.armed: bool = False
        self.in_land_mode: bool = False
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
            
        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        print(f"RC switch mode: {'On' if self.offboard_mode_rc_switch_on else 'Off'}")
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile
        )
        
        # MoCap related variables
        self.mocap_initialized: bool = False
        self.full_rotations: int = 0
        self.max_yaw_stray = 5 * np.pi / 180

        # PX4 variables
        self.offboard_heartbeat_counter: int = 0
        self.vehicle_status = VehicleStatus()
        # self.takeoff_height = -5.0

        # Callback function time constants
        self.heartbeat_period: float = 0.1 # (s) We want 10Hz for offboard heartbeat signal
        self.control_period: float = 0.01 # (s) We want 100Hz for direct control algorithm
        self.wind_estimate_period: float = 0.1 # (s) We want 10Hz for wind estimation update
        self.traj_idx = 0 # Index for trajectory setpoint

        self.OBS_DYN = jnp.array([
                                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # Timers for my callback functions
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback) #My control algorithm needs to execute at >= 100Hz
        self.rollout_timer = self.create_timer(self.control_period,
                                               self.rollout_callback) #My rollout function needs to execute at >= 100Hz
        self.wind_estimator = self.create_timer(self.wind_estimate_period,
                                                self.wind_estimator_callback)

        self.init_jit_compile_nr_rta() # Initialize JIT compilation for NR tracker and RTA pipeline
        self.T0 = time.time()  # Reset initial time after JIT compilation

        self.last_lqr_update_time: float = 0.0  # Initialize last LQR update time
        self.first_LQR: bool = True  # Flag to indicate if this is the first LQR update
        self.collection_time: float = 0.0  # Time at which the collection starts

        # Time variables
        self.T0 = time.time() # (s) initial time of program
        self.time_from_start = time.time() - self.T0 # (s) time from start of program 
        self.begin_actuator_control = 15 # (s) time after which we start sending actuator control commands
        self.land_time = self.begin_actuator_control + 20 # (s) time after which we start sending landing commands
        if self.sim:
            self.max_height = -12.5
            self.max_y = 4.0
        else:
            self.max_height = -2.5
            self.max_y = 0.75
            # raise NotImplementedError("Hardware not implemented yet.")

    def init_jit_compile_nr_rta(self):
        """
        Initialize JIT compilation for NR tracker and RTA pipeline.

        You must run jit-compiled functions the first time before actually using them in order to trigger the JIT compilation.

        Otherwise, you'll deploy code that hasn't yet been compiled, which can lead to runtime errors or suboptimal performance.
        """
        print(f"{BANNER}Initializing JIT compilation for NR tracker and RTA pipeline.")

        def time_fns(func):
            def wrapper(*args, **kwargs):
                time0 = time.time()
                result1 = func(*args, **kwargs)
                time1 = time.time()
                result2 = func(*args, **kwargs)
                time2 = time.time()

                tf1 = time1 - time0
                tf2 = time2 - time1
                speedup_factor = tf1 / tf2 if tf2 != 0 else 0
                print(f"\nTime taken for {func.__name__}: {time1 - time0}")
                print(f"Time taken for {func.__name__} (JIT): {time2 - time1}")
                print(f"Speedup factor for {func.__name__} (JIT): {speedup_factor}\n")

                return result2
            return wrapper
        
        @time_fns
        def jit_compile_nr_tracker():
            NR_tracker_original(init_state, init_input, init_ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS) # JIT-compile the NR tracker function

        @time_fns
        def jit_compile_linearize_system():
            A, B = jitted_linearize_system(self.quad_sys_planar, x0, u0, w0)
            return A, B
        

        @time_fns
        def jit_compile_lqr():
            K_reference, P, _ = control.lqr(A, B, self.Q_ref_planar, self.R_ref_planar)
            K_feedback, P, _ = control.lqr(A, B, self.Q_planar, self.R_planar)
            return K_feedback, K_reference
    
        @time_fns
        def jit_compile_rollout():
            reachable_tube, rollout_ref, rollout_feedfwd_input = jitted_rollout(jnp.array([0.]), ix0, x0, K_feedback, K_reference, self.gz_wind_obs_in_y, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM, self.MASS, self.ulim_planar, self.quad_sys_planar, self.GOAL_STATE)
            reachable_tube.block_until_ready()
            rollout_ref.block_until_ready()
            rollout_feedfwd_input.block_until_ready()
            return reachable_tube, rollout_ref, rollout_feedfwd_input
        
    
        @time_fns
        def jit_compile_collection_id():
            violation_safety_time_idx = collection_id_jax(rollout_ref, reachable_tube)
            return violation_safety_time_idx
 
        @time_fns
        def jit_compile_u_applied():
            applied_u = u_applied(x0, x0, u0, K_feedback, self.ulim_planar)
            return applied_u
        
        
        # Initialize NR algorithm parameters
        self.last_input: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.01, 0.01, 0.01]) # last input to the controller
        self.hover_input_planar: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.]) # hover input to the controller
        self.T_LOOKAHEAD: float = 0.8 # (s) lookahead time for the controller in seconds
        self.T_LOOKAHEAD_PRED_STEP: float = 0.1 # (s) we do state prediction for T_LOOKAHEAD seconds ahead in intervals of T_LOOKAHEAD_PRED_STEP seconds
        self.INTEGRATION_TIME: float = self.control_period # integration time constant for the controller in seconds

        # Initialize state, input, noise, ref variables
        init_state = jnp.array([0.1, 0.1, 0.1, 0.02, 0.03, 0.02, 0.01, 0.01, 0.03]) # Initial state vector for testing
        init_input = self.last_input  # Initial input vector for testing
        init_noise = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
        init_ref = jnp.array([0.0, 0.0, -3.0, 0.0])  # Initial reference vector for testing

        # Initialize rta_mm_gpr variables
        self.GOAL_STATE = jnp.array([0., -0.6, 0., 0., 0.])
        x0 = jnp.array(init_state[0:5])  # Initial state vector for testing

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


        # x0 = jnp.array([-20, -10, 0.0, 10.0, 20.0])  # Initial state vector for testing
        self.x_pert = 5e-4 * jnp.array([1., 1., 1., 1., 1.]) # 
        ix0 = irx.icentpert(x0, self.x_pert)
        u0 = jnp.array(init_input[0:2])  # Initial input vector for testing
        w0 = jnp.array(init_noise)  # Initial noise vector for testing
        print(f"{x0=}, {ix0=}, {ix0.shape=}")


        # Initialize rollout parameters
        self.n_obs = 9
        obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(initialization_GP, 0.0, x0)[0]]]),(self.n_obs,1))
        self.obs = obs
        # self.wind_obs_z0 = obs

        self.wind_count = 0
        self.gz_wind_obs_in_y = obs   # for example, 500 rows of 3-D data for wind observations in y-direction at various time and & z-heights
        self.gy_wind_obs_in_z = obs   # for example, 500 rows of 3-D data for wind observations in z-direction at various time and & y-heights

        self.quad_sys_planar = PlanarMultirotorTransformed(mass=self.MASS)
        self.ulim_planar = irx.interval([0, -1],[21, 1]) # type: ignore # Input saturation interval -> -5 <= u1 <= 15, -5 <= u2 <= 5
        self.Q_planar = jnp.array([10, 5, 1, 1, 1]) * jnp.eye(self.quad_sys_planar.xlen) # weights that prioritize overall tracking of the reference (defined below)
        self.R_planar = jnp.array([1, 1]) * jnp.eye(2)


        #(py,pz,h,v,theta)
        # self.Q_ref_planar = jnp.array([50, 50, 200, 200, 1]) * jnp.eye(self.quad_sys_planar.xlen) # Different weights that prioritize reference reaching origin

        self.Q_ref_planar =jnp.array([50, 20, 50, 20, 3]) * jnp.eye(self.quad_sys_planar.xlen) # Different weights that prioritize reference reaching origin
        self.R_ref_planar = jnp.array([50, 20]) * jnp.eye(2)


        t0 = 0.0  # Initial time
        self.tube_timestep = 0.01  # Time step
        self.tube_horizon = 30.0   # Reachable tube horizon
        self.sys_mjacM = irx.mjacM(self.quad_sys_planar.f) # create a mixed Jacobian inclusion matrix for the system dynamics function
        self.perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)) # create a permutation for the inclusion system calculation



        jit_compile_nr_tracker() # JIT-compile NR tracker
        A, B = jit_compile_linearize_system() # JIT-compile linearize system
        K_feedback, K_reference = jit_compile_lqr() # LQR JIT-compile
        reachable_tube, rollout_ref, rollout_feedfwd_input = jit_compile_rollout() # JIT-compile rollout
        violation_safety_time_idx = jit_compile_collection_id()
        applied_u = jit_compile_u_applied()

        print(f"{A=},{B=}")
        print(f"{K_feedback=}\n{K_reference=}")
        print(f"{reachable_tube[0:1, :]=}")
        print(f"{rollout_ref=},{rollout_feedfwd_input=}")
        print(f"{reachable_tube.shape = }, {rollout_ref.shape = }, {rollout_feedfwd_input.shape = }")
        print(f"Collection ID: {violation_safety_time_idx}")
        print(f"Applied u: {applied_u}")
        # exit(0)

        # Pause for 3 seconds to give myself time to read the print statements above
        print(f"\nPausing for 3 seconds to read the JIT compilation times above.\nContinuing...\n")
        time.sleep(3)



    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print(f"{BANNER}In RC Channel Callback")
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False

    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        print(f"{BANNER}Received odometry data: {msg=}")

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = (msg.position[2] + 0.5) if (self.sim and (abs(msg.position[2]) < 1.2)) else msg.position[2]  # Adjust for sim ground level if needed

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        self.ax = msg.acceleration[0]
        self.ay = msg.acceleration[1]
        self.az = msg.acceleration[2]

        self.roll, self.pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = adjust_yaw(self, yaw)  # Adjust yaw to account for full rotations
        self.rotation_object = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.ax, self.ay, self.az, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector_planar = np.array([self.y, self.z, self.vy, self.vz, self.roll])# px, py, h, v, theta = x
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.acceleration = np.array([self.ax, self.ay, self.az])
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        print(f"in odom, flat output: {self.output_vector}")
        if self.first_LQR:
            t00 = time.time()
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            A, B = jitted_linearize_system(self.quad_sys_planar, self.rta_mm_gpr_state_vector_planar, self.hover_input_planar, noise)
            A, B = np.array(A), np.array(B)
            # print(f"Time to linearize system: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, self.Q_planar, self.R_planar)
            self.feedback_K = 1 * K
            # print(f"Time taken for LQR synthesis for K_feedback: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, self.Q_ref_planar, self.R_ref_planar)  # Compute the LQR gain matrix
            self.reference_K = 1 * K  # Store the reference gain matrix
            # print(f"Time taken for LQR synthesis for K_reference: {time.time() - t0} seconds")

            
            self.last_lqr_update_time = time.time() - self.T0  # Set the last LQR update time to the current time
            print(f"Odom: time taken for entire LQR update: {time.time() - t00} seconds")



        ODOMETRY_DEBUG_PRINT = True
        if ODOMETRY_DEBUG_PRINT:
            print(f"{self.nr_state_vector=}")
            print(f"{self.output_vector=}")
            print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")


    def wind_estimator_callback(self):
        """Callback function for the wind estimation callback"""
        print(f"{BANNER}In wind callback")
        if not self.in_offboard_mode:
            print("Not in offboard mode, skipping wind estimation")
            return
        
        wind_estimate_time = time.time() - self.T0


        # Estimate wind in y and z (horizontal and vertical) directions using difference between measured and predicted acceleration
        _, ay_hat, az_hat = self.OBS_DYN@dynamics(self.nr_state_vector, self.last_input, self.MASS)
        print(f"{ay_hat = }, {az_hat = }")
        print(f"{self.ay}, {self.az}")
        tracking_error_estimate = 0.09  # (m/s^2) estimate of the tracking error due to unmodeled dynamics and state estimation errors


        # Estimate wind disturbance force in y-direction
        ay_wind = (self.ay - ay_hat) 
        gz_windforce_in_y = self.MASS * ay_wind
        self.wy = gz_windforce_in_y


        # Estimate wind disturbance force in z-direction
        az_wind = (self.az - az_hat)
        gy_windforce_in_z = self.MASS * az_wind
        self.wz = gy_windforce_in_z

        # Prep to fill in wind observation data for GPR
        wind_idx = self.wind_count % self.n_obs
        self.wind_count += 1

        gz_windforce_GPR_data_y = (wind_estimate_time, self.z, gz_windforce_in_y)
        self.gz_wind_obs_in_y.at[wind_idx, :].set(jnp.array(gz_windforce_GPR_data_y))


        gy_windforce_GPR_data_z = (wind_estimate_time, self.y, gy_windforce_in_z)
        self.gy_wind_obs_in_z.at[wind_idx, :].set(jnp.array(gy_windforce_GPR_data_z))


        # print(f"Wind in Y be added to buffer: [T, HEIGHT, WIND_CALC]: {gz_windforce_GPR_data_y}")
        # print(f"Wind in Z be added to buffer: [T, HEIGHT, WIND_CALC]: {gy_windforce_GPR_data_z}")
        # print(f"Default fake wind {self.obs}")
        
        if self.wind_count % 50 == 0 and wind_estimate_time < self.begin_actuator_control:
            tr0 = time.time()
            print("Running rollout after wind update")
            self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = jitted_rollout(jnp.array([wind_estimate_time]), irx.icentpert(self.rta_mm_gpr_state_vector_planar, self.x_pert), self.rta_mm_gpr_state_vector_planar, self.feedback_K, self.reference_K, self.gz_wind_obs_in_y, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM, self.MASS, self.ulim_planar, self.quad_sys_planar, self.GOAL_STATE)

            self.reachable_tube.block_until_ready()
            self.rollout_ref.block_until_ready()
            self.rollout_feedfwd_input.block_until_ready()

            # print(f"{self.reachable_tube=}")
            print(f"Ran rollout after wind update: {time.time() - tr0} seconds")
            # print(f"{self.reachable_tube[0:1, :]=}")
            # exit(0)


    def rollout_callback(self):
        """Callback function for the rollout timer."""
        print(f"{BANNER}In rollout callback at time: ", time.time() - self.T0)
        if self.begin_actuator_control - 1.0 <= self.time_from_start <= self.land_time:
            try:
                self.time_from_start = time.time() - self.T0
                t00 = time.time()  # Start time for rollout computation
                thresh = 1.0
                current_time = self.time_from_start
                # current_state = self.rta_mm_gpr_state_vector_planar
                # current_state_interval = irx.icentpert(current_state, self.x_pert)
                print(f"{current_time= }, {self.collection_time= }")

                if current_time >= self.collection_time:
                    print("Unsafe region begins now. Recomputing reachable tube and reference trajectory.")


                    tr0 = time.time()

                    self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = jitted_rollout(jnp.array([current_time]), irx.icentpert(self.rta_mm_gpr_state_vector_planar, self.x_pert), self.rta_mm_gpr_state_vector_planar, self.feedback_K, self.reference_K, self.gz_wind_obs_in_y, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM, self.MASS, self.ulim_planar, self.quad_sys_planar, self.GOAL_STATE)

                    self.reachable_tube.block_until_ready()
                    self.rollout_ref.block_until_ready()
                    self.rollout_feedfwd_input.block_until_ready()


                    # print(f"{self.reachable_tube[0:3,:]=}")
                    self.rollout_comptime = time.time() - tr0
                    print(f"rollout calc time: {self.rollout_comptime} seconds")


                    # self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = jitted_rollout(
                    #     current_time, current_state_interval, current_state, self.feedback_K, self.reference_K, self.wind_obs_z, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM, self.MASS, self.ulim_planar, self.quad_sys_planar, self.GOAL_STATE
                    # )

                    # self.reachable_tube.block_until_ready()
                    # self.rollout_ref.block_until_ready()
                    # self.rollout_feedfwd_input.block_until_ready()
                    # print(f"Time taken by rollout: {time.time() - t0:.4f} seconds")

                    # t0 = time.time()  # Reset time for collection index computation
                    t_index = collection_id_jax(self.rollout_ref, self.reachable_tube, thresh)
                    t_index = int(t_index)
                    # print(f"Time taken for collection index computation: {time.time() - t0:.4f} seconds")

                    safety_horizon = t_index * self.tube_timestep
                    self.collection_time = current_time + safety_horizon  # Update the collection time based on the current time and index
                    print(f"{self.collection_time=}\n{safety_horizon=}")

                    self.traj_idx = 0

                    self.save_tube = self.reachable_tube[self.tube_time_indices, self.tube_pos_indices] # maybe mess with this
                    # exit(0)


                else:
                    print("You're safe!")
                print(f"Time taken for whole rollout process: {time.time() - t00:.4f} seconds")


            # except AttributeError as e: # for if we 
            #     print("Ignoring missing attribute:", e)
            #     return
            except Exception as e:
                frame = inspect.currentframe()
                func_name = frame.f_code.co_name if frame is not None else "<unknown>"
                print(f"\nError in {__name__}:{func_name}: {e}")
                traceback.print_exc()
                raise  # Re-raise  
        else:
            pass


    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        self.in_offboard_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_land_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND)

        if not self.in_offboard_mode:
            print(f"{BANNER}"
                  f"Not in offboard mode yet!"
                  f"Current vehicle status: {vehicle_status.nav_state}\n"
                  f"{VehicleStatus.NAVIGATION_STATE_OFFBOARD = }\n"
                  f"{self.armed=}\n"
                  f"{self.in_land_mode=}\n"
                  f"{BANNER}")
            return
        print(f"{BANNER}In Offboard Mode!{BANNER}")

    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        self.time_from_start = time.time() - self.T0
        t = self.time_from_start
        print(f"{BANNER}In offboard callback at {self.time_from_start:.2f} seconds")

        if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
            print(f"Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
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

    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""
        print(f"{BANNER}In control callback at time: ", time.time() - self.T0)  
        self.time_from_start = time.time() - self.T0
        t = self.time_from_start
        if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
            print(f"Not in offboard mode.\n"
                  f"Current nav_state number: {self.vehicle_status.nav_state}\n"
                  f"nav_state number for offboard: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                  f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
            return  # skip the rest of this function if not in offboard mode

        if t < self.begin_actuator_control:
            publish_position_setpoint(self, 0., self.max_y, self.max_height, 0.0)
        elif t < self.land_time:
            self.control_administrator()
        elif t > self.land_time or (abs(self.z) <= 1.0 and t > 15):
            print("Landing...")
            publish_position_setpoint(self, 0.0, 0.0, -0.83, 0.0)
            if abs(self.x) < 0.25 and abs(self.y) < 0.25 and abs(self.z) <= 0.90:
                print("Vehicle is close to the ground, preparing to land.")
                land(self)
                disarm(self)
                exit(0)
        else:
            raise ValueError("Unexpected time_from_start value or unexpected termination conditions")

    def get_ref(self, time_from_start: float) -> jnp.ndarray:
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

        vx_des = 0.0
        vy_des = 0.0
        vz_des = 0.0

        roll_des = 0.0
        pitch_des = 0.0
        yaw_des = 0.0

        # ref_lqr_planar = jnp.array([y_des, z_des, vy_des, vz_des, roll_des])  # Reference position setpoint for planar LQR tracker (y, z, vy, vz, roll)
        # ref_lqr_3D = jnp.array([x_des, y_des, z_des, vx_des, vy_des, vz_des, roll_des, pitch_des, yaw_des])
        ref_nr = jnp.array([x_des, y_des, z_des, yaw_des])  # Reference position setpoint for NR tracker (x, y, z, yaw)
        return ref_nr  

    def control_administrator(self) -> None:
        self.time_from_start = time.time() - self.T0
        print(f"{BANNER}In control administrator at {self.time_from_start:.2f} seconds")
        ref_nr = self.get_ref(self.time_from_start)

        ctrl_T0 = time.time()
        NR_new_u, _ = NR_tracker_original(self.nr_state_vector, self.last_input, ref_nr, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        print(f"Time taken for NR tracker: {time.time() - ctrl_T0:.4f} seconds")

        rta_T0 = time.time()
        rta_new_u_planar = self.rta_mm_gpr_administrator(self.rta_mm_gpr_state_vector_planar, self.last_input[0:2])  # Compute RTA-MM GPR control input for planar system
        print(f"Time taken for RTA-MM GPR administrator: {time.time() - rta_T0:.4f} seconds")
        control_comp_time = time.time() - ctrl_T0 # Time taken for control computation
        print(f"\nEntire control Computation Time: {control_comp_time:.4f} seconds, Good for {1/control_comp_time:.2f}Hz control loop")

        print(f"{NR_new_u =}")
        print(f"{rta_new_u_planar =}")
        new_u = jnp.hstack([rta_new_u_planar, NR_new_u[2:]])  # New control input from the RTA-MM GPR tracker
        print(f"{new_u = }")

        self.last_input = new_u  # Update the last input for the next iteration
        new_force = new_u[0]
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3])    # Convert jax.numpy array to float
        publish_body_rate_setpoint(self, new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)
        # exit(0)

        # Log the states, inputs, and reference trajectories for data analysis
        state_input_ref_log_info = [self.time_from_start,
                                    float(self.x), float(self.y), float(self.z), float(self.yaw),
                                    control_comp_time,
                                    self.rollout_comptime,
                                    0., self.y_ref, self.z_ref, self.yaw_ref,
                                    new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate,
                                    self.wy, self.wz
                                    ]
        self.update_logged_data(state_input_ref_log_info)
        for reach_set in self.save_tube:
            self.update_tube_data(reach_set)

        for i in range(len(self.y_ref)):
            ref_data = [self.y_ref[i], self.z_ref[i], self.yaw_ref[i]]
            self.update_ref_data(ref_data)

        print("==" * 30)

    def update_lqr_feedback(self, sys, state, input, noise):
            print(f"{BANNER}UPDATING LQR")
            t0 = time.time()
            A, B = jitted_linearize_system(sys, state, input, noise)  # Linearize the system dynamics
            K, P, _ = control.lqr(A, B, self.Q_planar, self.R_planar)
            self.feedback_K = 1 * K

            K, P, _ = control.lqr(A, B, self.Q_ref_planar, self.R_ref_planar)  # Compute the LQR gain matrix
            self.reference_K = 1 * K  # Store the reference gain matrix
            print(f"LQR Update time: {time.time()-t0}")

            self.last_lqr_update_time = self.time_from_start  # Update the last LQR update time
            if self.first_LQR:
                self.first_LQR = False  # Set first_LQR to False after the first update
            PRINT_LQR_DEBUG = False  # Set to True to print debug information for LQR
            if PRINT_LQR_DEBUG:
                print(f"\n\n{'=' * 60}")
                print(f"Linearized System Matrices:\n{A=}\n{B=}")
                print(f"LQR Gain Matrix:\n{K=}")
                print(f"Feedback Gain Matrix:\n{self.feedback_K}")
                print(f"{A.shape=}, {B.shape=}, {self.feedback_K.shape=}")
                print(f"{'=' * 60}\n\n")

    def rta_mm_gpr_administrator(self, state, input):
        """Run the RTA-MM administrator to compute the control inputs."""
        self.time_from_start = time.time() - self.T0 # Update time from start of the program
        print(f"{BANNER}In RTA-MM GPR Administrator at {self.time_from_start=:.2f}")

        t0 = time.time()  # Start time for RTA-MM GPR computation
        current_state = self.rta_mm_gpr_state_vector_planar # Get the current state vector

        # Re-linearize and re-compute the LQR gain every X seconds or when the yaw exceeds the maximum stray
        if (self.time_from_start - self.last_lqr_update_time) >= 2.5 or abs(self.yaw) > self.max_yaw_stray:  
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            self.update_lqr_feedback(self.quad_sys_planar, state, input, noise)

        # Re-compute LQR input
        applied_input = u_applied(current_state, self.rollout_ref[self.traj_idx, :], self.rollout_feedfwd_input[self.traj_idx, :], self.feedback_K, self.ulim_planar)
        self.traj_idx += 1 #update trajectory index
        print(f"Ultimate ref (y,z): {self.rollout_ref[-1,:2]}")
        print(f"{self.traj_idx=}")


        ref_save_start = self.traj_idx + self.tube_start
        ref_save_end = ref_save_start + self.tube_extent
        row_indices = slice(ref_save_start, ref_save_end, self.tube_skip)

        self.y_ref = self.rollout_ref[row_indices, 0]
        self.z_ref = self.rollout_ref[row_indices, 1]
        self.yaw_ref = self.rollout_ref[row_indices, 4]
        # self.vy_ref = self.rollout_ref[self.traj_idx, 2]
        # self.vz_ref = self.rollout_ref[self.traj_idx, 3]

        PRINT_RTA_DEBUG = False  # Set to True to print debug information for RTA-MM GPR
        if PRINT_RTA_DEBUG:
            print(f"{current_state=}")
            print(f"{self.rollout_ref[self.traj_idx, :] =}")
            print(f"{self.rollout_feedfwd_input[self.traj_idx, :] =}")
            print(f"{applied_input=}")

        print(f"Time taken for RTA-MM GPR computation: {time.time() - t0:.4f} seconds")
        return applied_input

# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])

        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])

        self.ctrl_comp_time_log.append(data[5])
        self.rollout_comptime_log.append(data[6])

        self.throttle_log.append(data[11])
        self.roll_rate_log.append(data[12])
        self.pitch_rate_log.append(data[13])
        self.yaw_rate_log.append(data[14])

        self.wy_log.append(data[15])
        self.wz_log.append(data[16])

    def update_tube_data(self, data):
        self.save_tube_log.append(*data)

    def update_ref_data(self, data):
        self.y_ref_log.append(data[0])
        self.z_ref_log.append(data[1])
        self.yaw_ref_log.append(data[2])