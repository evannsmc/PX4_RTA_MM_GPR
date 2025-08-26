import os
import sys

import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy) # Import ROS2 QoS policy modules
from mocap4r2_msgs.msg import FullState
from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
    VehicleStatus, RcChannels #Import PX4 ROS2-API messages for receiving vehicle state information
)

import time
import traceback
from typing import Optional

import control
import math as m
import numpy as np  
from functools import partial
from scipy.spatial.transform import Rotation as R

from px4_rta_mm_gpr.px4_functions import *
from px4_rta_mm_gpr.utilities import test_function
from px4_rta_mm_gpr.jax_nr import NR_tracker_original#, dynamics, predict_output, get_jac_pred_u, fake_tracker, NR_tracker_flat, NR_tracker_linpred
from px4_rta_mm_gpr.utilities import sim_constants # Import simulation constants
from px4_rta_mm_gpr.jax_mm_rta import *

import jax
import jax.numpy as jnp
import immrax as irx
from Logger import Logger, LogType, VectorLogType, install_shutdown_logging # pyright: ignore[reportMissingImports]


# Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)

GP_instantiation_values = jnp.array([[-2, 0.0], #make the second column all zeros
                                    [0, 0.0],
                                    [2, 0.0],
                                    [4, 0.0],
                                    [6, 0.0],
                                    [8, 0.0],
                                    [10, 0.0],
                                    [12, 0.0]]) # at heights of y in the first column, disturbance to the values in the second column
# add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
actual_disturbance_GP = TVGPR(jnp.hstack((jnp.zeros((GP_instantiation_values.shape[0], 1)), GP_instantiation_values)), 
                                       sigma_f = 5.0, 
                                       l=2.0, 
                                       sigma_n = 0.01,
                                       epsilon=0.1,
                                       discrete=False
                                       )



class OffboardControl(Node):
    def __init__(self, sim: bool) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        test_function()
        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

        if self.sim:
            print("Using simulator constants and functions")
            from px4_rta_mm_gpr.utilities import sim_constants # Import simulation constants
            self.MASS = sim_constants.MASS
            self.THRUST_CONSTANT = sim_constants.THRUST_CONSTANT #x500 gazebo simulation motor thrust constant
            self.MOTOR_VELOCITY_ARMED = sim_constants.MOTOR_VELOCITY_ARMED #x500 gazebo motor velocity when armed
            self.MAX_ROTOR_SPEED = sim_constants.MAX_ROTOR_SPEED #x500 gazebo simulation max rotor speed
            self.MOTOR_INPUT_SCALING = sim_constants.MOTOR_INPUT_SCALING #x500 gazebo simulation motor input scaling
            # exit(0)

        elif not self.sim:
            print("Using hardware constants and functions")
            #TODO: do the hardware version of the above here
            try:
                from px4_rta_mm_gpr.utilities import hardware_constants
                self.MASS = hardware_constants.MASS
            except ImportError:
                raise ImportError("Hardware not implemented yet.")


        # Logging related variables
        self.time_log = LogType("time", 0)
        self.x_log = LogType("x", 1)
        self.y_log = LogType("y", 2)
        self.z_log = LogType("z", 3)
        self.yaw_log = LogType("yaw", 4)
        self.ctrl_comp_time_log = LogType("ctrl_comp_time", 5)
        self.x_ref_log = LogType("x_ref", 6)
        self.y_ref_log = LogType("y_ref", 7)
        self.z_ref_log = LogType("z_ref", 8)
        self.yaw_ref_log = LogType("yaw_ref", 9)
        self.throttle_log = LogType("throttle", 10)
        self.roll_rate_log = LogType("roll_rate", 11)
        self.pitch_rate_log = LogType("pitch_rate", 12)
        self.yaw_rate_log = LogType("yaw_rate", 13)
        self.save_tube_log = VectorLogType("save_tube", 14, ['pyL', 'pzL', 'hL', 'vL', 'thetaL', 'pyH', 'pzH', 'hH', 'vH', 'thetaH'])


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
        self.vehicle_odometry_subscriber = self.create_subscription(
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_odometry_subscriber_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_subscriber_callback, qos_profile)
            
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
        self.control_period: float = 0.01 # (s) We want 1000Hz for direct control algorithm
        self.traj_idx = 0 # Index for trajectory setpoint

        # Timers for my callback functions
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback) #My control algorithm needs to execute at >= 100Hz
        self.rollout_timer = self.create_timer(self.control_period,
                                               self.rollout_callback) #My rollout function needs to execute at >= 100Hz

        self.init_jit_compile_nr_rta() # Initialize JIT compilation for NR tracker and RTA pipeline

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
        else:
            self.max_height = -2.5
            # raise NotImplementedError("Hardware not implemented yet.")


    def init_jit_compile_nr_rta(self):
        """
        Initialize JIT compilation for NR tracker and RTA pipeline.

        You must run jit-compiled functions the first time before actually using them in order to trigger the JIT compilation.

        Otherwise, you'll deploy code that hasn't yet been compiled, which can lead to runtime errors or suboptimal performance.
        """

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
            A, B = jitted_linearize_system(quad_sys_planar, x0, u0, w0)
            return A, B
        

        @time_fns
        def jit_compile_lqr():
            K_reference, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)
            K_feedback, P, _ = control.lqr(A, B, Q_planar, R_planar)
            return K_feedback, K_reference
    
        @time_fns
        def jit_compile_rollout():
            reachable_tube, rollout_ref, rollout_feedfwd_input = jitted_rollout(0.0, ix0, x0, K_feedback, K_reference, self.obs, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM)
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
            applied_u = u_applied(x0, x0, u0, K_feedback)
            return applied_u
        
        
        # Initialize NR algorithm parameters
        self.last_input: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.01, 0.01, 0.01]) # last input to the controller
        self.hover_input_planar: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.]) # hover input to the controller
        self.odom_counter = 0
        self.T_LOOKAHEAD: float = 0.8 # (s) lookahead time for the controller in seconds
        self.T_LOOKAHEAD_PRED_STEP: float = 0.1 # (s) we do state prediction for T_LOOKAHEAD seconds ahead in intervals of T_LOOKAHEAD_PRED_STEP seconds
        self.INTEGRATION_TIME: float = self.control_period # integration time constant for the controller in seconds

        # Initialize state, input, noise, ref variables
        init_state = jnp.array([0.1, 0.1, 0.1, 0.02, 0.03, 0.02, 0.01, 0.01, 0.03]) # Initial state vector for testing
        init_input = self.last_input  # Initial input vector for testing
        init_noise = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
        init_ref = jnp.array([0.0, 0.0, -3.0, 0.0])  # Initial reference vector for testing

        # Initialize rta_mm_gpr variables
        n_obs = 9
        x0 = jnp.array(init_state[0:5])  # Initial state vector for testing
        self.obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))
        self.x_pert = 1e-4 * jnp.array([1., 1., 1., 1., 1.])
        ix0 = irx.icentpert(x0, self.x_pert)
        u0 = jnp.array(init_input[0:2])  # Initial input vector for testing
        w0 = jnp.array(init_noise)  # Initial noise vector for testing
        print(f"{x0=}, {ix0=}, {ix0.shape=}")


        # Initialize rollout parameters
        t0 = 0.0  # Initial time
        self.tube_timestep = 0.01  # Time step
        self.tube_horizon = 30.0   # Reachable tube horizon
        self.sys_mjacM = irx.mjacM(quad_sys_planar.f) # create a mixed Jacobian inclusion matrix for the system dynamics function
        self.perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)) # create a permutation for the inclusion system calculation



        jit_compile_nr_tracker() # JIT-compile NR tracker
        A, B = jit_compile_linearize_system() # JIT-compile linearize system
        K_feedback, K_reference = jit_compile_lqr() # LQR JIT-compile
        reachable_tube, rollout_ref, rollout_feedfwd_input = jit_compile_rollout() # JIT-compile rollout
        violation_safety_time_idx = jit_compile_collection_id()
        applied_u = jit_compile_u_applied()

        print(f"{A=},{B=}")
        print(f"{K_feedback=}\n{K_reference=}")
        print(f"{reachable_tube=},{rollout_ref=},{rollout_feedfwd_input=}")
        print(f"Collection ID: {violation_safety_time_idx}")
        print(f"Applied u: {applied_u}")

        # exit(0)

    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('In RC Channel Callback')
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False

    def adjust_yaw(self, yaw: float) -> float:
        """Adjust yaw angle to account for full rotations and return the adjusted yaw.

        This function keeps track of the number of full rotations both clockwise and counterclockwise, and adjusts the yaw angle accordingly so that it reflects the absolute angle in radians. It ensures that the yaw angle is not wrapped around to the range of -pi to pi, but instead accumulates the full rotations.
        This is particularly useful for applications where the absolute orientation of the vehicle is important, such as in control algorithms or navigation systems.
        The function also initializes the first yaw value and keeps track of the previous yaw value to determine if a full rotation has occurred.

        Args:
            yaw (float): The yaw angle in radians from the motion capture system after being converted from quaternion to euler angles.

        Returns:
            psi (float): The adjusted yaw angle in radians, accounting for full rotations.
        """        
        mocap_psi = yaw
        psi = None

        if not self.mocap_initialized:
            self.mocap_initialized = True
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi
            return psi

        # MoCap angles are from -pi to pi, whereas the angle state variable should be an absolute angle (i.e. no modulus wrt 2*pi)
        #   so we correct for this discrepancy here by keeping track of the number of full rotations.
        if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9: 
            self.full_rotations += 1  # Crossed 180deg in the CCW direction from +ve to -ve rad value so we add 2pi to keep it the equivalent positive value
        elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
            self.full_rotations -= 1 # Crossed 180deg in the CW direction from -ve to +ve rad value so we subtract 2pi to keep it the equivalent negative value

        psi = mocap_psi + 2*np.pi * self.full_rotations
        self.prev_mocap_psi = mocap_psi
        
        return psi

    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        print(f"Received odometry data: {msg=}")

        print("==" * 30)
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2] #+ (2.25 * self.sim) # Adjust z for simulation, new gazebo model has ground level at around -1.39m 

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        self.ax = msg.acceleration[0]
        self.ay = msg.acceleration[1]
        self.az = msg.acceleration[2]

        self.msg_quatval = msg.quaternion

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]


        self.roll, self.pitch, yaw = R.from_quat(msg.quaternion, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = self.adjust_yaw(yaw)  # Adjust yaw to account for full rotations
        r_final = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.rotation_object = r_final  # Store the final rotation object for further use


        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector_planar = np.array([self.y, self.z, self.vy, self.vz, self.roll])# px, py, h, v, theta = x
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        print(f"in odom, flat output: {self.output_vector}")
        if self.first_LQR:
            self.odom_counter += 1
            t00 = time.time()
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            # t0 = time.time()
            A, B = jitted_linearize_system(quad_sys_planar, self.rta_mm_gpr_state_vector_planar, self.hover_input_planar, noise)
            A, B = np.array(A), np.array(B)
            # print(f"Time to linearize system: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, Q_planar, R_planar)
            self.feedback_K = 1 * K
            # print(f"Time taken for LQR synthesis for K_feedback: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)  # Compute the LQR gain matrix
            self.reference_K = 1 * K  # Store the reference gain matrix
            # print(f"Time taken for LQR synthesis for K_reference: {time.time() - t0} seconds")

            
            self.last_lqr_update_time = time.time() - self.T0  # Set the last LQR update time to the current time
            print(f"Odom: time taken for entire LQR update: {time.time() - t00} seconds")

            # if self.odom_counter > 5:
            #     exit(0)
        ODOMETRY_DEBUG_PRINT = True
        if ODOMETRY_DEBUG_PRINT:
            # print(f"{self.full_state_vector=}")
            print(f"{self.nr_state_vector=}")
            # print(f"{self.flat_state_vector=}")
            print(f"{self.output_vector=}")
            print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")
            # print(f"{self.rotation_object.as_euler('xyz', degrees=True) = } (degrees)")
            # print(f"{self.ROT = }")
        # print("done")
        # exit(0)

    def rollout_callback(self):
        """Callback function for the rollout timer."""
        print(f"\nIn rollout callback at time: ", time.time() - self.T0)
        if self.begin_actuator_control - 1.0 <= self.time_from_start <= self.land_time:
            try:
                self.time_from_start = time.time() - self.T0
                t00 = time.time()  # Start time for rollout computation
                thresh = 1.0
                current_time = self.time_from_start
                current_state = self.rta_mm_gpr_state_vector_planar
                current_state_interval = irx.icentpert(current_state, self.x_pert)
                print(f"{current_time= }, {self.collection_time= }")

                if current_time >= self.collection_time:
                    print("Unsafe region begins now. Recomputing reachable tube and reference trajectory.")
                    t0 = time.time()  # Reset time for rollout computation
                    self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = jitted_rollout(
                        current_time, current_state_interval, current_state, self.feedback_K, self.reference_K, self.obs, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM
                    )
                    self.reachable_tube.block_until_ready()
                    self.rollout_ref.block_until_ready()
                    self.rollout_feedfwd_input.block_until_ready()
                    # print(f"Time taken by rollout: {time.time() - t0:.4f} seconds")

                    # t0 = time.time()  # Reset time for collection index computation
                    t_index = collection_id_jax(self.rollout_ref, self.reachable_tube, thresh)
                    t_index = int(t_index)
                    # print(f"Time taken for collection index computation: {time.time() - t0:.4f} seconds")

                    safety_horizon = t_index * self.tube_timestep
                    self.collection_time = current_time + safety_horizon  # Update the collection time based on the current time and index
                    print(f"{self.collection_time=}\n{safety_horizon=}")

                    self.traj_idx = 0
                    # print(f"{self.reachable_tube.shape = }")
                    self.save_tube = self.reachable_tube[::100]
                    # print(f"{self.save_tube.shape = }")
                    # print(f"{self.save_tube=  }")
                    # exit(0)

                else:
                    print("You're safe!")
                print(f"Time taken for whole rollout process: {time.time() - t00:.4f} seconds")


            except AttributeError as e:
                print("Ignoring missing attribute:", e)
                return
            except Exception as e:
                raise  # Re-raise all other types of exceptions            
        else:
            pass

    def vehicle_status_subscriber_callback(self, vehicle_status) -> None:
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status


    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        self.time_from_start = time.time() - self.T0
        print(f"In offboard callback at {self.time_from_start:.2f} seconds")

        if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
            print(f"Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_heartbeat_counter = 0
            return

        if self.time_from_start <= self.begin_actuator_control:
            publish_offboard_control_heartbeat_signal_position(self)
        elif self.time_from_start <= self.land_time:  
            publish_offboard_control_heartbeat_signal_body_rate(self)
        elif self.time_from_start > self.land_time:
            publish_offboard_control_heartbeat_signal_position(self)
        else:
            raise ValueError("Unexpected time_from_start value")

        if self.offboard_heartbeat_counter <= 10:
            if self.offboard_heartbeat_counter == 10:
                engage_offboard_mode(self)
                arm(self)
            self.offboard_heartbeat_counter += 1




    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""
        self.time_from_start = time.time() - self.T0
        if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
            print(f"Not in offboard mode.\n"
                  f"Current nav_state number: {self.vehicle_status.nav_state}\n"
                  f"nav_state number for offboard: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                  f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
            return

        if self.time_from_start <= self.begin_actuator_control:
            publish_position_setpoint(self, 0., 4.0, self.max_height, 0.0)

        elif self.time_from_start <= self.land_time:
            # f, M = self.control_administrator()
            # self.publish_force_moment_setpoint(f, M)
            self.control_administrator()

        elif self.time_from_start > self.land_time or (abs(self.z) <= 1.5 and self.time_from_start > 20):
            print("Landing...")
            publish_position_setpoint(self, 0.0, 0.0, -0.83, 0.0)
            if abs(self.x) < 0.2 and abs(self.y) < 0.2 and abs(self.z) <= 0.85:
                print("Vehicle is close to the ground, preparing to land.")
                land(self)
                disarm(self)
                exit(0)

        else:
            raise ValueError("Unexpected time_from_start value")

    def get_ref(self, time_from_start: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        ref_lqr_planar = jnp.array([y_des, z_des, vy_des, vz_des, roll_des])  # Reference position setpoint for planar LQR tracker (y, z, vy, vz, roll)
        ref_lqr_3D = jnp.array([x_des, y_des, z_des, vx_des, vy_des, vz_des, roll_des, pitch_des, yaw_des])
        ref_nr = jnp.array([x_des, y_des, z_des, yaw_des])  # Reference position setpoint for NR tracker (x, y, z, yaw)
        return ref_lqr_planar, ref_lqr_3D, ref_nr  

    def control_administrator(self) -> None:
        self.time_from_start = time.time() - self.T0
        print(f"\nIn control administrator at {self.time_from_start:.2f} seconds")
        ref_lqr_planar, ref_lqr_3D, ref_nr = self.get_ref(self.time_from_start)
        ctrl_T0 = time.time()

        NR_new_u, _ = NR_tracker_original(self.nr_state_vector, self.last_input, ref_nr, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        print(f"Time taken for NR tracker: {time.time() - ctrl_T0:.4f} seconds")
        # LQR_new_u_planar = self.lqr_administrator_planar(ref_lqr_planar, self.rta_mm_gpr_state_vector_planar, self.last_input[0:2], self.output_vector)  # Compute LQR control input for planar system
        # LQR_new_u_3D = self.lqr_administrator_3D(ref_lqr_3D, self.nr_state_vector, self.last_input, self.output_vector)  # Compute LQR control input
        rta_new_u_planar = self.rta_mm_gpr_administrator(ref_lqr_planar, self.rta_mm_gpr_state_vector_planar, self.last_input[0:2], self.output_vector)  # Compute RTA-MM GPR control input for planar system

        control_comp_time = time.time() - ctrl_T0 # Time taken for control computation
        print(f"\nEntire control Computation Time: {control_comp_time:.4f} seconds, Good for {1/control_comp_time:.2f}Hz control loop")


        print(f"{NR_new_u =}")
        # print(f"{LQR_new_u_planar =}")
        # print(f"{LQR_new_u_3D =}")  
        print(f"{rta_new_u_planar =}")

        # new_u = np.hstack([LQR_new_u_planar, NR_new_u[2:]])  # New control input from the LQR tracker
        new_u = jnp.hstack([rta_new_u_planar, NR_new_u[2:]])  # New control input from the RTA-MM GPR tracker
        # new_u = np.hstack([LQR_new_u, NR_new_u[2:]])  # New control input from the NR tracker
        # new_u = LQR_new_u  # Use the LQR control input directly
        print(f"{new_u = }")
        # exit(0)
        # if self.traj_idx > 11:
        #     print(f"Trajectory index {self.traj_idx} exceeded limit, stopping control.")
        #     exit(0)

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
                                    0., self.y_ref, self.z_ref, self.yaw_ref,
                                    new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate,
                                    ]
        # self.update_logged_data(state_input_ref_log_info)
        # for reach_set in self.save_tube:
        #     # print(f"{reach_set}")
        #     self.update_tube_data(reach_set)
            # exit(0)
        print("==" * 30)

    def update_lqr_feedback(self, sys, state, input, noise):
            print(f"\nUPDATING LQR")
            t0 = time.time()
            A, B = jitted_linearize_system(sys, state, input, noise)  # Linearize the system dynamics
            K, P, _ = control.lqr(A, B, Q_planar, R_planar)
            self.feedback_K = 1 * K

            K, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)  # Compute the LQR gain matrix
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

    def rta_mm_gpr_administrator(self, ref, state, input, output):
        """Run the RTA-MM administrator to compute the control inputs."""
        self.time_from_start = time.time() - self.T0 # Update time from start of the program
        print(f"\nIn RTA-MM GPR Administrator at {self.time_from_start=:.2f}")
        t0 = time.time()  # Start time for RTA-MM GPR computation

        if (self.time_from_start - self.last_lqr_update_time) >= 2.5 or self.first_LQR or abs(self.yaw) > self.max_yaw_stray:  # Re-linearize and re-compute the LQR gain X seconds
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            self.update_lqr_feedback(quad_sys_planar, state, input, noise)

        current_state = self.rta_mm_gpr_state_vector_planar
        current_state_interval = irx.icentpert(current_state, self.x_pert)
        applied_input = u_applied(current_state, self.rollout_ref[self.traj_idx, :], self.rollout_feedfwd_input[self.traj_idx, :], self.feedback_K)
        self.traj_idx += 1
        print(f"{self.traj_idx=}")
        self.y_ref = self.rollout_ref[self.traj_idx, 0]
        self.z_ref = self.rollout_ref[self.traj_idx, 1]
        self.vy_ref = self.rollout_ref[self.traj_idx, 2]
        self.vz_ref = self.rollout_ref[self.traj_idx, 3]
        self.yaw_ref = self.rollout_ref[self.traj_idx, 4]

        PRINT_RTA_DEBUG = True  # Set to True to print debug information for RTA-MM GPR
        if PRINT_RTA_DEBUG:
            print(f"{current_state=}")
            print(f"{current_state_interval=}")
            print(f"{self.rollout_ref[self.traj_idx, :] =}")
            print(f"{self.rollout_feedfwd_input[self.traj_idx, :] =}")
            print(f"{applied_input=}")

        print(f"Time taken for RTA-MM GPR computation: {time.time() - t0:.4f} seconds")
        return applied_input

    def get_throttle_command_from_force(self, collective_thrust) -> float: #Converts force to throttle command
        """ Convert the positive collective thrust force to a positive throttle command. """
        print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            try:
                motor_speed = m.sqrt(collective_thrust / (4.0 * self.THRUST_CONSTANT))
                throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
                return throttle_command
            except Exception as e:
                print(f"Error in throttle conversion: {e}")
                raise  # Raise the exception to ensure the error is handled properly

        if not self.sim: # I got these parameters from a curve fit of the throttle command vs collective thrust from the hardware spec sheet
            try:
                a = 0.00705385408507030
                b = 0.0807474474438391
                c = 0.0252575818743285
                throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c  # equation form is a*x + b*sqrt(x) + c = y
                return throttle_command
        
            except Exception as e:
                print(f"Error in throttle conversion (non-sim mode): {e}")
                raise  # Raise the exception to ensure the error is handled properly
        else:
            raise ValueError("self.sim must be True or False, not None")  # Ensure sim is set to True or False before calling this function
    


    # def lqr_administrator_3D(self, ref, state, input, output):
    #     if self.time_from_start % 1 == 0:  # Re-linearize and re-compute the LQR gain every X seconds
    #         A, B = jitted_linearize_system(quad_sys_3D, state, input, jnp.array([0.0]))  # Linearize the system dynamics
    #         K, P, _ = control.lqr(A, B, Q_3D, R_3D)
    #         self.feedback_K = 1 * K

    #     error = ref - state  # Compute the error between the reference and the current state
    #     nominal = self.feedback_K @ error
    #     nominalG = nominal + jnp.array([sim_constants.MASS * sim_constants.GRAVITY, 0.0, 0., 0.,])  # Add gravity compensation
    #     clipped = jnp.clip(nominalG, ulim_3D.lower, ulim_3D.upper)

    #     PRINT_LQR_DEBUG = False  # Set to True to print debug information for LQR
    #     if PRINT_LQR_DEBUG:
    #         print(f"\n\n{'=' * 60}")
    #         print(f"Linearized System Matrices:\n{A=}\n{B=}")
    #         print(f"LQR Gain Matrix:\n{K=}")
    #         print(f"Feedback Gain Matrix:\n{self.feedback_K}")
    #         print(f"{A.shape=}, {B.shape=}, {self.feedback_K.shape=}")
    #         print(f"Current State:\n{state=}")
    #         print(f"Reference:\n{ref=}")
    #         print(f"Error:\n{error}")

    #         print(f"Nominal Control Input (before clipping): {nominal}")
    #         print(f"Nominal Control Input with Gravity Compensation: {nominalG}")
    #         print(f"{ulim_3D.lower=}, {ulim_3D.upper=}")
    #         print(f"Clipped Control Input: {clipped}")
    #         print(f"{'=' * 60}\n\n")

    #     return clipped

# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])
        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])
        self.ctrl_comp_time_log.append(data[5])
        self.x_ref_log.append(data[6])
        self.y_ref_log.append(data[7])
        self.z_ref_log.append(data[8])
        self.yaw_ref_log.append(data[9])
        self.throttle_log.append(data[10])
        self.roll_rate_log.append(data[11])
        self.pitch_rate_log.append(data[12])
        self.yaw_rate_log.append(data[13])

    def update_tube_data(self, data):
        # print(f"Updating Tube Log:")
        self.save_tube_log.append(*data)
        # exit(0)



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