### Runtime Assurance with Mixed Monotonicity for Partially Unknown Systems with Gaussian Process Regressions for Quadrotor Hardware Deployment

A PX4 hardware-deployment-ready, JAX/immrax-based ROS 2 package for runtime assurance (RTA) of a quadrotor: a
reachable-tube certificate computed with mixed monotonicity around a reference trajectory, with the unknown wind
learned online by time-varying Gaussian processes. It builds on the numerical planar-quadrotor study of
[this paper](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf):

M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE
International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531,
doi: 10.1109/ICRA57147.2024.10611237.

See videos [here](https://gtvault-my.sharepoint.com/:f:/g/personal/egm9_gatech_edu/IgCTC42sLm-LSrAf6xMkPS_UAev1dNpcvioJ8PtQfRRZTEs?e=mRaYgF).

## Branches

| branch | what it is |
|---|---|
| **`main`** | the multithreaded Python node (this README) |
| **`cpp-version`** | `main` + a C++ fast loop: PX4 I/O, the 100 Hz control law and the certification watchdog in C++, with this Python node as the planner. Same features and tuning; faster and isolated by construction |
| tags `archive/*` | earlier branches, kept for reference: `archive/main-2026-05-02` (the single-threaded code the paper's hardware experiments were flown with), `archive/multithreaded`, `archive/cpp-fast-loop`, `archive/working_z`, `archive/working_last_y` |

## What the node does

* **Control:** Newton-Raphson tracker (pitch, yaw) + RTA feedback around the certified plan (thrust, roll rate),
  100 Hz body-rate setpoints to PX4 offboard.
* **Certification:** each rollout integrates an interval embedding of the planar dynamics with the GP wind
  bounds, and certifies the reference until the tube's **position** bounds (y, altitude) are more than 0.5 m from
  the reference position or the tube dips below a **ground floor** (0.3 m). Velocity and attitude bounds are not
  limited. The next rollout starts *before* the current certificate expires. If no certified plan exists for 20 ms,
  the node hands the vehicle to **PX4 LAND**.
* **Wind:** EKF on the acceleration residual at 40 Hz while the node's own commands fly. The y-wind is learned
  as a function of altitude and the z-wind as a function of lateral position. The GP mean is fed forward into the
  reference thrust.
* **Concurrency:** five callback groups on a `MultiThreadedExecutor` (or `EventsExecutor`). Rollouts can run in a
  worker process. Every in-flight JAX function is **ahead-of-time compiled** (it can never recompile mid-flight);
  rollouts stop at the certification point (3–7 ms instead of 60–190 ms).
* **Logging:** [flight_recorder](https://github.com/evannsmc/flight_recorder) (git submodule): one HDF5 file per
  flight, every plan stored once in full, autosaved every 5 s.

## Repository layout

```
docs/                       design notes and SITL results (Quarto sources + PDFs in docs/_output/)
packages/
  px4_rta_mm_gpr/           the ROS 2 package
    px4_rta_mm_gpr/         library: jax_mm_rta (model, GP, rollouts), jax_nr, concurrency, control_kernels,
                            flight_log, sim (numerical simulation), analysis (log analysis)
    scripts/                the node (rta_mm_gpr_node.py, px4_rta_mm_gpr.py) and data_analysis/ (notebooks)
    launch/                 sim_relay_rta_launch.py
  flight_recorder/          git submodule: flight-data logging library
tools/run_sitl.sh           PX4 SITL with this project's parameters
```

## Build and run

```bash
git clone --recurse-submodules https://github.com/evannsmc/PX4_RTA_MM_GPR.git   # into <workspace>/
sudo apt install libhdf5-dev python3-h5py
pip install --user immrax control  # see docs/02 for the numpy 1.26 / ROS Jazzy caveats
colcon build --symlink-install --base-paths PX4_RTA_MM_GPR/packages <other deps: px4_msgs, mocap_msgs, ...>
source install/setup.bash

PX4_RTA_MM_GPR/tools/run_sitl.sh                 # terminal 1: PX4 SITL + Gazebo x500 (HEADLESS=1 for no window)
MicroXRCEAgent udp4 -p 8888                      # terminal 2
ros2 launch px4_rta_mm_gpr sim_relay_rta_launch.py   # terminal 3 (or: ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --sim --log-file run.log)
```

`ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --help` lists every option. The recommended configuration is
`--executor events --rollout-backend process`.

Numerical simulation without ROS or PX4:

```python
from px4_rta_mm_gpr.sim import SimConfig, simulate, paper_winds
result = simulate(SimConfig(duration=20.0), winds=paper_winds(scale=1.2), log_path='sim.h5')
```

## Data analysis

`packages/px4_rta_mm_gpr/scripts/data_analysis/` has notebooks for flight logs and numerical simulations (summary and
timing, tubes, GP animations, numerical studies); see its README. The paper's **hardware** logs are in
`data_analysis/log_files/hardware/`, kept exactly as recorded. They come from the previous version of the code (tag
`archive/main-2026-05-02`).

## Documentation (`docs/_output/`)

1. `01_ros2_multithreading.pdf`: concurrency in ROS 2 / Python (executors, callback groups, GIL, processes, GC
   and JIT stalls)
2. `02_rta_node_changes.pdf`: the multithreaded node, its options, benchmarks, flight recorder
3. `03_compilation_rollouts_wind.pdf`: AOT compilation, early-exit rollouts, wind estimation, what caused the
   crashes
4. `04_altitude_and_ground.pdf`: GP feedforward (altitude offset), ground floor in the certificate, LAND backup

The C++ fast loop is documented on the `cpp-version` branch (`docs/_output/05_cpp_fast_loop.pdf`).
