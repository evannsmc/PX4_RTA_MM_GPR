### Runtime Assurance with Mixed Monotonicity for Partialy Unkown Systems with Gaussian Proccess Regressions for Quadrotor Hardware Deployment

See videos [here](https://gtvault-my.sharepoint.com/:f:/g/personal/egm9_gatech_edu/IgCTC42sLm-LSrAf6xMkPS_UAev1dNpcvioJ8PtQfRRZTEs?e=mRaYgF)

See data analysis and gifs of the hardware experiment data (reachable sets, planned trajectories, wind data, and true path flown) [here](packages/px4_rta_mm_gpr/scripts/data_analysis/log_files/hardware)


**Branch `cpp-fast-loop`:** clone with `git clone --recurse-submodules` (or run `git submodule update --init`):
flight logging uses [flight_recorder](https://github.com/evannsmc/flight_recorder) as a submodule in
`packages/flight_recorder` (needs `libhdf5-dev` and `python3-h5py`).

This is a multi-package repository. The Python node lives in `packages/px4_rta_mm_gpr/`;
`packages/px4_rta_mm_gpr_cpp/` holds the C++ fast loop (PX4 I/O, 100 Hz control law, certification watchdog) and
`packages/px4_rta_mm_gpr_msgs/` the messages between them. Build with
`colcon build --symlink-install --base-paths PX4_RTA_MM_GPR/packages ...` and fly with
`ros2 launch px4_rta_mm_gpr cpp_relay_rta_launch.py`. Design notes and SITL results: `docs/_output/*.pdf` (01-05).

1. How to run:

```bash
ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --sim --log-file logxxxx.log

```


#### This is a PX4 Hardware-deployment-ready and JAX/Immrax-infused package based on code that runs the numerical planar quadrotor simulation [in this paper](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf):

M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531, doi: 10.1109/ICRA57147.2024.10611237.
keywords: {Uncertain systems;Uncertainty;Runtime;Trajectory tracking;Gaussian processes;Trajectory;Electron tubes}
