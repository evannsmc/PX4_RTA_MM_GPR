### Runtime Assurance with Mixed Monotonicity for Partialy Unkown Systems with Gaussian Proccess Regressions for Quadrotor Hardware Deployment

See videos [here](https://gtvault-my.sharepoint.com/:f:/g/personal/egm9_gatech_edu/IgCTC42sLm-LSrAf6xMkPS_UAev1dNpcvioJ8PtQfRRZTEs?e=mRaYgF)

See data analysis and gifs of the hardware experiment data (reachable sets, planned trajectories, wind data, and true path flown) [here](scripts/data_analysis/log_files/hardware)


1. How to run:

```bash
ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --sim --log-file logxxxx.log

```


#### This is a PX4 Hardware-deployment-ready package based on code that runs the numerical planar quadrotor sim of this paper equipped and modified for hardware [pdf here](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf):

M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531, doi: 10.1109/ICRA57147.2024.10611237.
keywords: {Uncertain systems;Uncertainty;Runtime;Trajectory tracking;Gaussian processes;Trajectory;Electron tubes}
