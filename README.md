### Runtime Assurance with Mixed Monotonicity for Partialy Unkown Systems with Gaussian Proccess Regressions for PX4 Quadrotor Hardware Deployment

1. How to run:

```bash
ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --sim --log-file logxxxx.log

```

#### To-Do

- wind estimate callback at 10Hz
  1. predicted output in y vs true y value
  2. vyd_pred - measured = -wz/m \* cos(roll)
  3. vzd_pred - measured = wz/m \* sin(r0ll)
  4. make them into observations at every height
  5. then when rollback is called it’ll use this self.obs variable

#### This is a PX4 Hardware-deployment-ready package based on [my version](https://github.com/evannsm/RTA_MM_GPR_QUAD) of the JAX-ified code code that runs the numerical planar quadrotor sim of this paper equipped and modified for hardware [pdf here](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf):

M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531, doi: 10.1109/ICRA57147.2024.10611237.
keywords: {Uncertain systems;Uncertainty;Runtime;Trajectory tracking;Gaussian processes;Trajectory;Electron tubes}
