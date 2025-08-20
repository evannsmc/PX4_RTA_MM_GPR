### Runtime Assurance with Mixed Monotonicity for Partialy Unkown Systems with Gaussian Proccess Regressions for PX4 Quadrotor Hardware Deployment

#### To-Do
- wind estimate callback at 10Hz
    1. predicted output in y vs true y value 
    2. vehicle local position for acceleration
    3. vyd_pred - measured = -wz/m * cos(roll)
    4. vzd_pred - measured = wz/m * sin(r0ll)
    5. make them into observations at every height
    6. then when rollback is called it’ll use this self.obs variable


#### My Version of the JAX-ified version of the code that runs the numerical planar quadrotor sim of this paper [pdf here](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf): 
M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531, doi: 10.1109/ICRA57147.2024.10611237.
keywords: {Uncertain systems;Uncertainty;Runtime;Trajectory tracking;Gaussian processes;Trajectory;Electron tubes}
