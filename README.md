# PX4_RTA_MM_GPR


#To-Do
- wind estimate callback at 10Hz
    1. predicted output in y vs true y value 
    2. vehicle local position for acceleration
    3. vyd_pred - measured = -wz/m * cos(roll)
    4. vzd_pred - measured = wz/m * sin(r0ll)
    5. make them into observations at every height
    6. then when rollback is called it’ll use this self.obs variable
