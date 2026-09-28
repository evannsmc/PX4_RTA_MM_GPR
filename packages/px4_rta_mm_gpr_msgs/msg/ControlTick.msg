# C++ fast loop -> Python planner, every control tick: what was computed and sent (logged by the planner).

uint8 PHASE_WAIT=0               # waiting for the planner / offboard
uint8 PHASE_POSITION=1           # pre-RTA position setpoints
uint8 PHASE_RTA=2                # RTA body-rate control
uint8 PHASE_LANDING=3            # end-of-mission landing setpoints
uint8 PHASE_BACKUP=4             # backup (PX4 LAND) engaged

float64 t                        # (s, mission time)
uint8 phase
float64 control_period           # (s) since the previous tick
float64 ctrl_comp_time           # (s) control-law computation

float64[9] nr_state              # x y z vx vy vz roll pitch yaw (yaw unwrapped)
float64[5] planar_state          # py pz h v theta
float64[4] u                     # thrust (N), roll, pitch, yaw rate (rad/s)
float64 throttle

uint32 plan_seq
int32 traj_idx
bool plan_expired
bool backup_engaged
string backup_reason
