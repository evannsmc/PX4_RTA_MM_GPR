# Python planner -> C++ fast loop, 10 Hz. The planner is the single source of truth for the mission clock and
# for every tunable the control law uses; the fast loop does not start the mission before the first message.

float64 t0_epoch                 # (s, system clock) mission time = now - t0_epoch
bool sim

float64 mass                     # (kg)
float64 thrust_min               # (N) RTA thrust limits
float64 thrust_max
float64 roll_rate_max            # (rad/s) RTA roll-rate limit (symmetric)
float64 nr_rate_limit            # (rad/s) anti-windup bound on the NR pitch/yaw rates; +inf = off

float64 t_lookahead              # (s) NR tracker prediction horizon
float64 lookahead_step           # (s) NR forward-Euler step
float64 integration_step         # (s) NR input integration step (= control period)

float64 begin_actuator_control   # (s, mission time) RTA phase start
float64 land_time                # (s, mission time) RTA phase end
float64 max_y                    # (m) pre-RTA waypoint (NED)
float64 max_height

bool nr_ref_from_plan            # NR y/z reference = plan at t + t_lookahead
string backup                    # 'land' | 'none'
float64 backup_grace             # (s) continuous time without a certified plan before the backup engages
