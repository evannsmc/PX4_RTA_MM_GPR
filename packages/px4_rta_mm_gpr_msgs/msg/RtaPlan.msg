# Python planner -> C++ fast loop: one certified rollout plan (only the rows that were computed).

uint32 seq
bool warmup                      # pre-RTA plan, never flown
float64 t_start                  # (s, mission time) time of row 0
float64 dt                       # (s) row spacing
float64 collection_time          # (s, mission time) end of the certified part
int32 violation_idx              # first uncertified row (-1: none within the horizon)
uint32 n_rows
float64[] rollout_ref            # n_rows x 5, row-major: (py, pz, h, v, theta)
float64[] feedfwd_input          # (n_rows - 1) x 2, row-major: (thrust N, roll rate)
