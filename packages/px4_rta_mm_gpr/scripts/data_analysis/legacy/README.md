# Legacy analysis (previous version of the code)

Kept for reference; superseded by the notebooks one level up.

| here | what it was | replaced by |
|---|---|---|
| `log_files/sim/*.log` | SITL logs of the previous single-threaded node (ROS2Logger CSV) | new logs are HDF5 (`../log_files/sitl/`) |
| `DataAnalysis.ipynb`, `plot_error_tables.ipynb` | timing and RMSE from the CSV logs | `../01_flight_summary.ipynb` |
| `plot_log_static.ipynb` | path with reference horizons and tube slices | `../02_flight_static.ipynb` |
| `plot_log_animated.ipynb`, `plot_log_forward_horizon.ipynb`, `plot_log_with_gp_anim.ipynb` | animations (GPs refit from logged wind) | `../03_flight_animation.ipynb` |
| `rta_evanns.ipynb`, `rta_evanns_GPR2_ANIM.ipynb` | the original numerical studies, with their own inline model | `../04_numerical_sim.ipynb` (`px4_rta_mm_gpr.sim`) |
| `utilities.py`, `*.gif`, `*.png`, `example_frame.pdf`, `saved_frames/` | helpers and outputs of the above | – |

These notebooks read the old CSV layout (its "N reference rows per tick" padding) and use the model of the previous
code: the numerical studies use a unit-mass model with a single horizontal wind, and some notebooks import `TVGPR`
from a local path. They run against the code of tag **`archive/main-2026-05-02`**.
