# Data analysis

Notebooks for flight logs of the current code, and for the numerical (no ROS / no PX4) simulations.

| notebook | what it shows |
|---|---|
| `01_flight_summary.ipynb` | overview table of one or more flights; timing (control period, rollout compute, plan age); tracking errors and RMSE; time series with the certified tube row; wind estimates; events (e.g. LAND backup) |
| `02_flight_static.ipynb` | y-altitude path with the certified tubes of sampled plans, goal and ground floor; certified horizon per plan; tube width over time; one plan in detail |
| `03_flight_animation.ipynb` | GIF: vehicle, path, current certified tube and reference, and the y-/z-wind GPs exactly as each plan used them |
| `04_numerical_sim.ipynb` | closed-loop numerical simulations built from the node's own components (`px4_rta_mm_gpr.sim`): wind scenarios, paths, animation with the true wind, early-exit vs full-horizon rollouts, GP data collection strategies |

All four use `px4_rta_mm_gpr.analysis` (loading, summaries, tube and GP reconstruction, plots, animation), so a
notebook stays a few lines per figure. Figures go to `figures/` (PDF with TrueType fonts, and GIFs).

**Data.**
* `log_files/sitl/`: an example SITL flight of the current code (HDF5, flight_recorder layout). Your own runs are
  written by the node to `<workspace>/src/data_analysis/log_files/px4_rta_mm_gpr/<name>.h5`; point `LOGS` in a
  notebook at them.
* `log_files/numerical/`: written by `04_numerical_sim.ipynb` (same layout as a flight, so 01-03 work on them too).
* `log_files/hardware/`: the paper's hardware flights, recorded with the **previous version of the code** (tag
  `archive/main-2026-05-02`); kept as recorded, see its README.
* The previous analysis notebooks and SITL logs (old CSV layout) are not on this branch; they are in tag
  `archive/main-2026-05-02` under `scripts/data_analysis/`.

**Running.** From a sourced workspace (`source install/setup.bash`), start Jupyter in this directory. To re-run all
notebooks non-interactively:

```bash
for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```
