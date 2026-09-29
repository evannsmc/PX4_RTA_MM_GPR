### Runtime Assurance with Mixed Monotonicity for Partially Unknown Systems with Gaussian Process Regressions for Quadrotor Hardware Deployment

A PX4 hardware-deployment-ready, JAX/immrax-based ROS 2 package for runtime assurance (RTA) of a quadrotor: a
reachable-tube certificate computed with mixed monotonicity around a reference trajectory, with the unknown wind
learned online by time-varying Gaussian processes. It builds on the numerical planar-quadrotor study of
[this paper](https://coogan.ece.gatech.edu/papers/pdf/cao2024tracking.pdf):

M. E. Cao and S. Coogan, "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics," 2024 IEEE
International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11525-11531,
doi: 10.1109/ICRA57147.2024.10611237.

See videos [here](https://gtvault-my.sharepoint.com/:f:/g/personal/egm9_gatech_edu/IgCTC42sLm-LSrAf6xMkPS_UAev1dNpcvioJ8PtQfRRZTEs?e=mRaYgF).

## Baseline and main result

**Baseline (the default):** time-varying GP (**TV-GPR**) + the paper's embedding system **(66)-(67)**, a
0.25 m spatial certificate, and a 5 cm model-mismatch margin on the tube's starting box. In PX4 SITL:
the vehicle never left its certified tube (0 % of the time), no LAND backup was needed, certified
0.34 s ahead, tracking 13 / 16 mm, one rollout in about 4 ms.

**Main takeaways**

1. **TV-GPR over GPR.** When the wind drifts over time (numerical simulation with the paper's evolving
   wind; the SITL world has none), a time-invariant GP is confidently wrong: the
   vehicle leaves its "certified" tube about **40 %** of the time, by up to 0.29 m (more than the 0.25 m
   threshold). With TV-GPR it **never** does (worst 1 mm), whatever the embedding system.
2. **The embedding systems are equivalent here.** With TV-GPR, (68)-(69), (66)-(67) and (64)-(65) certify
   the same horizon (within 20 ms of 0.6 s) with the same tubes (within 2.5 %); (66)-(67) is the cheapest,
   so it is the default. The paper's expected nesting (68) ⊆ (66) ⊆ (64) does not show up for this vehicle
   (constant input matrix, GP uncertainty dominating the tube's growth); see the comparison below.
3. **A 5 cm model-mismatch margin makes SITL match the model.** Without it the real PX4 vehicle leaves the
   tube in the first moments of each plan (about 20 % of the time); with it, 0 %.

## The comparison: TV-GPR vs GPR × three embedding systems

The certified tube is only as good as its disturbance model. We computed the tube six ways, the
**time-varying GP (TV-GPR)** of this package and a **time-invariant GP (GPR)**, each with the paper's three
embedding systems (Appendix A), all at the same certificate (position within 0.25 m of the reference,
0.3 m floor), and measured how often the vehicle actually **left** its certified tube:

![Escapes from the certified tube](packages/px4_rta_mm_gpr/scripts/data_analysis/figures/comparison_escapes.png)

**Numerical simulation, drifting wind** (the paper's evolving wind; exact model; 9 wind runs per variant):

| embedding system | **TV-GPR** (this package) | GPR |
|---|---|---|
| (68)-(69): first order in *u* and *w* | **A: 0 %** outside, worst 0.9 mm (the paper's) | D: 41 %, worst 0.10 m |
| (66)-(67): first order in *u* | **E: 0 %**, worst 0.9 mm (**baseline**) | C: 40 %, worst 0.29 m |
| (64)-(65): no first-order terms | **F: 0.03 %**, worst 1.2 mm | B: 40 %, worst 0.28 m |

**PX4 SITL** (3 flights per variant; the Gazebo world has **no wind**, so the GPs only learn a steady
thrust offset, and SITL tests the model's mismatch with the real vehicle, not the wind model):

| | A | E | F | D | C | B |
|---|---|---|---|---|---|---|
| wind model + embedding | TV + (68) | **TV + (66), baseline** | TV + (64) | GPR + (68) | GPR + (66) | GPR + (64) |
| outside the tube, δ = 0 | 19 % | 20 % | 20 % | 65 % | 67 % | 71 % |
| **outside the tube, 5 cm margin (default)** | **0 %** | **0 %** | **0 %** | **0 %** | **0 %** | **0.02 %** |
| certified horizon, 5 cm margin | **0.34 s** | **0.34 s** | **0.34 s** | 0.17 s | 0.29 s | 0.29 s |
| tracking RMSE lateral / vertical, 5 cm margin | 13 / 18 mm | 13 / 16 mm | 12 / 15 mm | 7 / 9 mm | 11 / 12 mm | 12 / 11 mm |
| one rollout, 5 cm margin | 3.9 ms | 3.8 ms | 4.0 ms | 3.5 ms | 3.3 ms | 3.4 ms |

* **A static GP looks better and is wrong when the wind drifts.** With every embedding system its tubes are
  about 10× thinner and certify further ahead, but it is confidently wrong about a time-varying wind: in the
  numerical runs the true state is outside its "certified" tube about 40 % of the time, by up to 0.29 m,
  more than the 0.25 m threshold itself. **With the time-varying GP the certificate holds**, with every
  embedding system (worst 0.9-1.2 mm, from the 10 ms integration step).
* **The embedding system barely matters once the GP is right.** Rolled out from identical inputs, the
  three TV-GPR tubes certify within 20 ms of each other (A 0.597 s, E 0.598 s, F 0.599 s) and differ by
  under 2.5 % in width. The expected nesting (68) ⊆ (66) ⊆ (64) does not hold here: (66)-(67) bounds the GP
  mean by its exact range over the box, which on its own is tighter than the mean-value form of (68)-(69),
  and the GP's growing uncertainty dominates both; and the input matrix is a constant selector, so
  (64)-(65) equals (66)-(67) and gains a little from clipping to the actuator limits.
* **In SITL the planar model is the limit, and a 5 cm margin fixes it.** With δ = 0 every variant leaves
  its tube in the first moments of each plan (the thinner GPR tubes more often); with the default 5 cm
  model-mismatch margin all six stay inside, and the TV-GPR variants certify the longest (the GPR variants
  replan up to 1.6× as often, which re-anchors the reference and lowers their tracking error). None of the 36
  SITL flights needed the LAND backup. SITL cannot show the static GP's failure yet: that needs a
  time-varying wind in Gazebo.

![Six tubes from the same start](packages/px4_rta_mm_gpr/scripts/data_analysis/figures/comparison_tubes.png)

Setup: numerical, 10 cases per variant (calm, paper winds ×0.3 / ×0.6 / ×1.0 with 3 seeds; backup off, so
every variant flies the full mission; δ = 0, the model is exact); SITL, 3 flights per variant flown twice
(δ = 0 and the default 5 cm margin; backup on). All tables, the GP × embedding grids, the two-way effect
split and per-flight data:
[`05_embedding_comparison.ipynb`](packages/px4_rta_mm_gpr/scripts/data_analysis/05_embedding_comparison.ipynb).
Reproduce: `--gp tv|static --embedding uw|u|none --model-mismatch-margin 0.05` (node) or
`SimConfig(gp_epsilon=..., embedding=...)`.

## Branches

| branch | what it is |
|---|---|
| **`main`** | the multithreaded Python node |
| **`cpp-version`** (this branch) | `main` + a C++ fast loop: PX4 I/O, the 100 Hz control law and the certification watchdog in C++, with this Python node as the planner. Same features and tuning; faster and isolated by construction |
| tags `archive/*` | earlier branches, kept for reference: `archive/main-2026-05-02` (the single-threaded code the paper's hardware experiments were flown with), `archive/multithreaded`, `archive/cpp-fast-loop`, `archive/working_z`, `archive/working_last_y` |

## The C++ fast loop (this branch)

`cpp-version` is exactly `main` plus one layer (a single commit, so `main` merges into it cleanly):

* `packages/px4_rta_mm_gpr_cpp/`: **`rta_fast_loop`**, which owns everything PX4 sees: heartbeat, arming and
  modes, the 100 Hz control law (NR tracker with an exact dual-number Jacobian + RTA feedback, matching the Python/JAX
  kernels to ~1e-13) and the certification watchdog (PX4 LAND when no certified plan exists). It computes a control
  tick in about 4 µs and holds a 10.04 ms p99 period. It also lands the vehicle by itself if the planner dies.
* `packages/px4_rta_mm_gpr_msgs/`: `PlannerStatus` (mission clock + every tunable: one source of truth),
  `RtaPlan`, `RtaGains`, `ControlTick`.
* The Python node with `--cpp-control` becomes the **planner**: rollouts, GP, wind EKF, LQR gains, and logging
  of every `ControlTick` into the same flight log.

```bash
ros2 launch px4_rta_mm_gpr cpp_rta_launch.py            # rta_fast_loop + planner
```

## What the node does

* **Control:** Newton-Raphson tracker (pitch, yaw) + RTA feedback around the certified plan (thrust, roll rate),
  100 Hz body-rate setpoints to PX4 offboard.
* **Certification:** each rollout integrates an interval embedding of the planar dynamics with the GP wind
  bounds (the paper's embedding system (66)-(67) by default; `--embedding uw|u|none`), and certifies the reference until the tube's **position** bounds (y, altitude) are more than 0.25 m + δ from
  the reference position or the tube dips below a **ground floor** (0.3 m). δ = max(model-mismatch margin 0.05 m,
  position-estimate uncertainty: 0 in SITL, 3σ of EKF2 on hardware); it also widens the tube's initial box. Velocity and
  attitude bounds are not limited. The next rollout starts *before* the current certificate expires. If no certified plan exists for 20 ms,
  the node hands the vehicle to **PX4 LAND**.
* **Wind:** EKF on the acceleration residual at 100 Hz while the node's own commands fly. The y-wind is learned
  as a function of altitude and the z-wind as a function of lateral position. The GP mean is fed forward into the
  reference thrust.
* **Concurrency:** five callback groups on a `MultiThreadedExecutor` (or `EventsExecutor`). Rollouts can run in a
  worker process. Every in-flight JAX function is **ahead-of-time compiled** (it can never recompile mid-flight);
  rollouts stop at the certification point (3–7 ms instead of 60–190 ms).
* **Logging:** [flight_recorder](https://github.com/evannsmc/flight_recorder) (git submodule): one HDF5 file per
  flight, every plan stored once in full, autosaved every 5 s.

## Repository layout

```
docs/                       design notes and SITL results (Quarto sources + PDFs in docs/_output/)
packages/
  px4_rta_mm_gpr/           the ROS 2 package
    px4_rta_mm_gpr/         library: jax_mm_rta (model, GP, rollouts), jax_nr, concurrency, control_kernels,
                            flight_log, sim (numerical simulation), analysis (log analysis)
    scripts/                the node (rta_mm_gpr_node.py, px4_rta_mm_gpr.py) and data_analysis/ (notebooks)
    launch/                 sim_rta_launch.py, cpp_rta_launch.py
  px4_rta_mm_gpr_cpp/       rta_fast_loop (C++), control_law.hpp, control_law_check, test/
  px4_rta_mm_gpr_msgs/      PlannerStatus, RtaPlan, RtaGains, ControlTick
  flight_recorder/          git submodule: flight-data logging library
tools/run_sitl.sh           PX4 SITL with this project's parameters
```

## Build and run

```bash
git clone --recurse-submodules https://github.com/evannsmc/PX4_RTA_MM_GPR.git   # into <workspace>/
sudo apt install libhdf5-dev python3-h5py
pip install --user immrax control imageio-ffmpeg  # see docs/02 for the numpy 1.26 / ROS Jazzy caveats
colcon build --symlink-install --base-paths PX4_RTA_MM_GPR/packages <other deps: px4_msgs>
source install/setup.bash

PX4_RTA_MM_GPR/tools/run_sitl.sh                 # terminal 1: PX4 SITL + Gazebo x500 (HEADLESS=1 for no window)
MicroXRCEAgent udp4 -p 8888                      # terminal 2
ros2 launch px4_rta_mm_gpr cpp_rta_launch.py         # terminal 3: C++ fast loop + Python planner
# (Python-only, as on main: ros2 launch px4_rta_mm_gpr sim_rta_launch.py)
```

`ros2 run px4_rta_mm_gpr px4_rta_mm_gpr --help` lists every option. The recommended configuration is
`--executor events --rollout-backend process`.

Numerical simulation without ROS or PX4:

```python
from px4_rta_mm_gpr.sim import SimConfig, simulate, paper_winds
result = simulate(SimConfig(duration=20.0), winds=paper_winds(scale=1.2), log_path='sim.h5')
```

## Data analysis

`packages/px4_rta_mm_gpr/scripts/data_analysis/` has notebooks for flight logs and numerical simulations (summary and
timing, tubes, videos of every plan's reachable tube with the GPs, numerical studies); see its README. The paper's **hardware** logs are in
`data_analysis/log_files/hardware/`, kept exactly as recorded. They come from the previous version of the code (tag
`archive/main-2026-05-02`).

## Documentation (`docs/_output/`)

0. **`00_package_guide.pdf` (start here)**: how the package is laid out and how its parts connect, ahead-of-time
   JAX compilation, the threads / processes / locks, and every import explained
1. `01_ros2_multithreading.pdf`: concurrency in ROS 2 / Python (executors, callback groups, GIL, processes, GC
   and JIT stalls)
2. `02_rta_node_changes.pdf`: the multithreaded node, its options, benchmarks, flight recorder
3. `03_compilation_rollouts_wind.pdf`: AOT compilation, early-exit rollouts, wind estimation, what caused the
   crashes
4. `04_altitude_and_ground.pdf`: GP feedforward (altitude offset), ground floor in the certificate, LAND backup
5. `05_cpp_fast_loop.pdf`: the C++ fast loop: architecture, messages, dual-number Jacobian, equivalence with the
   Python kernels, SITL results, planner-killed test
6. `06_fixes_and_comparison.pdf`: five model fixes (feedback sign, Jacobian domain, row lookup, body-frame
   velocities, 100 Hz state) and the TV-GPR / GPR × embedding-system comparison
