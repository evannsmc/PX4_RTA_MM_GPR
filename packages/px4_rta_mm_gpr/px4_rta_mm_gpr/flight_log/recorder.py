"""RTA-MM-GPR flight log, built on the flight_recorder library (packages/flight_recorder, a git submodule).

flight_recorder does the generic work: preallocated column buffers (nothing for the GC to scan), one HDF5 file,
incremental flushing/autosave. This module only defines what an RTA-MM-GPR flight consists of:

    streams/ticks     one row per RTA control tick (TICK_COLUMNS)
    streams/wind      one row per wind-EKF update (WIND_COLUMNS)
    streams/gains     one row per LQR update, K matrices flattened row-major (GAIN_COLUMNS)
    records/plans/<seq>    every installed rollout plan, stored ONCE: reachable_tube, rollout_ref, feedfwd_input and
                           the inputs that produced it (state0, K_feedback, K_reference, obs_wy, obs_wz, goal);
                           attrs seq, t_start, dt, collection_time, violation_idx, compute_time, latency, warmup
    records/timing/<name>  loop periods / exec times, rollout compute / latency, GC pauses (arrays)
    events            e.g. backup_land
    root attrs        node options and tunables, git commit, host, date
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from flight_recorder import Recorder, git_commit

TICK_COLUMNS = (
    # time and state (NED, rad)
    'time', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'roll', 'pitch',
    # applied command
    'thrust', 'throttle', 'roll_rate', 'pitch_rate', 'yaw_rate',
    # timing
    'ctrl_comp_time', 'control_period',
    # wind estimate used by the GP
    'wy', 'wz',
    # which plan / which row of it this tick used
    'plan_seq', 'traj_idx', 'plan_age', 'plan_expired', 'rollout_latency', 'rollout_comptime',
    # reference row used by this tick: planar (py, pz, h, v, theta)
    'ref_py', 'ref_pz', 'ref_h', 'ref_v', 'ref_theta',
    # certified box at this tick's row (lower/upper of py and pz)
    'tube_py_lo', 'tube_pz_lo', 'tube_py_hi', 'tube_pz_hi',
)

WIND_COLUMNS = ('time', 'wy', 'wz', 'ay_meas', 'az_meas', 'ay_model', 'az_model', 'height_y', 'height_z')

GAIN_COLUMNS = ('time', 'first_lqr') + tuple(f'K_feedback_{i}{j}' for i in range(2) for j in range(5)) \
    + tuple(f'K_reference_{i}{j}' for i in range(2) for j in range(5))

PLAN_ARRAYS = ('reachable_tube', 'rollout_ref', 'feedfwd_input', 'state0', 'K_feedback', 'K_reference',
               'obs_wy', 'obs_wz', 'goal', 'delta')
PLAN_ATTRS = ('seq', 't_start', 'dt', 'collection_time', 'violation_idx', 'compute_time', 'latency', 'warmup')


class FlightRecorder:
    """The node's recorder. Same calls as before; storage is a flight_recorder.Recorder."""

    def __init__(self, metadata: Optional[dict] = None, tick_capacity: int = 60_000):
        meta = {'source': 'px4_rta_mm_gpr', 'git_commit': git_commit(os.path.dirname(os.path.realpath(__file__)))}
        meta.update(metadata or {})
        self.rec = Recorder(metadata={k: v for k, v in meta.items() if v is not None})
        self.ticks = self.rec.stream('ticks', TICK_COLUMNS, capacity=tick_capacity)
        self.wind = self.rec.stream('wind', WIND_COLUMNS, capacity=tick_capacity)
        self.gains = self.rec.stream('gains', GAIN_COLUMNS, capacity=4096)
        self.n_plans = 0
        self._last_tick_time = np.nan

    # ~~ hot path: one call per event ~~
    def tick(self, t, x, y, z, yaw, vx, vy, vz, roll, pitch, thrust, throttle, roll_rate, pitch_rate,
             yaw_rate, ctrl_comp_time, wy, wz, plan, traj_idx) -> None:
        ref = plan.rollout_ref[traj_idx]
        box = plan.reachable_tube[traj_idx]
        period = t - self._last_tick_time
        self._last_tick_time = t
        self.ticks.append(t, x, y, z, yaw, vx, vy, vz, roll, pitch,
                          thrust, throttle, roll_rate, pitch_rate, yaw_rate,
                          ctrl_comp_time, period, wy, wz,
                          plan.seq, traj_idx, t - plan.t_start, float(t > plan.collection_time),
                          plan.latency, plan.compute_time,
                          ref[0], ref[1], ref[2], ref[3], ref[4],
                          box[0], box[1], box[5], box[6])

    def wind_sample(self, t, wy, wz, ay_meas, az_meas, ay_model, az_model, height_y, height_z) -> None:
        self.wind.append(t, wy, wz, ay_meas, az_meas, ay_model, az_model, height_y, height_z)

    def gain_update(self, t, first_lqr: bool, K_feedback, K_reference) -> None:
        self.gains.append(t, float(first_lqr), *np.ravel(K_feedback), *np.ravel(K_reference))

    def event(self, t: float, kind: str, detail: str = '') -> None:
        self.rec.event(t, kind, detail)

    def add_plan(self, plan) -> None:
        """Store a plan once, by reference (RolloutPlan arrays are never mutated after construction)."""
        arrays = {k: getattr(plan, k) for k in PLAN_ARRAYS if getattr(plan, k, None) is not None}
        attrs = {k: getattr(plan, k) for k in PLAN_ATTRS}
        self.rec.record('plans', int(plan.seq), arrays, attrs)
        self.n_plans += 1

    # ~~ writing ~~
    @staticmethod
    def h5_path(path: str) -> str:
        return os.path.splitext(path)[0] + '.h5'

    def start_autosave(self, path: str, period: float) -> None:
        """Flush to ``path`` every ``period`` s during the flight, so a crash loses at most that much."""
        self.rec.start_autosave(period, self.h5_path(path))

    def save(self, path: str, timing: Optional[Dict[str, np.ndarray]] = None) -> str:
        for name, values in (timing or {}).items():
            self.rec.record('timing', name, {'values': np.asarray(values, dtype=float)})
        return self.rec.save(self.h5_path(path))
