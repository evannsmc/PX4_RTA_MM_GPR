"""Low-overhead flight recorder: preallocated NumPy buffers in flight, one HDF5 file at shutdown.

Why not per-tick Python lists (the ROS2Logger approach)?
  * every appended list/tuple is a new object the cyclic garbage collector must track and scan;
    in this node that made full collections take ~300 ms (every thread frozen), and
  * the reachable tube was re-appended (12 rows x 4 values) on EVERY control tick, although it only
    changes when a new rollout plan is installed.

Here each control tick writes one row into a preallocated float64 array (no allocation), and every
rollout plan is stored exactly once, in full precision, together with everything that went into it
(initial state, gains, GP data), so any plan can be reproduced offline.

File layout (``h5dump -n flight.h5``)::

    /                       attrs: metadata (options, tunables, git commit, host, date)
    /ticks/<column>         one row per RTA control tick
    /wind/<column>          one row per wind-estimator tick
    /gains/<column>         one row per LQR update (K matrices flattened row-major)
    /plans/<seq>/           reachable_tube (N+1,10), rollout_ref (N+1,5), feedfwd_input (N,2),
                            state0, K_feedback, K_reference, obs_wy, obs_wz; attrs t_start, dt,
                            collection_time, violation_idx, compute_time, latency, warmup
    /timing/<name>          raw loop periods / exec times, rollout compute/latency, GC pauses
"""
from __future__ import annotations

import datetime
import os
import socket
import subprocess
import threading
from typing import Dict, Iterable, List, Optional

import numpy as np


class ColumnBuffer:
    """Append-only table of float64 columns backed by one preallocated 2-D array.

    ``append`` writes a row in place (amortised O(1), grows by doubling), so the hot path creates
    no Python containers. Each buffer must have a single writer thread.
    """

    def __init__(self, columns: Iterable[str], capacity: int = 1024):
        self.columns: List[str] = list(columns)
        self._index: Dict[str, int] = {c: i for i, c in enumerate(self.columns)}
        self._data = np.full((capacity, len(self.columns)), np.nan)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def append(self, *values: float) -> None:
        if self._n == self._data.shape[0]:
            grown = np.full((2 * self._data.shape[0], self._data.shape[1]), np.nan)
            grown[:self._n] = self._data
            self._data = grown
        self._data[self._n] = values
        self._n += 1

    def column(self, name: str) -> np.ndarray:
        return self._data[:self._n, self._index[name]]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {c: self._data[:self._n, i].copy() for i, c in enumerate(self.columns)}


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


class FlightRecorder:
    def __init__(self, metadata: Optional[dict] = None, tick_capacity: int = 60_000):
        self.metadata = dict(metadata or {})
        self.metadata.setdefault('created', datetime.datetime.now().isoformat(timespec='seconds'))
        self.metadata.setdefault('host', socket.gethostname())
        self.metadata.setdefault('git_commit', _git_commit())
        self.ticks = ColumnBuffer(TICK_COLUMNS, tick_capacity)
        self.wind = ColumnBuffer(WIND_COLUMNS, 4096)
        self.gains = ColumnBuffer(GAIN_COLUMNS, 1024)
        self.plans: list = []            # RolloutPlan objects (immutable; stored by reference, no copy)
        self._plans_lock = threading.Lock()
        self._last_tick_time = np.nan

    # ~~ hot-path API: one call per event, no allocation beyond the argument tuple ~~
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

    def add_plan(self, plan) -> None:
        with self._plans_lock:
            self.plans.append(plan)

    # ~~ shutdown ~~
    def save(self, path: str, timing: Optional[Dict[str, np.ndarray]] = None) -> str:
        import h5py  # imported lazily: only needed at shutdown
        path = os.path.splitext(path)[0] + '.h5'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        comp = dict(compression='gzip', compression_opts=4, shuffle=True)
        with h5py.File(path, 'w') as f:
            for key, value in self.metadata.items():
                f.attrs[key] = _attr(value)
            for group_name, buf in (('ticks', self.ticks), ('wind', self.wind), ('gains', self.gains)):
                g = f.create_group(group_name)
                for name, col in buf.as_dict().items():
                    g.create_dataset(name, data=col, **(comp if col.size > 64 else {}))
            plans = f.create_group('plans')
            with self._plans_lock:
                saved = list(self.plans)
            for plan in saved:
                g = plans.create_group(f'{plan.seq:05d}')
                for name in ('reachable_tube', 'rollout_ref', 'feedfwd_input'):
                    g.create_dataset(name, data=getattr(plan, name), **comp)
                for name in ('state0', 'K_feedback', 'K_reference', 'obs_wy', 'obs_wz'):
                    g.create_dataset(name, data=np.asarray(getattr(plan, name)))
                for name in ('seq', 't_start', 'dt', 'collection_time', 'violation_idx', 'compute_time',
                             'latency', 'warmup'):
                    g.attrs[name] = getattr(plan, name)
            if timing:
                g = f.create_group('timing')
                for name, values in timing.items():
                    g.create_dataset(name, data=np.asarray(values, dtype=float))
        return path


def _attr(value):
    if isinstance(value, (str, int, float, bool, np.number)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    return str(value)


def _git_commit() -> str:
    try:
        here = os.path.dirname(os.path.realpath(__file__))
        return subprocess.run(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, timeout=2).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'
