"""Immutable rollout plan shared between the rollout thread (writer) and control thread (reader).

The writer builds a complete ``RolloutPlan`` and publishes it with a single reference
assignment; the reader grabs the reference once per control tick. Because a plan is never
mutated after construction, the reader can never observe half of an old plan and half of a
new one - no lock is needed on the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolloutPlan:
    t_start: float                # (s since node T0) time the rollout was computed from
    dt: float                     # (s) rollout time step
    reachable_tube: np.ndarray    # (N+1, 10)
    rollout_ref: np.ndarray       # (N+1, 5)  (py, pz, h, v, theta)
    feedfwd_input: np.ndarray     # (N, 2)    (thrust, roll rate)
    collection_time: float        # (s since T0) when this plan stops being certified safe
    compute_time: float           # (s) engine compute time
    latency: float                # (s) submit -> installed (includes queueing / IPC)
    save_tube: np.ndarray         # slice of the tube stored in the log for plotting
    tube_start: int               # first tube row stored in the log
    seq: int                      # monotonically increasing plan number
    # Inputs that produced this plan (logged so every plan can be reproduced offline)
    state0: np.ndarray = None     # planar state the rollout started from
    K_feedback: np.ndarray = None
    K_reference: np.ndarray = None
    obs_wy: np.ndarray = None     # GP data (t, height, wind) used for the y-wind GP
    obs_wz: np.ndarray = None
    violation_idx: int = -1       # first row where the tube left the threshold (-1: none)
    warmup: bool = False
    goal: np.ndarray = None       # goal state the rollout's reference was steered to
    delta: np.ndarray = None      # (m) position uncertainty (py, pz) the rollout used: box >= delta, threshold + delta

    def index_at(self, t: float) -> int:
        """Time-indexed lookup: the reference row that corresponds to time ``t``.

        Independent of how often the control loop actually ran since the plan was computed,
        so control-rate jitter or rollout latency cannot desynchronise the reference.
        """
        # + 1e-6 row: a tick exactly on a row time must not truncate to the previous row ((0.03 - 0) / 0.01 = 2.9999...)
        idx = int((t - self.t_start) / self.dt + 1e-6)
        return min(max(idx, 0), self.feedfwd_input.shape[0] - 1)
