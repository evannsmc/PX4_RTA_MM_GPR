"""Rollout (reachable tube + reference trajectory) computation backends.

The RTA rollout is by far the most expensive computation in the node (a 3000-step
``jax.lax.scan`` over an interval embedding system with two GPs). This module isolates it
behind a tiny request/result interface so the ROS node never has to care *where* it runs:

    backend.submit(request)      # hand off a snapshot of the inputs
    result = backend.poll()      # None until a result is ready

Two backends are provided:

* ``ThreadRolloutBackend``  - computes synchronously inside the calling thread. Combined with a
  ``MultiThreadedExecutor`` and a dedicated callback group, the rollout runs in its own executor
  thread. XLA releases the GIL while the compiled rollout executes, so the control thread keeps
  running in parallel.
* ``ProcessRolloutBackend`` - computes in a separate OS process (its own interpreter, its own
  GIL, its own JAX runtime). ``submit`` and ``poll`` never block, so even a single-threaded
  executor keeps its control loop running while a rollout is in flight.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RolloutConfig:
    """Everything the rollout needs that does NOT change during a flight.

    Plain Python/NumPy values only, so it can be pickled and sent to a worker process, which
    then rebuilds the JAX/immrax objects (system, mixed Jacobian, permutation, input limits).
    """
    mass: float
    horizon: float                # (s) reachable tube horizon
    timestep: float               # (s) rollout integration step
    ulim_lower: tuple
    ulim_upper: tuple
    goal_state: tuple             # (py, pz, h, v, theta)
    x_pert: tuple                 # half-widths of the initial state interval
    collection_threshold: float   # tube-vs-reference deviation that ends the safety horizon


@dataclass(frozen=True)
class RolloutRequest:
    """A snapshot of the time-varying rollout inputs, taken at submission time."""
    t_start: float                # (s since node T0) time the rollout starts from
    state: np.ndarray             # planar state (py, pz, h, v, theta)
    K_feedback: np.ndarray
    K_reference: np.ndarray
    obs_wy: np.ndarray            # GP data (t, height, wind force) for the y-direction wind
    obs_wz: np.ndarray            # GP data (t, height, wind force) for the z-direction wind
    warmup: bool = False          # pre-control rollouts do not move the collection time
    submitted_at: float = 0.0     # time.time() at submission (to measure end-to-end latency)


@dataclass(frozen=True)
class RolloutResult:
    request: RolloutRequest
    reachable_tube: np.ndarray    # (N+1, 10) lower/upper embedding states
    rollout_ref: np.ndarray       # (N+1, 5) reference trajectory
    feedfwd_input: np.ndarray     # (N, 2) feedforward inputs
    violation_idx: int            # first index where tube deviates > threshold, -1 if none
    compute_time: float           # (s) pure compute time inside the engine


class RolloutEngine:
    """Owns the JAX objects for the rollout and turns requests into results.

    Used directly by the thread backend and, inside the worker process, by the process backend.
    """

    def __init__(self, config: RolloutConfig):
        # Imported here so that a spawned worker process initialises JAX itself.
        import jax.numpy as jnp
        import immrax as irx
        from px4_rta_mm_gpr.jax_mm_rta import (
            jitted_rollout, collection_id_jax, PlanarMultirotorTransformed)

        self._jnp = jnp
        self._irx = irx
        self._jitted_rollout = jitted_rollout
        self._collection_id = collection_id_jax

        self.config = config
        self.quad_sys = PlanarMultirotorTransformed(mass=config.mass)
        self.ulim = irx.interval(list(config.ulim_lower), list(config.ulim_upper))
        self.sys_mjacM = irx.mjacM(self.quad_sys.f)
        self.perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.goal = jnp.array(config.goal_state)
        self.x_pert = jnp.array(config.x_pert)

    def compute(self, req: RolloutRequest) -> RolloutResult:
        jnp, irx, cfg = self._jnp, self._irx, self.config
        t0 = time.perf_counter()
        state = jnp.asarray(req.state)
        tube, ref, u_ff = self._jitted_rollout(
            jnp.array([req.t_start]), irx.icentpert(state, self.x_pert), state,
            jnp.asarray(req.K_feedback), jnp.asarray(req.K_reference),
            jnp.asarray(req.obs_wy), jnp.asarray(req.obs_wz),
            cfg.horizon, cfg.timestep, self.perm, self.sys_mjacM, cfg.mass,
            self.ulim, self.quad_sys, self.goal)
        violation_idx = int(self._collection_id(ref, tube, cfg.collection_threshold))
        # np.asarray blocks until XLA is done and gives plain arrays (cheap to pickle / index)
        result = RolloutResult(request=req,
                               reachable_tube=np.asarray(tube),
                               rollout_ref=np.asarray(ref),
                               feedfwd_input=np.asarray(u_ff),
                               violation_idx=violation_idx,
                               compute_time=0.0)
        return _with_compute_time(result, time.perf_counter() - t0)


def _with_compute_time(result: RolloutResult, compute_time: float) -> RolloutResult:
    return RolloutResult(result.request, result.reachable_tube, result.rollout_ref,
                         result.feedfwd_input, result.violation_idx, compute_time)


class ThreadRolloutBackend:
    """Runs the rollout in whichever thread calls ``submit`` (blocking that thread only)."""

    name = 'thread'

    def __init__(self, config: RolloutConfig):
        self.engine = RolloutEngine(config)
        self._result: Optional[RolloutResult] = None

    def warmup(self, request: RolloutRequest) -> float:
        t0 = time.perf_counter()
        self.engine.compute(request)
        return time.perf_counter() - t0

    @property
    def busy(self) -> bool:
        return False  # submit() only returns once the result is ready

    def submit(self, request: RolloutRequest) -> None:
        self._result = self.engine.compute(request)

    def poll(self) -> Optional[RolloutResult]:
        result, self._result = self._result, None
        return result

    def close(self) -> None:
        pass


def _worker_main(conn, config: RolloutConfig, warmup_request: RolloutRequest,
                 cpu_affinity: Optional[frozenset]) -> None:
    """Entry point of the rollout worker process."""
    try:
        if cpu_affinity:
            import os
            os.sched_setaffinity(0, cpu_affinity)  # keep the worker off the control thread's cores
        import px4_rta_mm_gpr.utilities.jax_setup  # noqa: F401  (enables float64 like the node)
        engine = RolloutEngine(config)
        t0 = time.perf_counter()
        engine.compute(warmup_request)          # first call triggers JIT compilation
        conn.send(('ready', time.perf_counter() - t0))
        while True:
            request = conn.recv()
            if request is None:
                break
            conn.send(('result', engine.compute(request)))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:  # report to the parent instead of dying silently
        try:
            conn.send(('error', traceback.format_exc()))
        except Exception:
            pass
    finally:
        conn.close()


class ProcessRolloutBackend:
    """Runs the rollout in a dedicated OS process; ``submit``/``poll`` never block.

    Uses the 'spawn' start method: forking a process that has already started JAX/XLA thread
    pools (and ROS 2 middleware threads) is unsafe, so the child starts a fresh interpreter.
    """

    name = 'process'

    def __init__(self, config: RolloutConfig, warmup_request: RolloutRequest,
                 cpu_affinity: Optional[frozenset] = None):
        ctx = mp.get_context('spawn')
        self._conn, child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(target=_worker_main, name='rta_rollout_worker', daemon=True,
                                 args=(child_conn, config, warmup_request, cpu_affinity))
        self._proc.start()
        child_conn.close()
        self._busy = False
        self.warmup_time: Optional[float] = None

    def wait_ready(self, timeout: Optional[float] = None) -> float:
        """Block until the worker finished its JIT warm-up; returns the warm-up time."""
        if not self._conn.poll(timeout):
            raise TimeoutError('rollout worker did not finish JIT warm-up in time')
        kind, payload = self._conn.recv()
        if kind == 'error':
            raise RuntimeError(f'rollout worker failed during warm-up:\n{payload}')
        self.warmup_time = payload
        return payload

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, request: RolloutRequest) -> None:
        if self._busy:
            raise RuntimeError('rollout worker is busy; poll() for the pending result first')
        self._conn.send(request)
        self._busy = True

    def poll(self) -> Optional[RolloutResult]:
        if not self._busy or not self._conn.poll(0):
            if self._busy and not self._proc.is_alive():
                raise RuntimeError('rollout worker process died')
            return None
        kind, payload = self._conn.recv()
        self._busy = False
        if kind == 'error':
            raise RuntimeError(f'rollout worker failed:\n{payload}')
        return payload

    def close(self) -> None:
        try:
            if self._proc.is_alive():
                self._conn.send(None)
                self._proc.join(timeout=2.0)
        except (BrokenPipeError, OSError):
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
