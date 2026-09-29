"""Numerical closed-loop simulation of the planar RTA-MM-GPR loop, built from the node's own components.

No ROS and no PX4: the "true" vehicle is the planar model (PlanarMultirotorTransformed) integrated with RK4 under a
time-varying wind field, and everything else is exactly what the node runs:

  * rollouts:   RolloutEngine (AOT-compiled, early exit at the certification point, ground floor, GP feedforward)
  * control:    u_applied (RTA feedback around the time-indexed plan row), input limits scaled with mass
  * gains:      LQR on the AOT-compiled linearization, re-linearized every ``relinearize_period`` seconds
  * GP data:    the same 9-row ring buffers (t, altitude-coordinate, wind) the node fills
  * replanning: pre-emptive (``replan_lead``) at each plan's certification point
  * logging:    the node's FlightRecorder -> the same HDF5 layout as a flight, so every analysis notebook works on it

    from px4_rta_mm_gpr.sim import SimConfig, simulate, paper_winds
    result = simulate(SimConfig(duration=20.0), winds=paper_winds(scale=0.6), log_path='sim.h5')
    print(result.summary)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

import px4_rta_mm_gpr.utilities.jax_setup  # noqa: F401  (float64 + compilation cache before anything compiles)
import control
import jax
import jax.numpy as jnp

from px4_rta_mm_gpr.concurrency import RolloutConfig, RolloutEngine, RolloutPlan, RolloutRequest
from px4_rta_mm_gpr.control_kernels import ControlKernels
from px4_rta_mm_gpr.flight_log import FlightRecorder
from px4_rta_mm_gpr.jax_mm_rta import PlanarMultirotorTransformed, u_applied
from .wind import calm

GRAVITY = 9.806
HW_MASS, HW_THRUST_LIMITS = 1.75, (13.0, 21.0)   # node's input limits, tuned at 1.75 kg (scaled with mass)


@dataclass
class SimConfig:
    duration: float = 20.0                 # (s)
    dt: float = 0.01                       # (s) control + integration step (= rollout step)
    mass: float = 2.0                      # (kg) SITL model mass
    x0: Tuple[float, ...] = (-4.0, -6.0, 0.0, 0.0, 0.1)   # (py, pz [NED], h, v, theta): the original studies' start
    goal: Tuple[float, ...] = (0.0, -1.0, 0.0, 0.0, 0.0)  # node's GOAL_STATE (1.0 m altitude)
    thrust_limits: Optional[Tuple[float, float]] = None   # None: hardware ratios scaled with mass (node default)
    roll_rate_limit: float = 1.0
    Q_feedback: Tuple[float, ...] = (20, 5, 10, 1, 1)
    R_feedback: Tuple[float, ...] = (50, 20)
    Q_reference: Tuple[float, ...] = (40, 10, 40, 40, 3)
    R_reference: Tuple[float, ...] = (50, 20)
    relinearize_period: float = 1.8        # (s) like the node's LQR update
    tube_horizon: float = 30.0
    early_exit: bool = True
    tube_margin: float = 1.0               # (s)
    collection_threshold: float = 0.25  # m: tube position bounds vs reference position (py, pz)
    position_uncertainty: float = 0.0   # (m) delta: position-estimate uncertainty (exact state in simulation)
    x_pert: float = 5e-4
    min_altitude: Optional[float] = 0.3    # (m) ground floor in the certificate (None: no floor)
    gp_feedforward: bool = True
    gp_epsilon: float = 0.25               # TVGPR forgetting rate; 0 = time-invariant GP
    embedding: str = 'u'                   # 'u' (66)-(67) (default), 'uw' (68)-(69), 'none' (64)-(65)
    gp_learn: bool = True
    observe_period: Optional[float] = 0.1  # (s) wind observations for the GP; None: only at replans (original studies)
    obs_noise_std: float = 0.0             # (N) noise added to each wind observation
    n_obs: int = 9
    replan_lead: float = 0.0               # (s) rollouts are instantaneous in simulated time
    backup_grace: float = 0.02             # (s) uncertified this long -> the run stops (the node would LAND)
    stop_on_backup: bool = True
    seed: int = 0


@dataclass
class SimResult:
    log_path: Optional[str]
    summary: dict
    recorder: FlightRecorder = field(repr=False, default=None)


def _rk4(f, x, u, wy, wz, dt):
    k1 = f(x, u, wy, wz)
    k2 = f(x + 0.5 * dt * k1, u, wy, wz)
    k3 = f(x + 0.5 * dt * k2, u, wy, wz)
    k4 = f(x + dt * k3, u, wy, wz)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(config: SimConfig = SimConfig(), winds: Optional[Tuple[Callable, Callable]] = None,
             log_path: Optional[str] = None, verbose: bool = False) -> SimResult:
    """Run the closed loop; returns the summary and (if log_path) writes a flight log in the node's format."""
    cfg = config
    wind_y, wind_z = winds if winds is not None else calm()
    rng = np.random.default_rng(cfg.seed)
    t_wall0 = time.perf_counter()

    # ---- model, limits, compiled pieces (all shared with the node) ----------------------------------------------
    quad = PlanarMultirotorTransformed(mass=cfg.mass)
    if cfg.thrust_limits is None:
        scale = cfg.mass / HW_MASS
        thrust_lo, thrust_hi = HW_THRUST_LIMITS[0] * scale, HW_THRUST_LIMITS[1] * scale
    else:
        thrust_lo, thrust_hi = cfg.thrust_limits
    ulim_lower, ulim_upper = (thrust_lo, -cfg.roll_rate_limit), (thrust_hi, cfg.roll_rate_limit)
    import immrax as irx
    ulim = irx.interval(list(ulim_lower), list(ulim_upper))
    kernels = ControlKernels(cfg.mass, 0.8, 0.1, cfg.dt, ulim_lower, ulim_upper, quad)   # for linearize()
    engine = RolloutEngine(RolloutConfig(
        mass=cfg.mass, horizon=cfg.tube_horizon, timestep=cfg.dt, ulim_lower=ulim_lower, ulim_upper=ulim_upper,
        goal_state=tuple(cfg.goal), x_pert=(cfg.x_pert,) * 5, collection_threshold=cfg.collection_threshold,
        n_obs=cfg.n_obs, early_exit=cfg.early_exit, margin_steps=int(round(cfg.tube_margin / cfg.dt)),
        min_altitude=cfg.min_altitude, gp_feedforward=cfg.gp_feedforward, gp_epsilon=cfg.gp_epsilon,
        embedding=cfg.embedding))
    f_true = jax.jit(lambda x, u, wy, wz: quad.f(0.0, x, u, wy, wz))
    step = lambda x, u, wy, wz: np.asarray(f_true(x, u, jnp.array([wy]), jnp.array([wz])))
    hover = np.array([cfg.mass * GRAVITY, 0.0])

    def gains_at(x, u):
        A, B = kernels.linearize(x, u)
        K_fb = control.lqr(A, B, np.diag(cfg.Q_feedback), np.diag(cfg.R_feedback))[0]
        K_ref = control.lqr(A, B, np.diag(cfg.Q_reference), np.diag(cfg.R_reference))[0]
        return np.asarray(K_fb), np.asarray(K_ref)

    # ---- logging (the node's recorder -> same file layout as a flight) ---------------------------------------
    meta = {k: (np.asarray(v, dtype=float) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
    meta.update(platform='numerical_sim', thrust_lo=thrust_lo, thrust_hi=thrust_hi,
                wind=f'{type(wind_y).__name__}(scale={getattr(wind_y, "scale", "?")})')
    rec = FlightRecorder(metadata={k: v for k, v in meta.items() if v is not None},
                         tick_capacity=int(cfg.duration / cfg.dt) + 10)

    # ---- initial state, GP buffers, gains ------------------------------------------------------------------
    x = np.asarray(cfg.x0, dtype=float)
    u = hover.copy()
    wy0, wz0 = float(wind_y(0.0, -x[1])), float(wind_z(0.0, x[0]))
    obs_wy = np.tile([0.0, x[1], wy0], (cfg.n_obs, 1))   # (t, pz, wy): y-wind as a function of altitude
    obs_wz = np.tile([0.0, x[0], wz0], (cfg.n_obs, 1))   # (t, py, wz): z-wind as a function of lateral position
    n_obs_written = 0
    K_fb, K_ref = gains_at(x, hover)
    rec.gain_update(0.0, True, K_fb, K_ref)
    last_lin = 0.0
    plan, plan_seq, collection_time = None, 0, -np.inf
    uncertified_since, uncertified_ticks, stop_reason = None, 0, None
    compute_times = []

    def observe(t, wy, wz):
        nonlocal obs_wy, obs_wz, n_obs_written
        if not cfg.gp_learn:
            return
        i = n_obs_written % cfg.n_obs
        n_obs_written += 1
        obs_wy = obs_wy.copy()
        obs_wy[i] = (t, x[1], wy + rng.normal(scale=cfg.obs_noise_std) if cfg.obs_noise_std else wy)
        obs_wz = obs_wz.copy()
        obs_wz[i] = (t, x[0], wz + rng.normal(scale=cfg.obs_noise_std) if cfg.obs_noise_std else wz)
        rec.wind_sample(t, wy, wz, np.nan, np.nan, np.nan, np.nan, x[1], x[0])

    n_steps = int(round(cfg.duration / cfg.dt))
    next_obs = 0.0
    k = 0
    for k in range(n_steps):
        t = k * cfg.dt
        wy = float(wind_y(t, -x[1]))          # true wind at the vehicle (fields take altitude = -pz)
        wz = float(wind_z(t, x[0]))

        if cfg.observe_period is not None and t >= next_obs - 1e-9:
            observe(t, wy, wz)
            next_obs += cfg.observe_period

        if t - last_lin >= cfg.relinearize_period:
            K_fb, K_ref = gains_at(x, u)
            rec.gain_update(t, False, K_fb, K_ref)
            last_lin = t

        # ---- pre-emptive replanning at the certification point ----
        if plan is None or t >= collection_time - cfg.replan_lead:
            if cfg.observe_period is None:
                observe(t, wy, wz)            # original studies: one observation per replan
            res = engine.compute(RolloutRequest(t_start=t, state=x, K_feedback=K_fb, K_reference=K_ref,
                                                obs_wy=obs_wy, obs_wz=obs_wz, goal=np.asarray(cfg.goal),
                                                delta=np.full(2, cfg.position_uncertainty)))
            compute_times.append(res.compute_time)
            collection_time = t + res.violation_idx * cfg.dt
            plan_seq += 1
            plan = RolloutPlan(t_start=t, dt=cfg.dt, reachable_tube=res.reachable_tube, rollout_ref=res.rollout_ref,
                               feedfwd_input=res.feedfwd_input, collection_time=collection_time,
                               compute_time=res.compute_time, latency=res.compute_time,
                               save_tube=np.zeros((0, 4)), tube_start=0, seq=plan_seq, state0=x.copy(),
                               K_feedback=K_fb, K_reference=K_ref, obs_wy=obs_wy, obs_wz=obs_wz,
                               violation_idx=res.violation_idx, warmup=False, goal=np.asarray(cfg.goal),
                               delta=np.full(2, cfg.position_uncertainty))
            rec.add_plan(plan)

        # ---- certification watchdog (the node commands LAND here) ----
        if t > plan.collection_time:
            uncertified_ticks += 1
            uncertified_since = t if uncertified_since is None else uncertified_since
            if cfg.stop_on_backup and t - uncertified_since > cfg.backup_grace:
                stop_reason = (f'no certified plan for {t - uncertified_since:.3f} s '
                               f'(plan #{plan.seq}, violation row {plan.violation_idx})')
                rec.event(t, 'backup', stop_reason)
                break
        else:
            uncertified_since = None

        # ---- control: RTA feedback around the time-indexed plan row ----
        c0 = time.perf_counter()
        idx = plan.index_at(t)
        u = np.asarray(u_applied(x, plan.rollout_ref[idx], plan.feedfwd_input[idx], K_fb, ulim))
        comp = time.perf_counter() - c0
        rec.tick(t, 0.0, x[0], x[1], 0.0, 0.0, x[2], x[3], x[4], 0.0,
                 u[0], np.nan, u[1], 0.0, 0.0, comp, wy, wz, plan, idx)

        # ---- true vehicle ----
        x = _rk4(step, x, u, wy, wz, cfg.dt)
        if -x[1] <= 0.0:
            stop_reason = 'ground contact'
            rec.event(t + cfg.dt, 'ground_contact', f'altitude {-x[1]:.3f} m')
            break
        if not np.all(np.isfinite(x)):
            stop_reason = 'state diverged'
            rec.event(t + cfg.dt, 'diverged', '')
            break

    ticks = rec.ticks
    alt = -ticks.column('z')
    summary = dict(
        completed=stop_reason is None, stop_reason=stop_reason, simulated_time=float(ticks.column('time')[-1]),
        final_state=x.tolist(), final_error_to_goal=float(np.linalg.norm(x[:2] - np.asarray(cfg.goal[:2]))),
        min_altitude=float(alt.min()), max_abs_theta_deg=float(np.degrees(np.abs(ticks.column('roll')).max())),
        plans=plan_seq, uncertified_fraction=uncertified_ticks / max(len(ticks), 1),
        rollout_ms_median=1e3 * float(np.median(compute_times)), rollout_ms_max=1e3 * float(np.max(compute_times)),
        wall_time_s=time.perf_counter() - t_wall0)
    path = None
    if log_path:
        path = rec.save(log_path, timing={'rollout_compute': np.asarray(compute_times)})
    if verbose:
        print(summary)
    return SimResult(path, summary, rec)
