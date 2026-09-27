"""Ahead-of-time compiled kernels for everything the node calls in flight.

Why AOT instead of "call the jitted function once in __init__"?
``jax.jit`` caches one executable per *signature*: shapes, dtypes, weak types (a Python float is a
weakly-typed scalar, a float64 array is not) and whether an array is already committed to a device.
A warm-up call only helps if its arguments have exactly the in-flight signature. The original node's
warm-up didn't (in flight, ``last_input`` became the controller's own output array), so XLA
recompiled NR_tracker_original mid-flight (322 ms).

``jax.jit(f).lower(ShapeDtypeStructs).compile()`` produces a ``Compiled`` object for ONE signature.
It never recompiles: a mismatched argument raises TypeError at once. Together with canonicalising
every argument to a float64 NumPy array (``_f64``), there is nothing left to compile after __init__.

Fusing: one control tick used to launch ~6 separate XLA computations (NR tracker, slicing, u_applied,
hstack, ...). ``control_step`` does all of it in one executable, i.e. one dispatch per tick.
"""
from __future__ import annotations

import time

import numpy as np

import px4_rta_mm_gpr.utilities.jax_setup  # noqa: F401  (float64 + compilation cache, before any compile)
import jax
import jax.numpy as jnp
import immrax as irx

from px4_rta_mm_gpr.jax_nr import NR_tracker_original, dynamics
from px4_rta_mm_gpr.jax_mm_rta import u_applied


def _f64(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float64)


def as_f64(x, shape):
    """Canonical in-flight argument: contiguous float64 NumPy array of the compiled shape."""
    return np.ascontiguousarray(x, dtype=np.float64).reshape(shape)


class ControlKernels:
    # NR integral-CBF limits (see jax_nr.integral_cbf); used as hard anti-windup bounds on the NR channels
    NR_RATE_LIMIT = 0.8

    def __init__(self, mass: float, T_lookahead: float, lookahead_step: float, integration_step: float,
                 ulim_lower, ulim_upper, quad_sys, anti_windup: bool = True):
        self.anti_windup = anti_windup
        ulim = irx.interval(list(ulim_lower), list(ulim_upper))
        obs_dyn = jnp.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=jnp.float64)
        zero_w = jnp.zeros(1)

        def control_step(nr_state, planar_state, last_input, ref_nr, ref_row, ff_row, K_fb, u_lo, u_hi):
            """NR tracker (pitch, yaw rates) + RTA feedback (thrust, roll rate) -> clipped 4-vector."""
            nr_u, _ = NR_tracker_original(nr_state, last_input, ref_nr, T_lookahead, lookahead_step,
                                          integration_step, mass)
            rta_u = u_applied(planar_state, ref_row, ff_row, K_fb, ulim)
            return jnp.clip(jnp.concatenate([rta_u, nr_u[2:]]), u_lo, u_hi)

        def wind_model(nr_state, u):
            """Model-predicted (ay, az) for the wind EKF's measurement equation."""
            return (obs_dyn @ dynamics(nr_state, u, mass))[1:]

        def linearize(state, u):
            return jax.jacfwd(quad_sys.f, argnums=(1, 2))(0.0, state, u, zero_w, zero_w)

        t0 = time.perf_counter()
        self._control_step = jax.jit(control_step).lower(
            _f64(9), _f64(5), _f64(4), _f64(4), _f64(5), _f64(2), _f64(2, 5), _f64(4), _f64(4)).compile()
        self._wind_model = jax.jit(wind_model).lower(_f64(9), _f64(4)).compile()
        self._linearize = jax.jit(linearize).lower(_f64(5), _f64(2)).compile()
        self.compile_time = time.perf_counter() - t0

        inf = np.inf
        r = self.NR_RATE_LIMIT
        self.u_lo = np.array([-inf, -inf, -r, -r]) if anti_windup else np.full(4, -inf)
        self.u_hi = np.array([inf, inf, r, r]) if anti_windup else np.full(4, inf)

    def control_step(self, nr_state, planar_state, last_input, ref_nr, ref_row, ff_row, K_fb) -> np.ndarray:
        return np.asarray(self._control_step(
            as_f64(nr_state, (9,)), as_f64(planar_state, (5,)), as_f64(last_input, (4,)), as_f64(ref_nr, (4,)),
            as_f64(ref_row, (5,)), as_f64(ff_row, (2,)), as_f64(K_fb, (2, 5)), self.u_lo, self.u_hi))

    def wind_model(self, nr_state, u) -> np.ndarray:
        return np.asarray(self._wind_model(as_f64(nr_state, (9,)), as_f64(u, (4,))))

    def linearize(self, state, u):
        A, B = self._linearize(as_f64(state, (5,)), as_f64(u, (2,)))
        return np.asarray(A), np.asarray(B)
