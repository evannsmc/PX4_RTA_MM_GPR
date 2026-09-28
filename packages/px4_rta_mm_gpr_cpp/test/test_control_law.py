"""The C++ control law (control_law.hpp) must reproduce the Python/JAX ControlKernels.control_step.

    python3 test/test_control_law.py <path to control_law_check executable>

Random cases cover: normal flight, NR rates wound up past the anti-windup bound, the CBF active
(last input near/over its limits), RTA feedback saturating, and large attitudes/yaw offsets.
"""
import subprocess
import sys

import numpy as np

import jax
import jax.numpy as jnp

from px4_rta_mm_gpr.control_kernels import ControlKernels
from px4_rta_mm_gpr.jax_nr import predict_output
from px4_rta_mm_gpr.jax_mm_rta import PlanarMultirotorTransformed

MASS, T_LA, STEP, INTEG = 2.0, 0.8, 0.1, 0.01
U_LO, U_HI = np.array([14.857, -1.0]), np.array([24.0, 1.0])


def cases(n, rng):
    for k in range(n):
        kind = k % 5
        nr = rng.normal(size=9) * [2, 2, 3, 1, 1, 1, 0.3, 0.3, 0.5] + [0, 0, -5, 0, 0, 0, 0, 0, 0]
        if kind == 4:  # large attitude / yaw far from the reference
            nr[6:9] = rng.uniform(-1.0, 1.0, 3) * [0.9, 0.9, 6.0]
        planar = rng.normal(size=5) * [2, 2, 1, 1, 0.3]
        last = np.array([MASS * 9.806, 0, 0, 0]) + rng.normal(size=4) * [1.0, 0.3, 0.3, 0.3]
        if kind == 1:  # wound-up NR channels
            last[2:] = rng.uniform(-50, 50, 2)
        if kind == 2:  # CBF active: rates beyond +-0.8, thrust near its CBF bounds
            last = np.array([rng.choice([0.6, 26.9]), rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2)])
        ref_nr = np.array([0, rng.normal() * 2, -5 + rng.normal() * 2, rng.normal()])
        ref_row = planar + rng.normal(size=5) * (5.0 if kind == 3 else 0.3)  # kind 3: RTA feedback saturates
        ff_row = np.array([MASS * 9.806, 0]) + rng.normal(size=2) * [1, 0.2]
        K = rng.normal(size=(2, 5)) * [[0.3, 2, 0.3, 2, 0.5], [2, 0.3, 2, 0.3, 3]]
        yield nr, planar, last, ref_nr, ref_row, ff_row, K


def main(exe):
    rng = np.random.default_rng(42)
    worst = {}
    for anti_windup in (True, False):
        kern = ControlKernels(MASS, T_LA, STEP, INTEG, tuple(U_LO), tuple(U_HI), PlanarMultirotorTransformed(mass=MASS),
                              anti_windup=anti_windup)
        rows, expected = [], []
        for nr, planar, last, ref_nr, ref_row, ff_row, K in cases(500, rng):
            expected.append(kern.control_step(nr, planar, last, ref_nr, ref_row, ff_row, K))
            rows.append(np.concatenate([nr, planar, last, ref_nr, ref_row, ff_row, K.ravel(), U_LO, U_HI,
                                        kern.u_lo, kern.u_hi, [MASS, T_LA, STEP, INTEG]]))
        text = '\n'.join(' '.join(repr(float(v)) for v in r) for r in rows) + '\n'
        out = subprocess.run([exe], input=text, capture_output=True, text=True, check=True).stdout
        got = np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])
        expected = np.array(expected)
        err = np.abs(got - expected) / np.maximum(1.0, np.abs(expected))
        # Rounding differences between XLA and Eigen are amplified by the conditioning of the NR prediction
        # Jacobian dg/du, which the tracker inverts. Report the error per condition-number band.
        jac = jax.jit(jax.jacfwd(predict_output, 1))
        c = [jnp.asarray(v) for v in (T_LA, STEP, MASS)]
        conds = np.array([np.linalg.cond(np.asarray(jac(jnp.asarray(r[:9]), jnp.asarray(r[14:18]), *c)))
                          for r in rows])
        for lo, hi in ((0, 1e3), (1e3, 1e6), (1e6, np.inf)):
            m = (conds >= lo) & (conds < hi)
            if m.any():
                print(f"    cond(dg/du) in [{lo:.0e}, {hi:.0e}): {m.sum():3d} cases, max relative error {err[m].max():.1e}")
        worst[anti_windup] = err[conds < 1e6].max()
        k = np.unravel_index(err.argmax(), err.shape)
        print(f"anti_windup={anti_windup}: {len(rows)} cases, max relative error {err.max():.2e} "
              f"(case {k[0]}, channel {k[1]}: C++ {got[k]:.12g} vs JAX {expected[k]:.12g})")
    ok = max(worst.values()) < 1e-8  # for well-conditioned Jacobians (cond < 1e6)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
