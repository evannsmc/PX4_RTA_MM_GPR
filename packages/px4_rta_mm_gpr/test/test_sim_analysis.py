"""Numerical simulation (px4_rta_mm_gpr.sim) and analysis helpers (px4_rta_mm_gpr.analysis)."""
import numpy as np

import px4_rta_mm_gpr.utilities.jax_setup  # noqa: F401
import jax.numpy as jnp
from px4_rta_mm_gpr import analysis as rta
from px4_rta_mm_gpr.jax_mm_rta import TVGPR
from px4_rta_mm_gpr.sim import SimConfig, calm, paper_winds, simulate


def test_plan_gp_matches_tvgpr():
    rng = np.random.default_rng(0)
    obs = np.column_stack([np.sort(rng.uniform(0, 3, 9)), rng.uniform(-8, -1, 9), rng.normal(size=9)])
    gp = TVGPR(jnp.asarray(obs), sigma_f=5.0, l=2.0, sigma_n=0.01, epsilon=0.25)
    ours = rta.PlanGP(obs)
    for t, s in ((3.1, -4.0), (3.5, -1.2), (4.0, -7.5)):
        m, sd = ours.predict(t, [s])
        q = jnp.array([t, s])
        np.testing.assert_allclose(m[0], float(gp.mean(q).ravel()[0]), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(sd[0] ** 2, float(gp.variance(q).ravel()[0]), rtol=1e-7, atol=1e-9)


def test_wind_field_is_a_pure_function_of_time():
    a, _ = paper_winds(scale=1.0, seed=3)
    b, _ = paper_winds(scale=1.0, seed=3)
    s = np.linspace(0, 12, 7)
    late_first = a(7.3, s)
    for t in np.arange(0, 7.3, 0.1):   # forward sweep on b
        b(t, s)
    np.testing.assert_array_equal(late_first, b(7.3, s))
    np.testing.assert_array_equal(a(0.5, s), b(0.5, s))   # going back in time is fine too


def test_short_simulation_round_trip(tmp_path):
    res = simulate(SimConfig(duration=4.0), winds=paper_winds(scale=0.6), log_path=str(tmp_path / 'sim.h5'))
    s = res.summary
    assert s['completed'] and s['uncertified_fraction'] == 0.0 and s['plans'] > 3
    with rta.load(res.log_path) as log:
        assert log.metadata['platform'] == 'numerical_sim'
        assert len(log.ticks) == 400
        row = rta.summary(log).iloc[0]
        assert row['plans'] == s['plans'] and row['uncertified_pct'] == 0.0
        p = log.plan(1)
        assert p['reachable_tube'].shape[1] == 10 and p['obs_wy'].shape == (9, 3)
        gif = rta.animate(log, str(tmp_path / 'a.gif'), every=40, fps=5, winds=paper_winds(scale=0.6))
        assert (tmp_path / 'a.gif').stat().st_size > 1000 and gif.endswith('a.gif')


def test_calm_simulation_reaches_goal(tmp_path):
    res = simulate(SimConfig(duration=20.0), winds=calm())
    assert res.summary['completed'] and res.summary['final_error_to_goal'] < 0.05
