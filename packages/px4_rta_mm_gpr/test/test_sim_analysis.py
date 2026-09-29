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
        outs = rta.animate(log, str(tmp_path / 'a.mp4'), gif_path=str(tmp_path / 'a.gif'), fps=2, gif_fps=2,
                           winds=paper_winds(scale=0.6))
        assert outs == [str(tmp_path / 'a.mp4'), str(tmp_path / 'a.gif')]
        assert all((tmp_path / n).stat().st_size > 1000 for n in ('a.mp4', 'a.gif'))


def test_certificate_is_spatial():
    """Only the position bounds (py, pz) and the floor end a certificate; velocity/attitude spread does not."""
    import jax.numpy as jnp
    from px4_rta_mm_gpr.jax_mm_rta import collection_id_jax
    ref = jnp.zeros((4, 5)).at[:, 1].set(-2.0)                       # hover at 2 m altitude
    lo, hi = ref - 0.01, ref + 0.01
    lo, hi = lo.at[1, 2:].add(-5.0), hi.at[1, 2:].add(5.0)          # row 1: huge velocity / attitude spread
    hi = hi.at[2, 0].set(0.6)                                          # row 2: y bound 0.6 m from the reference
    tube = jnp.hstack([lo, hi])
    assert int(collection_id_jax(ref, tube, 0.5)) == 2
    assert int(collection_id_jax(ref, tube.at[2, 5].set(0.01), 0.5)) == -1
    floor = tube.at[2, 5].set(0.01).at[3, 6].set(-0.2)                 # row 3: lowest point at 0.2 m altitude
    assert int(collection_id_jax(ref, floor, 0.5, z_max=-0.3)) == 3
    p = {'violation_idx': 2, 'reachable_tube': np.asarray(tube), 'rollout_ref': np.asarray(ref)}
    row, reason = rta.tube_violation(p, 0.5, 0.3)
    assert row == 2 and reason.startswith('tube y bound 0.60 m')


def test_calm_simulation_reaches_goal(tmp_path):
    res = simulate(SimConfig(duration=20.0), winds=calm())
    assert res.summary['completed'] and res.summary['final_error_to_goal'] < 0.05


def test_position_uncertainty_widens_box_and_threshold(tmp_path):
    """delta (per axis): the initial box is at least delta wide and the threshold is base + delta."""
    import jax.numpy as jnp
    from px4_rta_mm_gpr.jax_mm_rta import collection_id_jax
    ref = jnp.zeros((2, 5))
    tube = jnp.hstack([ref - 0.3, ref + 0.3])                       # 0.3 m in every direction
    assert int(collection_id_jax(ref, tube, jnp.array([0.25, 0.25]))) == 0
    assert int(collection_id_jax(ref, tube, jnp.array([0.25 + 0.06, 0.25 + 0.06]))) == -1
    assert int(collection_id_jax(ref, tube, jnp.array([0.31, 0.25]))) == 0   # per axis: altitude still fails
    res = simulate(SimConfig(duration=1.0, position_uncertainty=0.05), winds=calm(), log_path=str(tmp_path / 'd.h5'))
    assert res.summary['completed']
    with rta.load(res.log_path) as log:
        p = log.plan(1)
        assert np.allclose(p['delta'], 0.05)
        half = 0.5 * (p['reachable_tube'][0, 5:7] - p['reachable_tube'][0, 0:2])
        assert np.all(half >= 0.05 - 1e-12)                              # initial box contains the estimate error
        assert np.allclose(rta.plan_threshold(p, 0.25), 0.30)


def test_embedding_variants_run():
    """All three embedding systems (paper (68)-(69), (66)-(67), (64)-(65)) and the static GP certify and fly."""
    for eps, emb in ((0.25, 'uw'), (0.0, 'uw'), (0.0, 'u'), (0.0, 'none')):
        res = simulate(SimConfig(duration=1.0, gp_epsilon=eps, embedding=emb), winds=calm())
        assert res.summary['completed'] and res.summary['uncertified_fraction'] == 0.0, (eps, emb)
