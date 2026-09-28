"""Analysis helpers for RTA-MM-GPR flight logs (SITL/hardware flights and numerical simulations alike).

    from px4_rta_mm_gpr import analysis as rta
    log = rta.load('flight.h5')
    rta.summary(log)                              # one-row table: timing, certification, tracking, altitude
    rta.plot_timeseries(log)                      # y / altitude / attitude / commands vs time, with plan references
    rta.plot_path(log)                            # y-altitude path with sampled certified tubes, goal and floor
    rta.animate(log, 'flight.gif')                # path + current tube + the GP each plan used

Everything is read from the flight log: each plan record carries its full tube, reference and the GP training data
it was computed with, so the GP used at any moment can be reconstructed exactly (``plan_gp``).
Conventions: planar state (py, pz, h, v, theta) with pz in NED (negative up); plots use altitude = -pz.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from px4_rta_mm_gpr.flight_log import FlightLog

# GP hyper-parameters used by the rollouts (jax_mm_rta/mm_rta.py)
GP_SIGMA_F, GP_LENGTH, GP_SIGMA_N, GP_EPSILON = 5.0, 2.0, 0.01, 0.25

PLOT_STYLE = {'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10,
              'axes.spines.top': False, 'axes.spines.right': False}


def load(path: str) -> FlightLog:
    """Open a flight log (flight_recorder layout or the earlier px4_rta_mm_gpr layout)."""
    return FlightLog(path)


def use_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_STYLE)   # no Type 3 fonts in saved PDFs


# ----------------------------------------------------------------------------------------------- tables
def summary(log: FlightLog, name: Optional[str] = None) -> pd.DataFrame:
    """One-row summary of a flight: control timing, rollouts, certification, tracking and altitude."""
    T = log.ticks
    row = {'flight': name or log.path.split('/')[-1], 'platform': log.metadata.get('platform', '?')}
    if len(T):
        t = T['time'].to_numpy()
        p = np.diff(t)
        att = np.degrees(np.maximum(np.abs(T['roll']), np.abs(T['pitch'])))
        row.update(
            duration_s=t[-1] - t[0], ticks=len(T), control_rate_hz=1 / p.mean() if p.size else np.nan,
            period_p99_ms=1e3 * np.percentile(p, 99) if p.size else np.nan,
            ctrl_us_median=1e6 * np.nanmedian(T['ctrl_comp_time']),
            uncertified_pct=100 * T['plan_expired'].mean(),
            rmse_y=float(np.sqrt(np.mean((T['y'] - T['ref_py']) ** 2))),
            rmse_altitude=float(np.sqrt(np.mean((T['z'] - T['ref_pz']) ** 2))),
            final_altitude=float(-T['z'].iloc[-min(len(T), 100):].mean()),
            lowest_altitude=float(-T['z'].max()),
            lowest_certified_tube=float(-T['tube_pz_hi'].max()),
            max_attitude_deg=float(att.max()))
    rc = rollout_times(log)
    row.update(plans=len(log.plan_seqs), rollout_ms_median=1e3 * np.median(rc) if rc.size else np.nan,
               rollout_ms_max=1e3 * rc.max() if rc.size else np.nan,
               events='; '.join(f'{r.time:.2f}s {r.kind}' for r in log.events.itertuples()) or '-')
    return pd.DataFrame([row])


def rollout_times(log: FlightLog) -> np.ndarray:
    if 'rollout_compute' in log.timing:
        return np.asarray(log.timing['rollout_compute'], dtype=float)
    return np.array([log.plan(s)['compute_time'] for s in log.plan_seqs], dtype=float)


def tracking_errors(log: FlightLog) -> pd.DataFrame:
    """Per-tick deviation from the plan row in use (what the RTA feedback is correcting)."""
    T = log.ticks
    return pd.DataFrame({'time': T['time'], 'e_y': T['y'] - T['ref_py'], 'e_altitude': -(T['z'] - T['ref_pz']),
                         'plan_age': T['plan_age'], 'expired': T['plan_expired']})


# ----------------------------------------------------------------------------------------------- plans and GPs
def certified_rows(plan: dict) -> int:
    """Number of certified rows of a plan (rows before the first violation)."""
    v = int(plan.get('violation_idx', -1))
    return plan['reachable_tube'].shape[0] if v < 0 else max(v, 1)


def plan_times(plan: dict) -> np.ndarray:
    return plan['t_start'] + plan['dt'] * np.arange(plan['reachable_tube'].shape[0])


class PlanGP:
    """The time-varying GP a rollout used (same kernel and forgetting as jax_mm_rta.TVGPR), evaluated in NumPy."""

    def __init__(self, obs: np.ndarray):
        obs = np.asarray(obs, dtype=float)
        self.ts, self.s, self.y = obs[:, 0], obs[:, 1], obs[:, 2:3]
        k = GP_SIGMA_F * np.exp(-0.5 * (self.s[:, None] - self.s[None, :]) ** 2 / GP_LENGTH ** 2)
        d = (1 - GP_EPSILON) ** (np.abs(self.ts[:, None] - self.ts[None, :]) / 2)
        self.L = np.linalg.inv(k * d + GP_SIGMA_N ** 2 * np.eye(len(self.s)))

    def predict(self, t: float, s) -> tuple:
        """Mean and standard deviation at time t and positions s (array)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        ks = GP_SIGMA_F * np.exp(-0.5 * (self.s[:, None] - s[None, :]) ** 2 / GP_LENGTH ** 2)
        ks = ks * ((1 - GP_EPSILON) ** ((t - self.ts) / 2))[:, None]
        mean = (ks.T @ (self.L @ self.y)).ravel()
        var = GP_SIGMA_F - np.einsum('ij,ik,kj->j', ks, self.L, ks)
        return mean, np.sqrt(np.maximum(var, 0.0))


def plan_gp(plan: dict, which: str = 'y') -> PlanGP:
    """GP of the y-wind (a function of altitude coordinate pz) or z-wind (a function of py) used by a plan."""
    return PlanGP(plan['obs_wy' if which == 'y' else 'obs_wz'])


# ----------------------------------------------------------------------------------------------- plots
def plot_timeseries(log: FlightLog, fig=None):
    """Lateral position, altitude, roll and commands vs time, with the plan reference and certified tube bounds."""
    import matplotlib.pyplot as plt
    use_style()
    T = log.ticks
    t = T['time']
    fig = fig or plt.figure(figsize=(10, 7))
    axes = fig.subplots(2, 2, sharex=True)
    ax = axes[0, 0]
    ax.fill_between(t, T['tube_py_lo'], T['tube_py_hi'], color='tab:orange', alpha=0.3, lw=0, label='tube row in use')
    ax.plot(t, T['ref_py'], 'k--', lw=0.8, label='plan reference')
    ax.plot(t, T['y'], color='tab:blue', lw=1.2, label='flown')
    ax.set_ylabel('y (m)')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    ax.fill_between(t, -T['tube_pz_hi'], -T['tube_pz_lo'], color='tab:orange', alpha=0.3, lw=0)
    ax.plot(t, -T['ref_pz'], 'k--', lw=0.8)
    ax.plot(t, -T['z'], color='tab:blue', lw=1.2)
    floor = log.metadata.get('min_altitude')
    if isinstance(floor, (int, float)) and floor > 0:
        ax.axhline(floor, color='tab:red', lw=0.8, ls=':', label=f'floor {floor} m')
        ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel('altitude (m)')
    ax = axes[1, 0]
    ax.plot(t, np.degrees(T['roll']), lw=1, label='roll')
    ax.plot(t, np.degrees(T['pitch']), lw=1, label='pitch')
    ax.set_ylabel('attitude (deg)')
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel('time (s)')
    ax = axes[1, 1]
    ax.plot(t, T['thrust'], lw=1, color='tab:green')
    ax.set_ylabel('thrust command (N)', color='tab:green')
    ax2 = ax.twinx()
    ax2.plot(t, T['roll_rate'], lw=0.8, color='tab:purple')
    ax2.set_ylabel('roll-rate command (rad/s)', color='tab:purple')
    ax.set_xlabel('time (s)')
    fig.tight_layout()
    return fig


def plot_path(log: FlightLog, ax=None, n_tubes: int = 12, show_reference: bool = True):
    """y-altitude path with the certified tubes of evenly sampled plans, the goal and the ground floor."""
    import matplotlib.pyplot as plt
    use_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    T = log.ticks
    used = np.unique(T['plan_seq'].to_numpy().astype(int)) if len(T) else log.plan_seqs
    cmap = plt.get_cmap('viridis')
    for i, seq in enumerate(used[np.linspace(0, len(used) - 1, min(n_tubes, len(used))).astype(int)]):
        p = log.plan(int(seq))
        n = certified_rows(p)
        tube, ref = p['reachable_tube'][:n], p['rollout_ref'][:n]
        c = cmap(i / max(1, n_tubes - 1))
        for lo_y, lo_z, hi_y, hi_z in tube[::5, [0, 1, 5, 6]]:
            ax.add_patch(plt.Rectangle((lo_y, -hi_z), hi_y - lo_y, hi_z - lo_z, fc=c, ec='none', alpha=0.35))
        if show_reference:
            ax.plot(p['rollout_ref'][:, 0], -p['rollout_ref'][:, 1], color=c, lw=0.6, ls='--')
    ax.plot(T['y'], -T['z'], color='k', lw=1.2, label='flown path')
    ax.plot(T['y'].iloc[0], -T['z'].iloc[0], 'o', color='tab:green', label='RTA start')
    goal = log.metadata.get('goal_state')
    if goal is not None:
        ax.plot(goal[0], -goal[1], '*', color='tab:red', ms=12, label='goal')
    floor = log.metadata.get('min_altitude')
    if isinstance(floor, (int, float)) and floor > 0:
        ax.axhspan(-1.0, floor, color='tab:red', alpha=0.1, lw=0, label=f'below floor ({floor} m)')
    ax.set_xlabel('y (m)')
    ax.set_ylabel('altitude (m)')
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.set_title('certified tubes (solid) and plan references (dashed) of sampled plans', fontsize=9)
    return ax


def _quad_outline(y, alt, theta, scale):
    ts = -0.5 * np.arange(np.pi, 1.5 * np.pi, 0.2) + 0.3
    xt, yt = np.cos(ts), np.sin(ts) * np.cos(ts)
    xs = scale * np.hstack((0.4 * xt - 1, -1, -1, 1, 1, 0.4 * xt + 1))
    ys = scale * np.hstack((0.3 * yt + 0.4, 0.4, 0, 0, 0.4, 0.3 * yt + 0.4))
    c, s = np.cos(theta), np.sin(theta)
    return c * xs - s * ys + y, s * xs + c * ys + alt


def animate(log: FlightLog, out_path: str, every: int = 10, fps: int = 20, winds=None,
            altitude_range: Sequence[float] = (-0.5, 13.0), y_range: Optional[Sequence[float]] = None):
    """GIF: flight with the current plan's certified tube and reference, plus the y- and z-wind GPs of that plan
    (mean +- 3 sigma, observations sized by recency) and, if ``winds`` = (wy_field, wz_field) is given (numerical
    simulations), the true wind profiles and a wind-field quiver."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    use_style()
    T = log.ticks
    t_all = T['time'].to_numpy()
    y_all, alt_all, th_all = T['y'].to_numpy(), -T['z'].to_numpy(), T['roll'].to_numpy()
    seq_all = T['plan_seq'].to_numpy().astype(int)
    frames = np.arange(0, len(T), every)
    y_range = y_range or (min(-6.0, y_all.min() - 1), max(6.0, y_all.max() + 1))
    alts = np.linspace(altitude_range[0], altitude_range[1], 120)
    ys = np.linspace(y_range[0], y_range[1], 120)

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.4, 1, 1]})
    ax.set_xlim(*y_range)
    ax.set_ylim(*altitude_range)
    ax.set_xlabel('y (m)')
    ax.set_ylabel('altitude (m)')
    floor = log.metadata.get('min_altitude')
    if isinstance(floor, (int, float)) and floor > 0:
        ax.axhspan(altitude_range[0], floor, color='tab:red', alpha=0.1, lw=0)
    goal = log.metadata.get('goal_state')
    if goal is not None:
        ax.plot(goal[0], -goal[1], '*', color='tab:red', ms=12)
    quiver = None
    if winds is not None:   # wind field: y-wind as horizontal arrows, z-wind (NED, + = down) as vertical arrows
        q_y, q_alt = np.meshgrid(np.linspace(y_range[0], y_range[1], 11), np.linspace(0.5, altitude_range[1], 11))
        quiver = ax.quiver(q_y, q_alt, np.zeros_like(q_y), np.zeros_like(q_y), color='0.55', alpha=0.6,
                           scale=60, width=0.003)
    path_line, = ax.plot([], [], 'k', lw=1.0)
    ref_line, = ax.plot([], [], 'k--', lw=0.8)
    lower_line, = ax.plot([], [], color='tab:red', lw=1.0)
    upper_line, = ax.plot([], [], color='tab:blue', lw=1.0)
    quad_line, = ax.plot([], [], color='tab:purple', lw=2)
    title = ax.set_title('')
    panels = []
    for axis, label, xs in ((bx, 'y-wind vs altitude', alts), (cx, 'z-wind vs lateral y', ys)):
        mean_l, = axis.plot([], [], color='tab:blue', lw=1.2, label='GP mean')
        band = [axis.fill_between(xs, 0 * xs, 0 * xs, color='tab:blue', alpha=0.2)]
        true_l, = axis.plot([], [], color='k', lw=0.8, ls=':', label='true wind' if winds is not None else '_')
        pts = axis.scatter([], [], color='tab:orange', zorder=3, label='observations')
        axis.set_title(label, fontsize=9)
        axis.set_xlabel('altitude (m)' if axis is bx else 'y (m)')
        axis.set_ylabel('wind force (N)')
        axis.set_xlim(xs[0], xs[-1])
        axis.legend(frameon=False, fontsize=7, loc='upper right')
        panels.append((axis, xs, mean_l, band, true_l, pts))

    def update(k):
        t, seq = t_all[k], seq_all[k]
        p = log.plan(int(seq))
        n = certified_rows(p)
        path_line.set_data(y_all[:k + 1], alt_all[:k + 1])
        ref_line.set_data(p['rollout_ref'][:, 0], -p['rollout_ref'][:, 1])
        lower_line.set_data(p['reachable_tube'][:n, 0], -p['reachable_tube'][:n, 1])
        upper_line.set_data(p['reachable_tube'][:n, 5], -p['reachable_tube'][:n, 6])
        quad_line.set_data(*_quad_outline(y_all[k], alt_all[k], -th_all[k], 0.35))
        title.set_text(f't = {t:5.2f} s   plan #{seq}   certified for {n * p["dt"]:.2f} s')
        if quiver is not None:
            wy_f, wz_f = winds
            quiver.set_UVC(wy_f(t, q_alt), -wz_f(t, q_y))
        for (axis, xs, mean_l, band, true_l, pts), which in zip(panels, ('y', 'z')):
            gp = plan_gp(p, which)
            s_query = -xs if which == 'y' else xs          # the y-wind GP input is pz (NED) = -altitude
            m, sd = gp.predict(t, s_query)
            mean_l.set_data(xs, m)
            band[0].remove()
            band[0] = axis.fill_between(xs, m - 3 * sd, m + 3 * sd, color='tab:blue', alpha=0.2, lw=0)
            obs = p['obs_wy' if which == 'y' else 'obs_wz']
            ox = -obs[:, 1] if which == 'y' else obs[:, 1]
            pts.set_offsets(np.column_stack([ox, obs[:, 2]]))
            pts.set_sizes(60 * np.exp(0.5 * np.minimum(obs[:, 0] - t, 0)))
            if winds is not None:
                field = winds[0] if which == 'y' else winds[1]
                true_l.set_data(xs, field(t, xs))
            lo, hi = np.nanmin(m - 3 * sd), np.nanmax(m + 3 * sd)
            axis.set_ylim(min(lo, -1.0) - 0.5, max(hi, 1.0) + 0.5)
        return path_line, ref_line, lower_line, upper_line, quad_line

    def frame(i):
        return update(frames[i])

    fig.tight_layout()
    anim = FuncAnimation(fig, frame, frames=len(frames), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
