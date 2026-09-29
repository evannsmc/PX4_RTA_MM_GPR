"""Analysis helpers for RTA-MM-GPR flight logs (SITL/hardware flights and numerical simulations alike).

    from px4_rta_mm_gpr import analysis as rta
    log = rta.load('flight.h5')
    rta.summary(log)                              # one-row table: timing, certification, tracking, altitude
    rta.plot_timeseries(log)                      # y / altitude / attitude / commands vs time, with plan references
    rta.plot_path(log)                            # y-altitude path with sampled certified tubes, goal and floor
    rta.animate(log, 'flight.mp4', gif_path='flight.gif')   # overview + follow-cam of every plan's tube + GPs

Everything is read from the flight log: each plan record carries its full tube, reference and the GP training data
it was computed with, so the GP used at any moment can be reconstructed exactly (``plan_gp``).
Conventions: planar state (py, pz, h, v, theta) with pz in NED (negative up); plots use altitude = -pz.
"""
from __future__ import annotations

import textwrap
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
        esc = tube_escapes(log)
        e = esc['escape'].to_numpy()[esc['certified'].to_numpy()]
        row.update(escape_pct=100 * float((e > 1e-3).mean()) if e.size else np.nan,   # > 1 mm (see tube_escapes)
                   escape_max_m=float(e.max()) if e.size else np.nan)
    rc = rollout_times(log)
    row.update(plans=len(log.plan_seqs), rollout_ms_median=1e3 * np.median(rc) if rc.size else np.nan,
               rollout_ms_max=1e3 * rc.max() if rc.size else np.nan,
               events='; '.join(f'{r.time:.2f}s {r.kind}' for r in log.events.itertuples()) or '-')
    return pd.DataFrame([row])


def tube_escapes(log: FlightLog) -> pd.DataFrame:
    """Per certified control tick: how far the vehicle's position is OUTSIDE the certified box of the row in use
    (0 = inside). In a numerical simulation this is the true state; in SITL / hardware it is the state estimate.
    A certificate is only as good as its disturbance model: a GP that is confidently wrong about the wind shows up
    here as escapes.
    The box is looked up in the stored plan at row floor((t - t_start) / dt + 1e-6) (logs written before that fix
    recorded some ticks one row early). Escapes below ~1 mm come from stepping the tube with 10 ms Euler steps."""
    T = log.ticks
    cert = T['plan_expired'].to_numpy() < 0.5
    t, y, z, seqs = T['time'].to_numpy(), T['y'].to_numpy(), T['z'].to_numpy(), T['plan_seq'].to_numpy().astype(int)
    out = np.zeros(len(T))
    for s_ in np.unique(seqs):
        m = seqs == s_
        p = log.plan(int(s_))
        tube = p['reachable_tube']
        k = np.clip(((t[m] - p['t_start']) / p['dt'] + 1e-6).astype(int), 0, len(tube) - 1)
        b = tube[k]
        out[m] = np.maximum.reduce([b[:, 0] - y[m], y[m] - b[:, 5], b[:, 1] - z[m], z[m] - b[:, 6], np.zeros(m.sum())])
    return pd.DataFrame({'time': t, 'certified': cert, 'escape': np.where(cert, out, 0.0)})


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


STATE_LABELS = ('y', 'altitude', 'h', 'v', 'theta')


def valid_rows(plan: dict) -> int:
    """Rows actually integrated (an early-exit rollout stops at the violation + margin; the rest is NaN)."""
    return int(np.isfinite(plan['reachable_tube']).all(axis=1).sum())


def plan_threshold(plan: dict, threshold: float = 0.25) -> np.ndarray:
    """The per-axis (y, altitude) threshold a plan was certified with: the base threshold + the plan's position
    uncertainty delta (0 in simulation; logs from before delta existed have none)."""
    delta = plan.get('delta')
    return threshold + (np.zeros(2) if delta is None else np.asarray(delta, dtype=float).reshape(2))


def tube_violation(plan: dict, threshold: float = 0.25, min_altitude: Optional[float] = None):
    """Why a plan's certificate ends: (row, reason) for the first failing row, or (None, '') if none fails.

    Same test as the rollout (jax_mm_rta.mm_rta._row_fails): a position bound (y or altitude) more than ``threshold``
    from the reference position, or the tube's lowest point (upper bound of pz, NED) below the floor. Logs recorded
    before the check became spatial may end on a velocity or attitude bound; that is reported too."""
    v = int(plan.get('violation_idx', -1))
    if v < 0:
        return None, ''
    tube, ref = plan['reachable_tube'][v], plan['rollout_ref'][v]
    if not (np.isfinite(tube).all() and np.isfinite(ref).all()):
        return v, 'NaN in the rollout'
    reasons = []
    if min_altitude is not None and tube[6] > -min_altitude:
        reasons.append(f'tube below the {min_altitude:g} m floor')
    dev = np.maximum(np.abs(ref - tube[:5]), np.abs(ref - tube[5:]))
    thr = plan_threshold(plan, threshold)
    for k in np.flatnonzero(dev[:2] > thr):
        reasons.append(f'tube {STATE_LABELS[k]} bound {dev[k]:.2f} m from the reference (> {thr[k]:.3g} m)')
    if not reasons:   # logs from before the spatial-only check (threshold applied to every state)
        reasons = [f'{STATE_LABELS[k]} bound {dev[k]:.2f} from the reference (old all-state check)'
                   for k in np.flatnonzero(dev > thr.max())]
    return v, '; '.join(reasons) or 'threshold'


def plan_times(plan: dict) -> np.ndarray:
    return plan['t_start'] + plan['dt'] * np.arange(plan['reachable_tube'].shape[0])


class PlanGP:
    """The time-varying GP a rollout used (same kernel and forgetting as jax_mm_rta.TVGPR), evaluated in NumPy."""

    def __init__(self, obs: np.ndarray, epsilon: float = GP_EPSILON):
        obs = np.asarray(obs, dtype=float)
        self.epsilon = float(epsilon)
        self.ts, self.s, self.y = obs[:, 0], obs[:, 1], obs[:, 2:3]
        k = GP_SIGMA_F * np.exp(-0.5 * (self.s[:, None] - self.s[None, :]) ** 2 / GP_LENGTH ** 2)
        d = (1 - self.epsilon) ** (np.abs(self.ts[:, None] - self.ts[None, :]) / 2)
        self.L = np.linalg.inv(k * d + GP_SIGMA_N ** 2 * np.eye(len(self.s)))

    def predict(self, t: float, s) -> tuple:
        """Mean and standard deviation at time t and positions s (array)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        ks = GP_SIGMA_F * np.exp(-0.5 * (self.s[:, None] - s[None, :]) ** 2 / GP_LENGTH ** 2)
        ks = ks * ((1 - self.epsilon) ** ((t - self.ts) / 2))[:, None]
        mean = (ks.T @ (self.L @ self.y)).ravel()
        var = GP_SIGMA_F - np.einsum('ij,ik,kj->j', ks, self.L, ks)
        return mean, np.sqrt(np.maximum(var, 0.0))


def plan_gp(plan: dict, which: str = 'y', epsilon: float = GP_EPSILON) -> PlanGP:
    """GP of the y-wind (a function of altitude coordinate pz) or z-wind (a function of py) used by a plan."""
    return PlanGP(plan['obs_wy' if which == 'y' else 'obs_wz'], epsilon)


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
        tube = p['reachable_tube'][:n]
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
    ys = scale * (np.hstack((0.3 * yt + 0.4, 0.4, 0, 0, 0.4, 0.3 * yt + 0.4)) - 0.2)   # centred on the position
    c, s = np.cos(theta), np.sin(theta)
    return c * xs - s * ys + y, s * xs + c * ys + alt


def _boxes(tube: np.ndarray) -> np.ndarray:
    """(n, 4, 2) rectangles in (y, altitude) from tube rows [lo(5), hi(5)] (pz is NED: altitude = -pz)."""
    ylo, yhi, alo, ahi = tube[:, 0], tube[:, 5], -tube[:, 6], -tube[:, 1]
    return np.stack([np.column_stack(c) for c in ((ylo, alo), (yhi, alo), (yhi, ahi), (ylo, ahi))], axis=1)


class _TubeArtists:
    """The tube of one plan in one axes: past rows (faded), certified rows (colored by look-ahead), the post-violation
    margin (hatched), the failing row (red) with the threshold box around the reference, and the reference."""

    def __init__(self, ax, cmap, norm, lw):
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Rectangle
        self.cmap, self.norm, self._trail = cmap, norm, []
        self.trail = ax.add_collection(PolyCollection([], facecolors='0.55', edgecolors='none', alpha=0.06, zorder=1))
        self.margin = ax.add_collection(PolyCollection([], facecolors='none', edgecolors='0.45', hatch='////',
                                                       linewidths=0.3, alpha=0.35, zorder=2))
        self.past = ax.add_collection(PolyCollection([], edgecolors='none', alpha=0.10, zorder=2))
        self.cert = ax.add_collection(PolyCollection([], linewidths=lw, zorder=3))
        self.fail = ax.add_patch(Rectangle((0, 0), 0, 0, fc='none', ec='tab:red', lw=1.6, zorder=5, visible=False))
        self.thr = ax.add_patch(Rectangle((0, 0), 0, 0, fc='none', ec='tab:red', lw=1.0, ls='--', zorder=5,
                                          visible=False))
        self.ref, = ax.plot([], [], color='k', lw=0.9, ls='--', zorder=4)
        self.ref_cert, = ax.plot([], [], color='k', lw=1.4, zorder=4)

    def add_trail(self, polys):
        self._trail.extend(polys)
        self.trail.set_verts(self._trail)

    def update(self, p, row_now, n_cert, n_valid, v_row, thr, stride):
        tube, ref, dt = p['reachable_tube'], p['rollout_ref'], float(p['dt'])
        r0 = min(row_now, n_cert)
        idx_cert = np.arange(r0, n_cert, stride)[::-1]                # draw far (large) boxes first
        rgba = self.cmap(self.norm((idx_cert - row_now) * dt))
        rgba[:, 3] = 0.30
        edge = rgba.copy()
        edge[:, 3] = 0.9
        self.cert.set_verts(_boxes(tube[idx_cert]))
        self.cert.set_facecolors(rgba)
        self.cert.set_edgecolors(edge)
        idx_past = np.arange(0, r0, stride)
        self.past.set_verts(_boxes(tube[idx_past]))
        self.past.set_facecolors(self.cmap(np.zeros(len(idx_past))))
        # margin rows (not certified), drawn only while they stay within a few thresholds (they grow very fast)
        half = 0.5 * np.maximum(tube[n_cert:n_valid, 5] - tube[n_cert:n_valid, 0],
                                tube[n_cert:n_valid, 6] - tube[n_cert:n_valid, 1])
        n_m = n_cert + int(np.argmax(half > 2 * thr.max())) if (half > 2 * thr.max()).any() else n_valid
        idx_m = np.arange(n_cert, n_m, 2 * stride)[::-1]
        self.margin.set_verts(_boxes(tube[idx_m]))
        self.ref.set_data(ref[:n_valid, 0], -ref[:n_valid, 1])
        self.ref_cert.set_data(ref[r0:n_cert, 0], -ref[r0:n_cert, 1])
        show = v_row is not None and v_row < n_valid
        self.fail.set_visible(show)
        self.thr.set_visible(show)
        if show:
            b = _boxes(tube[v_row:v_row + 1])[0]
            self.fail.set_bounds(b[0, 0], b[0, 1], b[1, 0] - b[0, 0], b[2, 1] - b[0, 1])
            ry, ra = ref[v_row, 0], -ref[v_row, 1]
            self.thr.set_bounds(ry - thr[0], ra - thr[1], 2 * thr[0], 2 * thr[1])   # (y, altitude) half-widths


def _gif_frame(rgb: np.ndarray, width: int):
    from PIL import Image
    im = Image.fromarray(rgb)
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return im.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def animate(log: FlightLog, out_path: Optional[str] = None, gif_path: Optional[str] = None, fps: int = 20,
            speed: float = 1.0, winds=None, altitude_range: Optional[Sequence[float]] = None,
            y_range: Optional[Sequence[float]] = None, zoom: float = 1.2, gif_fps: int = 8, gif_width: int = 640,
            stride: int = 2, t_range: Optional[Sequence[float]] = None, dpi: int = 100, progress: bool = False):
    """Video of a flight (SITL, hardware or numerical simulation) that shows every plan's mixed-monotone tube.

    Panels: the whole flight (path, current tube, faint trail of all earlier certified tubes, goal, floor, and the
    true wind field if ``winds`` = (wy_field, wz_field) is given); a follow-cam zoom on the vehicle where the tube is
    drawn box by box (one interval box per rollout step, colored by look-ahead time; rows the vehicle has already
    passed are faded; the uncertified margin after the violation is hatched; the failing box is outlined in red with
    the +-threshold box around the reference and the reason in the title); and the y-/z-wind GPs the plan used
    (mean +- 3 sigma, observations sized by recency, the range the tube spans shaded).

    Real time by default (``speed`` = 1): frames every 1/fps s of flight, so each plan is on screen while it is in
    use. Writes ``out_path`` (.mp4 via imageio-ffmpeg, or .gif) and optionally a smaller ``gif_path`` preview from the
    same rendering pass. Returns the paths written."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    use_style()
    T = log.ticks
    t_all = T['time'].to_numpy()
    y_all, alt_all, th_all = T['y'].to_numpy(), -T['z'].to_numpy(), T['roll'].to_numpy()
    seq_all = T['plan_seq'].to_numpy().astype(int)
    row_all = T['traj_idx'].to_numpy().astype(int)
    expired = T['plan_expired'].to_numpy().astype(bool)
    md = log.metadata
    thr = float(md.get('collection_threshold', 0.25))
    gp_eps = float(md.get('gp_epsilon', GP_EPSILON))
    floor = md.get('min_altitude')
    floor = float(floor) if isinstance(floor, (int, float, np.floating)) and floor > 0 else None
    goal = md.get('goal', md.get('goal_state'))

    t0, t1 = (t_range if t_range is not None else (t_all[0], t_all[-1]))
    frame_t = np.arange(t0, t1, speed / fps)
    frame_k = np.clip(np.searchsorted(t_all, frame_t), 0, len(t_all) - 1)
    plans, used = {}, np.unique(seq_all[frame_k])
    for s_ in np.unique(seq_all):
        p = log.plan(int(s_))
        v_row, reason = tube_violation(p, thr, floor)
        plans[int(s_)] = dict(p=p, n_cert=certified_rows(p), n_valid=valid_rows(p), v_row=v_row, reason=reason)
    horizon = max(plans[int(s_)]['n_cert'] * float(plans[int(s_)]['p']['dt']) for s_ in used)
    y_range = y_range or (min(-6.0, y_all.min() - 1), max(6.0, y_all.max() + 1))
    altitude_range = altitude_range or (-0.5, max(8.0, alt_all.max() + 1))

    fig = plt.figure(figsize=(12.8, 9.6), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], width_ratios=[1, 1.15], left=0.06, right=0.93,
                          bottom=0.07, top=0.91, hspace=0.28, wspace=0.18)
    ax, zx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    bx, cx = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap('viridis_r')
    norm = mcolors.Normalize(0.0, horizon)
    for a in (ax, zx):
        a.set_aspect('equal', adjustable='datalim' if a is zx else 'box')
        a.set_xlabel('y (m)')
        a.set_ylabel('altitude (m)')
        if floor is not None:
            a.axhspan(altitude_range[0] - 50, floor, color='tab:red', alpha=0.08, lw=0, zorder=0)
            a.axhline(floor, color='tab:red', lw=0.8, ls=':', zorder=0)
        if goal is not None:
            a.plot(goal[0], -goal[1], '*', color='tab:red', ms=14 if a is ax else 18, zorder=6)
    ax.set_xlim(*y_range)
    ax.set_ylim(*altitude_range)
    ax.set_title('whole flight', fontsize=10)
    zx.set_title('follow-cam: the current plan\'s reachable tube, one interval box per rollout step', fontsize=10)
    quiver = None
    if winds is not None:
        q_y, q_alt = np.meshgrid(np.linspace(y_range[0], y_range[1], 13), np.linspace(0.5, altitude_range[1], 13))
        quiver = ax.quiver(q_y, q_alt, np.zeros_like(q_y), np.zeros_like(q_y), color='0.55', alpha=0.55,
                           scale=60, width=0.003, zorder=1)
    tubes = [_TubeArtists(ax, cmap, norm, 0.3), _TubeArtists(zx, cmap, norm, 0.6)]
    paths = [a.plot([], [], color='tab:blue', lw=1.1, zorder=4)[0] for a in (ax, zx)]
    quads = [a.plot([], [], color='tab:purple', lw=1.5 if a is ax else 2.5, zorder=7)[0] for a in (ax, zx)]
    cam = ax.add_patch(Rectangle((0, 0), 0, 0, fc='none', ec='0.3', lw=0.8, zorder=8))
    reason_txt = zx.text(0.02, 0.02, '', transform=zx.transAxes, fontsize=8.5, color='tab:red', va='bottom',
                         bbox=dict(fc='white', ec='none', alpha=0.8), zorder=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=zx, fraction=0.035, pad=0.02)
    cb.set_label('look-ahead from now (s)')
    zx.legend(handles=[
        Patch(fc=cmap(0.3), ec=cmap(0.3), alpha=0.6, label='certified tube (interval boxes)'),
        Patch(fc='none', ec='0.45', hatch='////', label='margin after violation (not certified)'),
        Patch(fc='none', ec='tab:red', lw=1.6, label='first failing box'),
        Patch(fc='none', ec='tab:red', ls='--', label=f'reference +- threshold ({thr:g} m + delta) at that step'),
        Line2D([], [], color='k', lw=1.4, label='plan reference (certified part)'),
        Line2D([], [], color='tab:blue', lw=1.1, label='flown path'),
        Patch(fc='0.6', alpha=0.3, label='earlier certified tubes')],
        loc='upper left', fontsize=7.5, frameon=True, framealpha=0.85)
    status = fig.suptitle('', fontsize=11.5, x=0.06, ha='left')

    alts = np.linspace(altitude_range[0], altitude_range[1], 160)
    ys = np.linspace(y_range[0], y_range[1], 160)
    lim = 1.0
    for s_ in used:
        p = plans[int(s_)]['p']
        lim = max(lim, *np.abs(p['obs_wy'][:, 2]), *np.abs(p['obs_wz'][:, 2]))
    if winds is not None:
        lim = max(lim, np.abs(winds[0](t0, alts)).max(), np.abs(winds[1](t0, ys)).max())
    panels = []
    for axis, label, xs, xl in ((bx, 'y-wind GP (input: altitude)', alts, 'altitude (m)'),
                                (cx, 'z-wind GP (input: y)', ys, 'y (m)')):
        mean_l, = axis.plot([], [], color='tab:blue', lw=1.3, label='GP mean +- 3 sigma')
        band = [axis.fill_between(xs, 0 * xs, 0 * xs, color='tab:blue', alpha=0.18)]
        true_l, = axis.plot([], [], color='k', lw=0.9, ls=':', label='true wind' if winds is not None else '_')
        pts = axis.scatter([], [], color='tab:orange', edgecolor='k', linewidth=0.3, zorder=4, label='observations')
        span = [axis.axvspan(0, 0, color=cmap(0.3), alpha=0.25, lw=0, label='range the tube spans')]
        now_l = axis.axvline(0, color='tab:purple', lw=1.0, label='vehicle')
        axis.set_title(label, fontsize=10)
        axis.set_xlabel(xl)
        axis.set_ylabel('wind force (N)')
        axis.set_xlim(xs[0], xs[-1])
        axis.set_ylim(-1.6 * lim, 1.6 * lim)
        axis.legend(frameon=False, fontsize=7.5, loc='upper right', ncol=2)
        panels.append((axis, xs, mean_l, band, true_l, pts, span, now_l))

    trail_done, cam_c, cam_h = set(), None, None
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    writer = gif_frames = None
    outs = []
    mp4 = out_path is not None and out_path.lower().endswith('.mp4')
    if mp4:
        import imageio_ffmpeg   # pip install imageio-ffmpeg (bundles an ffmpeg binary)
        writer = imageio_ffmpeg.write_frames(out_path, (w, h), fps=fps, codec='libx264', quality=8,
                                             pix_fmt_out='yuv420p', macro_block_size=16)
        import warnings
        with warnings.catch_warnings():   # ffmpeg is fork+exec'd; JAX's fork warning does not apply
            warnings.filterwarnings('ignore', message='os.fork')
            writer.send(None)
    if gif_path is not None or (out_path is not None and not mp4):
        gif_frames = []
    gif_every = max(1, round(fps / gif_fps))

    for i, (t, k) in enumerate(zip(frame_t, frame_k)):
        seq = int(seq_all[k])
        P = plans[seq]
        p, n_cert, n_valid, v_row = P['p'], P['n_cert'], P['n_valid'], P['v_row']
        row_now = int(np.clip(row_all[k], 0, n_valid))
        for s_ in used[used < seq]:                        # trail: certified tubes of plans no longer in use
            if int(s_) not in trail_done:
                trail_done.add(int(s_))
                Q = plans[int(s_)]
                for ta in tubes:
                    ta.add_trail(_boxes(Q['p']['reachable_tube'][:Q['n_cert']:2]))
        for ta in tubes:
            ta.update(p, row_now, n_cert, n_valid, v_row, plan_threshold(p, thr), stride)
        for pl in paths:
            pl.set_data(y_all[:k + 1], alt_all[:k + 1])
        for q, sc in zip(quads, (0.25, 0.25)):
            q.set_data(*_quad_outline(y_all[k], alt_all[k], -th_all[k], sc))
        # follow-cam: centre on the vehicle, size to fit the certified tube and the failing box (smoothed)
        rows = p['reachable_tube'][row_now:(v_row + 1 if v_row is not None else n_cert)]
        b = _boxes(rows).reshape(-1, 2) if len(rows) else np.array([[y_all[k], alt_all[k]]])
        c = np.array([y_all[k], alt_all[k]])
        need = max(zoom, 1.15 * np.abs(b - c).max(), plan_threshold(p, thr).max() + 0.3)
        cam_c = c if cam_c is None else 0.7 * cam_c + 0.3 * c
        cam_h = need if cam_h is None else max(need, 0.9 * cam_h + 0.1 * need)
        cam_h = min(cam_h, 6.0)
        zx.set_xlim(cam_c[0] - cam_h, cam_c[0] + cam_h)
        zx.set_ylim(cam_c[1] - cam_h, cam_c[1] + cam_h)
        cam.set_bounds(cam_c[0] - cam_h, cam_c[1] - cam_h, 2 * cam_h, 2 * cam_h)
        cert_s = max(n_cert - row_now, 0) * float(p['dt'])
        state = 'UNCERTIFIED (no valid plan)' if expired[k] else f'certified {cert_s:.2f} s ahead'
        status.set_text(f't = {t:6.2f} s    plan #{seq}  (age {t - float(p["t_start"]):.2f} s, rollout '
                        f'{1e3 * float(p["compute_time"]):.1f} ms)    {state}')
        status.set_color('tab:red' if expired[k] else 'k')
        reason_txt.set_text(textwrap.fill(f'certificate ends at +{(v_row - row_now) * float(p["dt"]):.2f} s: '
                                          f'{P["reason"]}', 85) if v_row is not None
                            else 'no violation within the rollout horizon')
        if quiver is not None:
            wy_f, wz_f = winds
            quiver.set_UVC(wy_f(t, q_alt), -wz_f(t, q_y))
        tube_c = p['reachable_tube'][row_now:n_cert]
        for (axis, xs, mean_l, band, true_l, pts, span, now_l), which in zip(panels, ('y', 'z')):
            gp = plan_gp(p, which, gp_eps)
            m, sd = gp.predict(t, -xs if which == 'y' else xs)      # the y-wind GP input is pz (NED) = -altitude
            mean_l.set_data(xs, m)
            band[0].remove()
            band[0] = axis.fill_between(xs, m - 3 * sd, m + 3 * sd, color='tab:blue', alpha=0.12, lw=0)
            obs = p['obs_wy' if which == 'y' else 'obs_wz']
            pts.set_offsets(np.column_stack([-obs[:, 1] if which == 'y' else obs[:, 1], obs[:, 2]]))
            pts.set_sizes(70 * np.exp(0.5 * np.minimum(obs[:, 0] - t, 0)))
            if len(tube_c):
                lo, hi = ((-tube_c[:, 6].max(), -tube_c[:, 1].min()) if which == 'y'
                          else (tube_c[:, 0].min(), tube_c[:, 5].max()))
                span[0].remove()
                span[0] = axis.axvspan(lo, hi, color=cmap(0.3), alpha=0.25, lw=0)
            now_l.set_xdata([alt_all[k] if which == 'y' else y_all[k]] * 2)
            if winds is not None:
                true_l.set_data(xs, (winds[0] if which == 'y' else winds[1])(t, xs))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if writer is not None:
            writer.send(np.ascontiguousarray(rgb))
        if gif_frames is not None and i % gif_every == 0:
            gif_frames.append(_gif_frame(rgb, gif_width))
        if progress and i % 50 == 0:
            print(f'frame {i}/{len(frame_t)}', flush=True)
    plt.close(fig)
    if writer is not None:
        writer.close()
        outs.append(out_path)
    if gif_frames:
        target = gif_path if gif_path is not None else out_path
        gif_frames[0].save(target, save_all=True, append_images=gif_frames[1:], duration=int(1000 / fps * gif_every),
                           loop=0, optimize=True)
        outs.append(target)
    return outs
