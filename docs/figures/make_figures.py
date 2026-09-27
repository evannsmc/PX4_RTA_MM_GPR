"""Generate every figure used by the docs/*.qmd files.

    python3 docs/figures/make_figures.py

Concept figures are drawn from the timings measured on this machine (see docs/data/).
Result figures are computed from the flight logs in docs/data/<config>_<run>.csv, which are
copies of the ROS2Logger CSVs produced by the benchmark flights.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# No Type 3 fonts in the PDFs (journal / LaTeX friendly)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'

CONFIGS = [  # (file prefix, label, colour)
    ('original', 'original (single-threaded)', '#b2182b'),
    ('single_thread', 'refactor, SingleThreadedExecutor', '#ef8a62'),
    ('multi_thread', 'refactor, MultiThreadedExecutor + thread rollouts', '#2166ac'),
    ('events_process', 'refactor, EventsExecutor + process rollouts', '#1b7837'),
]
HORIZON_CONFIGS = [
    ('multi_thread_h3', 'MultiThreadedExecutor + thread, 3 s horizon', '#67a9cf'),
    ('events_process_h3', 'EventsExecutor + process, 3 s horizon', '#5aae61'),
]

SHORT = {'original': 'original', 'single_thread': 'single executor', 'multi_thread': 'multi + thread',
         'events_process': 'events + process', 'multi_thread_h3': 'multi + thread, 3 s',
         'events_process_h3': 'events + process, 3 s'}

# Measured on this machine (see 02_rta_node_changes.qmd)
CONTROL_EXEC = 0.0016   # s, control callback body after the refactor
ROLLOUT = 0.165         # s, one rollout
PERIOD = 0.01


def _bar(ax, y, start, dur, color, label=None, hatch=None):
    ax.broken_barh([(start, dur)], (y - 0.35, 0.7), facecolors=color, edgecolor='k',
                   linewidth=0.3, hatch=hatch, label=label)


def executor_timelines():
    """Callback schedules for the three executor/back-end combinations."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True)
    T = 0.45
    c_ctrl, c_roll, c_miss = '#2166ac', '#ef8a62', '#d6604d'

    # (a) SingleThreadedExecutor: one thread, everything serialized
    ax = axes[0]
    t, roll_next, missed = 0.0, 0.03, []
    while t < T:
        if t >= roll_next:
            _bar(ax, 0, t, ROLLOUT, c_roll)
            skipped = np.arange(t + PERIOD, t + ROLLOUT, PERIOD)
            missed += list(skipped)
            t += ROLLOUT
            roll_next = t + 0.12
            continue
        _bar(ax, 0, t, CONTROL_EXEC * 3, c_ctrl)
        t += PERIOD
    for m in missed:
        ax.plot([m, m], [0.45, 0.62], color=c_miss, lw=0.8)
    ax.set_yticks([0], ['spin thread'])
    ax.set_title('(a) SingleThreadedExecutor', loc='left', fontsize=9)

    # (b) MultiThreadedExecutor + callback groups
    ax = axes[1]
    for t in np.arange(0, T, PERIOD):
        _bar(ax, 1, t, CONTROL_EXEC * 3, c_ctrl)
    t = 0.03
    while t + ROLLOUT < T:
        # the rollout thread only holds the GIL briefly at the start/end (Python glue); XLA runs GIL-free
        _bar(ax, 0, t, 0.006, c_roll)
        _bar(ax, 0, t + 0.006, ROLLOUT - 0.012, 'white', hatch='////')
        _bar(ax, 0, t + ROLLOUT - 0.006, 0.006, c_roll)
        t += ROLLOUT + 0.004
    ax.set_yticks([0, 1], ['rollout thread', 'control thread'])
    ax.set_title('(b) MultiThreadedExecutor + separate callback groups (thread backend)', loc='left', fontsize=9)

    # (c) EventsExecutor + rollout process
    ax = axes[2]
    for t in np.arange(0, T, PERIOD):
        _bar(ax, 1, t, CONTROL_EXEC * 3, c_ctrl)
    for t in np.arange(0.03, T - ROLLOUT, ROLLOUT + 0.01):
        ax.plot([t, t], [0.6, 1.0], color='k', lw=0.6)                       # submit
        _bar(ax, 0, t, ROLLOUT, c_roll)
        ax.plot([t + ROLLOUT, t + ROLLOUT], [0.4, 1.0], color='k', lw=0.6, ls=':')  # result polled
    ax.set_yticks([0, 1], ['worker process', 'spin thread'])
    ax.set_title('(c) EventsExecutor + ProcessRolloutBackend', loc='left', fontsize=9)
    ax.set_xlabel('time (s)')
    for ax in axes:
        ax.set_ylim(-0.7, 1.7 if ax is not axes[0] else 0.9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c_ctrl), plt.Rectangle((0, 0), 1, 1, fc=c_roll),
               plt.Rectangle((0, 0), 1, 1, fc='white', ec='k', hatch='////'),
               plt.Line2D([], [], color=c_miss)]
    fig.legend(handles, ['control callback', 'rollout', 'rollout inside XLA (GIL released)',
                         'missed control tick'], loc='lower center',
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / 'executor_timelines.pdf')
    plt.close(fig)


def gc_pause_timeline():
    """What a stop-the-world full garbage collection does to a 100 Hz loop."""
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ticks = np.arange(0, 0.6, PERIOD)
    gc_start, gc_len = 0.2, 0.30
    ran = ticks[(ticks < gc_start) | (ticks > gc_start + gc_len)]
    ax.vlines(ran, 0, 1, color='#2166ac', lw=1)
    blocked = ticks[(ticks >= gc_start) & (ticks <= gc_start + gc_len)]
    ax.vlines(blocked, 0, 1, color='#d6604d', lw=1, alpha=0.5, linestyles=':')
    ax.axvspan(gc_start, gc_start + gc_len, color='0.85')
    ax.text(gc_start + gc_len / 2, 1.1, 'gen-2 collection (~300 ms, GIL held: every Python thread frozen)',
            ha='center', fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(0, 1.35)
    ax.set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(HERE / 'gc_pause.pdf')
    plt.close(fig)


def load_runs():
    runs = {}
    for prefix, label, color in CONFIGS + HORIZON_CONFIGS:
        files = sorted(DATA.glob(f'{prefix}_[0-9].csv.gz'))
        runs[prefix] = [pd.read_csv(f) for f in files]
    return runs


def rta_window(df):
    t = df['time'].to_numpy()
    return t[~np.isnan(t)]


def control_period_cdf(runs):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for prefix, label, color in CONFIGS:
        periods = [np.diff(rta_window(d)) * 1e3 for d in runs[prefix]]
        if not periods:
            continue
        p = np.sort(np.concatenate(periods))
        ax.step(p, np.arange(1, p.size + 1) / p.size, where='post', color=color, label=label)
    ax.axvline(10, color='k', lw=0.6, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('control period during the RTA phase (ms, log scale)')
    ax.set_ylabel('fraction of ticks')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'control_period_cdf.pdf')
    plt.close(fig)


def trajectories(runs, configs=CONFIGS, name='trajectories.pdf'):
    fig, axes = plt.subplots(1, len(configs), figsize=(1.8 * len(configs) + 0.4, 2.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (prefix, label, color) in zip(axes, configs):
        for d in runs[prefix]:
            m = ~np.isnan(d['time'].to_numpy())
            ax.plot(d['y'].to_numpy()[m], -d['z'].to_numpy()[m], color=color, lw=0.9, alpha=0.8)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(SHORT.get(prefix, prefix), fontsize=7.5)
        ax.set_xlabel('y (m)')
    axes[0].set_ylabel('altitude = -z (m)')
    fig.tight_layout()
    fig.savefig(HERE / name)
    plt.close(fig)


def summary_table(runs):
    rows = []
    for prefix, label, _ in CONFIGS + HORIZON_CONFIGS:
        for i, d in enumerate(runs[prefix], 1):
            t = rta_window(d)
            dt = np.diff(t)
            z = d['z'].to_numpy()[~np.isnan(d['time'].to_numpy())]
            rows.append(dict(config=prefix, run=i, ticks=t.size, rate_hz=1 / dt.mean(),
                             p50_ms=1e3 * np.median(dt), p99_ms=1e3 * np.percentile(dt, 99),
                             max_ms=1e3 * dt.max(), late_pct=100 * np.mean(dt > 0.015),
                             ctrl_ms=1e3 * np.nanmedian(d['ctrl_comp_time']),
                             rollout_ms=1e3 * np.nanmedian(d['rollout_comptime']),
                             plan_age_ms=1e3 * np.nanmedian(d['plan_age']) if 'plan_age' in d else np.nan,
                             expired_pct=100 * np.nanmean(d['plan_expired']) if 'plan_expired' in d else np.nan,
                             crashed=bool(np.any(z > 0.0))))
    table = pd.DataFrame(rows)
    table.to_csv(DATA / 'summary.csv', index=False, float_format='%.2f')
    return table


def plan_freshness(runs):
    """Distribution of plan age (time since the plan's rollout started) at each control tick."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for prefix, label, color in CONFIGS[2:] + HORIZON_CONFIGS:
        ages = [d['plan_age'].dropna().to_numpy() * 1e3 for d in runs.get(prefix, []) if 'plan_age' in d]
        if not ages:
            continue
        a = np.sort(np.concatenate(ages))
        ax.step(a, np.arange(1, a.size + 1) / a.size, where='post', color=color, label=label)
    ax.set_xlabel('age of the plan used by a control tick (ms)')
    ax.set_ylabel('fraction of ticks')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'plan_age_cdf.pdf')
    plt.close(fig)


def tube_snapshot(h5_path):
    """Stored tubes from a new-format flight log: (a) every plan's certified tube along the path,
    (b) one plan's full stored tube versus time, with its safety-horizon cut-off."""
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from px4_rta_mm_gpr.flight_log import FlightLog
    with FlightLog(str(h5_path)) as log:
        t = log.ticks
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1, 1.3]})
        for seq in log.plan_seqs:
            p = log.plan(seq)
            if p['warmup']:
                continue
            n = p['violation_idx'] if p['violation_idx'] > 0 else 20
            for lo_y, lo_z, hi_y, hi_z in p['reachable_tube'][: n + 1, [0, 1, 5, 6]]:
                ax.add_patch(plt.Rectangle((lo_y, -hi_z), hi_y - lo_y, hi_z - lo_z, fc='#ef8a62', ec='none', alpha=0.15))
        ax.plot(t['y'], -t['z'], color='k', lw=0.9, label='flown path')
        ax.set_xlabel('y (m)')
        ax.set_ylabel('altitude = -z (m)')
        ax.set_title('(a) certified tube of every plan', loc='left', fontsize=8.5)
        ax.autoscale_view()

        mid = [q for q in log.plan_seqs if not log.plan(q)['warmup']]
        p = log.plan(mid[len(mid) // 3])
        tt = p['t_start'] + p['dt'] * np.arange(p['reachable_tube'].shape[0])
        tube, ref = p['reachable_tube'], p['rollout_ref']
        width = np.maximum(tube[:, 5] - tube[:, 0], tube[:, 6] - tube[:, 1])
        wide = np.flatnonzero(~(width < 4.0))                       # the interval bound diverges quickly
        n = int(wide[0]) if wide.size else tube.shape[0]
        tt, tube, ref = tt[:n], tube[:n], ref[:n]
        for (lo, hi, r, lab, col) in ((0, 5, 0, 'py', '#2166ac'), (1, 6, 1, 'pz', '#b2182b')):
            bx.fill_between(tt, tube[:, lo], tube[:, hi], color=col, alpha=0.25, lw=0, label=f'{lab} tube')
            bx.plot(tt, ref[:, r], color=col, lw=0.8, label=f'{lab} reference')
        if p['violation_idx'] > 0:
            bx.axvline(tt[p['violation_idx']], color='k', ls='--', lw=0.8, label='safety horizon')
        bx.set_xlabel('time (s)')
        bx.set_ylabel('position (m)')
        bx.set_title(f"(b) plan #{p['seq']}: stored tube until it is 4 m wide", loc='left', fontsize=8.5)
        bx.legend(frameon=False, fontsize=6.5, loc='best')
        fig.tight_layout()
        fig.savefig(HERE / 'tube_snapshot.pdf')
        plt.close(fig)


if __name__ == '__main__':
    executor_timelines()
    gc_pause_timeline()
    runs = load_runs()
    if any(runs.values()):
        control_period_cdf(runs)
        trajectories(runs)
        trajectories(runs, HORIZON_CONFIGS, 'trajectories_h3.pdf')
        plan_freshness(runs)
        example = DATA / 'example_flight.h5'
        if example.exists():
            tube_snapshot(example)
        print(summary_table(runs).to_string(index=False, float_format=lambda v: f'{v:.1f}'))
