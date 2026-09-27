"""Round-3 ablation: per-flight table and figures from the HDF5 flight logs.

    python3 docs/figures/make_round3.py <log_dir>

Reads <log_dir>/<config>_<run>.h5, writes docs/data/round3_flights.csv (one row per flight), a
compact per-tick CSV per flight for the figures (docs/data/round3/), and the figures.
Outcome per flight:
  UPSET  attitude above 60 deg or more than 8 m lateral flyaway during the RTA phase
  touch  reached z > 0 in the node's (ground-offset) frame, i.e. within ~0.5 m of the ground
  ok     neither
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
sys.path.insert(0, str(HERE.parents[1]))

CONFIGS = [  # prefix, label
    ('r3_base', 'infrastructure only'),
    ('r3_thrust', '+ mass-scaled thrust limits'),
    ('r3_ramp', '+ entry ramp'),
    ('r3_all', '+ thrust limits + entry ramp'),
    ('r3_base_nogp', 'infrastructure, no GP learning'),
    ('r4_old_nrref', 'infrastructure (repeat)'),
    ('r4_nrplan', '+ NR reference from plan'),
    ('r4_nrplan_thrust', '+ NR reference from plan + thrust limits'),
]


def classify(T):
    att = np.degrees(np.maximum(np.abs(T['roll']), np.abs(T['pitch']))).max()
    if att > 60 or np.abs(T['y']).max() > 8:
        return 'UPSET', att
    return ('touch' if (T['z'] > 0).any() else 'ok'), att


def main(log_dir):
    from px4_rta_mm_gpr.flight_log import FlightLog
    rows = []
    (DATA / 'round3').mkdir(parents=True, exist_ok=True)
    for prefix, label in CONFIGS:
        for f in sorted(Path(log_dir).glob(f'{prefix}_[0-9].h5')):
            with FlightLog(str(f)) as log:
                T = log.ticks
                outcome, att = classify(T)
                p = np.diff(T['time'].to_numpy())
                rows.append(dict(config=prefix, label=label, run=int(f.stem.split('_')[-1]), outcome=outcome,
                                 max_attitude_deg=att, max_pitch_deg=np.degrees(np.abs(T['pitch']).max()),
                                 lowest_altitude_m=-T['z'].max(), rate_hz=1 / p.mean(), p99_period_ms=1e3 * np.percentile(p, 99),
                                 cert_gap_pct=100 * T['plan_expired'].mean(),
                                 rollout_ms=1e3 * np.median(log.timing['rollout_compute']),
                                 ctrl_us=1e6 * T['ctrl_comp_time'].median()))
                T[['time', 'y', 'z', 'roll', 'pitch', 'thrust', 'pitch_rate', 'ref_py', 'ref_pz']].to_csv(
                    DATA / 'round3' / f'{f.stem}.csv.gz', index=False, float_format='%.5g')
    df = pd.DataFrame(rows)
    df.to_csv(DATA / 'round3_flights.csv', index=False, float_format='%.2f')

    summary = df.groupby(['config', 'label'], sort=False).agg(
        flights=('run', 'size'), upsets=('outcome', lambda s: int((s == 'UPSET').sum())),
        touches=('outcome', lambda s: int((s == 'touch').sum())),
        max_pitch_med=('max_pitch_deg', 'median'), rate=('rate_hz', 'mean'), p99=('p99_period_ms', 'mean'),
        gaps=('cert_gap_pct', 'mean'), rollout=('rollout_ms', 'mean'), ctrl_us=('ctrl_us', 'mean')).reset_index()
    summary.to_csv(DATA / 'round3_summary.csv', index=False, float_format='%.2f')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    # Figure: max pitch per flight, by configuration
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    colors = {'ok': '#1b7837', 'touch': '#e08214', 'UPSET': '#b2182b'}
    labels = [l for p, l in CONFIGS if (df['config'] == p).any()]
    for i, (prefix, label) in enumerate([c for c in CONFIGS if (df['config'] == c[0]).any()]):
        d = df[df['config'] == prefix]
        for _, r in d.iterrows():
            ax.scatter(min(r['max_pitch_deg'], 180), i + np.random.uniform(-0.15, 0.15), color=colors[r['outcome']], s=18)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlabel('max |pitch| during the RTA phase (deg, log scale; capped at 180)')
    for k, c in colors.items():
        ax.scatter([], [], color=c, label=k)
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(HERE / 'round3_pitch.pdf')
    plt.close(fig)

    # Figure: pitch time series, old vs new NR reference
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, prefix, title in ((axes[0], 'r4_old_nrref', 'NR reference y = 0, z = -12.5 + 0.1 t'),
                              (axes[1], 'r4_nrplan', 'NR reference from the RTA plan')):
        for f in sorted((DATA / 'round3').glob(f'{prefix}_[0-9].csv.gz')):
            d = pd.read_csv(f)
            ax.plot(d['time'], np.degrees(d['pitch']), lw=0.7)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('time (s)')
        ax.set_ylim(-90, 90)
    axes[0].set_ylabel('pitch (deg)')
    fig.tight_layout()
    fig.savefig(HERE / 'round3_pitch_timeseries.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         '/home/egmc/Projects/old_projects/OJCSYS27/src/data_analysis/log_files/px4_rta_mm_gpr')
