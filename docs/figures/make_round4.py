"""Round-4 figure: altitude during the RTA phase, old vs new reference tuning, with floor, goal and backup events.

    python3 docs/figures/make_round4.py [log_dir]

Copies compact per-tick data to docs/data/round4/ so the figure can be regenerated without the HDF5 logs.
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
OUT = HERE.parent / 'data' / 'round4'
sys.path.insert(0, str(HERE.parents[1] / 'packages' / 'px4_rta_mm_gpr'))
sys.path.insert(0, str(HERE.parents[1] / 'packages' / 'flight_recorder'))  # git submodule

PANELS = [('r5_defaults', '(a) goal 0.6 m, old Q_ref', 0.6, 0.3),
          ('r5_tuned', '(b) goal 1.0 m, new Q_ref', 1.0, 0.3),
          ('r5_negtest', '(c) negative test, floor 1.2 m', 1.0, 1.2)]


def export(log_dir):
    from px4_rta_mm_gpr.flight_log import FlightLog
    OUT.mkdir(parents=True, exist_ok=True)
    for prefix, *_ in PANELS:
        for f in sorted(Path(log_dir).glob(f'{prefix}_[0-9].h5')):
            with FlightLog(str(f)) as log:
                T = log.ticks[['time', 'y', 'z', 'ref_pz', 'tube_pz_hi']].copy()
                T.to_csv(OUT / f'{f.stem}.csv.gz', index=False, float_format='%.5g')
                log.events.to_csv(OUT / f'{f.stem}.events.csv', index=False)


def figure():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.4, 2.6), sharey=True)
    for ax, (prefix, title, goal, floor) in zip(axes, PANELS):
        for f in sorted(OUT.glob(f'{prefix}_[0-9].csv.gz')):
            d = pd.read_csv(f)
            line, = ax.plot(d['time'], -d['z'], lw=0.9)
            ax.plot(d['time'], -d['tube_pz_hi'], lw=0.5, ls=':', color=line.get_color())
            ev = pd.read_csv(str(f).replace('.csv.gz', '.events.csv'))
            for t in ev.get('time', []):
                ax.plot([t], [-d['z'].iloc[-1]], marker='x', color='k', ms=6)
        ax.axhline(goal, color='#1b7837', lw=0.8, ls='--')
        ax.axhspan(0, floor, color='#b2182b', alpha=0.12, lw=0)
        ax.set_title(title, fontsize=8, loc='left')
        ax.set_xlabel('time (s)')
        ax.set_ylim(0, 13)
    axes[0].set_ylabel('altitude (m)')
    fig.tight_layout()
    fig.savefig(HERE / 'round4_altitude.pdf')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1 or not OUT.exists():
        export(sys.argv[1] if len(sys.argv) > 1 else
               '/home/egmc/Projects/old_projects/OJCSYS27/src/data_analysis/log_files/px4_rta_mm_gpr')
    figure()
