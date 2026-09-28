"""Round-5 figures: Python vs C++ control-loop timing, and the planner-kill test.

    python3 docs/figures/make_round5.py [log_dir]
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
OUT = HERE.parent / 'data' / 'round5'
sys.path.insert(0, str(HERE.parents[1] / 'packages' / 'px4_rta_mm_gpr'))

GROUPS = [('r5_tuned', 'Python loop (events executor + process rollouts)', '#2166ac'),
          ('r6_cpp', 'C++ fast loop + Python planner', '#b2182b')]


def export(log_dir):
    from px4_rta_mm_gpr.flight_log import FlightLog
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for prefix in ['r5_tuned', 'r6_cpp', 'r6_cpp_negtest', 'r6_cpp_plannerkill']:
        for f in sorted(Path(log_dir).glob(f'{prefix}_[0-9].h5')):
            with FlightLog(str(f)) as log:
                T = log.ticks
                T[['time', 'y', 'z', 'control_period', 'ctrl_comp_time', 'plan_expired', 'tube_pz_hi']].to_csv(
                    OUT / f'{f.stem}.csv.gz', index=False, float_format='%.6g')
                log.events.to_csv(OUT / f'{f.stem}.events.csv', index=False)
                p = T['control_period'].to_numpy()[1:]
                rows.append(dict(flight=f.stem, ticks=len(T), rate_hz=1 / np.nanmean(p),
                                 p50_ms=1e3 * np.nanmedian(p), p99_ms=1e3 * np.nanpercentile(p, 99),
                                 max_ms=1e3 * np.nanmax(p), ctrl_us_p50=1e6 * T['ctrl_comp_time'].median(),
                                 ctrl_us_max=1e6 * T['ctrl_comp_time'].max(),
                                 final_alt=-T['z'].iloc[-150:].mean(), lowest_alt=-T['z'].max(),
                                 gaps_pct=100 * T['plan_expired'].mean(),
                                 events='; '.join(f"{r.time:.2f}s {r.kind}" for r in log.events.itertuples())))
    pd.DataFrame(rows).to_csv(HERE.parent / 'data' / 'round5_flights.csv', index=False, float_format='%.3f')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.2f}'))


def figures():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'width_ratios': [1.2, 1]})
    for prefix, label, color in GROUPS:
        cs = [pd.read_csv(f)['ctrl_comp_time'].to_numpy() * 1e6 for f in sorted(OUT.glob(f'{prefix}_[0-9].csv.gz'))]
        c = np.sort(np.concatenate(cs))
        ax.step(c, np.arange(1, c.size + 1) / c.size, where='post', color=color, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('control-law computation per tick (us, log scale)')
    ax.set_ylabel('fraction of ticks')
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    ax.set_title('(a) 5 flights each, same mission', loc='left', fontsize=8)

    f = OUT / 'r6_cpp_plannerkill_1.csv.gz'
    if f.exists():
        d = pd.read_csv(f)
        bx.plot(d['time'], -d['z'], color='#b2182b', lw=1.0, label='altitude (logged by the planner)')
        bx.plot(d['time'], -d['tube_pz_hi'], color='#b2182b', lw=0.6, ls=':', label='certified tube, lowest altitude')
        bx.axvline(d['time'].iloc[-1], color='k', lw=0.8, ls='--')
        bx.text(d['time'].iloc[-1], 11, ' planner killed', fontsize=7)
        bx.axvline(23.45, color='#1b7837', lw=0.8)
        bx.text(23.45, 8, ' C++ watchdog:\n LAND (t = 23.45 s)', fontsize=7, color='#1b7837')
        bx.set_xlabel('mission time (s)')
        bx.set_ylabel('altitude (m)')
        bx.set_title('(b) planner killed during the RTA phase', loc='left', fontsize=8)
        bx.set_xlim(14.5, 25.5)
        bx.legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(HERE / 'round5_cpp.pdf')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1 or not OUT.exists():
        export(sys.argv[1] if len(sys.argv) > 1 else
               '/home/egmc/Projects/old_projects/OJCSYS27/src/data_analysis/log_files/px4_rta_mm_gpr')
    figures()
