"""Read RTA-MM-GPR flight logs, and export the legacy ROS2Logger CSV.

    from px4_rta_mm_gpr.flight_log import FlightLog
    log = FlightLog('log.h5')
    log.ticks                 # pandas DataFrame, one row per RTA control tick
    log.plan(12)              # dict with the full tube / reference / inputs of plan #12
    log.plan_at(21.3)         # the plan that was in use at t = 21.3 s
    log.tube_from(21.3)       # remaining tube (lower/upper) of that plan from t = 21.3 s onwards

Reads both layouts:
  * flight_recorder files (format="flight_recorder"; written since the switch to the flight_recorder submodule),
    through flight_recorder.FlightLog;
  * the earlier self-contained layout (/ticks, /wind, /gains, /plans/<seq:05d>, /timing, /events), so older logs
    and the figures made from them keep working.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Legacy log geometry (see the original rta_mm_gpr_node.py)
_TUBE_EXTENT, _TUBE_SKIP = 23, 2
_TUBE_POS_INDICES = [0, 1, 5, 6]          # py_lo, pz_lo, py_hi, pz_hi


class FlightLog:
    def __init__(self, path: str):
        import h5py
        self.path = path
        with h5py.File(path, 'r') as f:
            fmt = _py(f.attrs.get('format', ''))
        self.legacy_format = fmt != 'flight_recorder'
        if self.legacy_format:
            self._fr = None
            self._f = h5py.File(path, 'r')
            self.metadata = {k: _py(v) for k, v in self._f.attrs.items()}
            self.ticks = _frame(self._f['ticks'])
            self.wind = _frame(self._f['wind'])
            self.gains = _frame(self._f['gains'])
            self.plan_seqs = np.array(sorted(int(k) for k in self._f['plans'].keys()))
            self.timing = {k: self._f['timing'][k][()] for k in self._f['timing']} if 'timing' in self._f else {}
            self.events = (pd.DataFrame({k: [_py(v) for v in self._f['events'][k][()]] for k in self._f['events']})
                           if 'events' in self._f else pd.DataFrame(columns=['time', 'kind', 'detail']))
        else:
            from flight_recorder import FlightLog as _FRLog
            self._f = None
            self._fr = _FRLog(path)
            self.metadata = self._fr.metadata
            self.ticks = self._fr['ticks']
            self.wind = self._fr['wind']
            self.gains = self._fr['gains']
            groups = self._fr.record_groups
            self.plan_seqs = np.array([int(k) for k in self._fr.record_keys('plans')] if 'plans' in groups else [],
                                      dtype=int)
            self.timing = ({k: self._fr.record('timing', k)['values'] for k in self._fr.record_keys('timing')}
                           if 'timing' in groups else {})
            self.events = self._fr.events

    def close(self):
        (self._fr or self._f).close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def plan(self, seq: int) -> dict:
        if self._fr is not None:
            return self._fr.record('plans', int(seq))
        g = self._f['plans'][f'{int(seq):05d}']
        out = {k: g[k][()] for k in g.keys()}
        out.update({k: _py(v) for k, v in g.attrs.items()})
        return out

    def plan_at(self, t: float) -> Optional[dict]:
        """The plan the controller was using at time t (from the tick table)."""
        rows = self.ticks[self.ticks['time'] <= t]
        if rows.empty:
            return None
        return self.plan(int(rows['plan_seq'].iloc[-1]))

    def tube_from(self, t: float) -> Optional[pd.DataFrame]:
        """Remaining reachable tube of the plan in use at time t, as a DataFrame indexed by time."""
        p = self.plan_at(t)
        if p is None:
            return None
        n = p['reachable_tube'].shape[0]
        times = p['t_start'] + p['dt'] * np.arange(n)
        names = ['py', 'pz', 'h', 'v', 'theta']
        df = pd.DataFrame(p['reachable_tube'], columns=[f'{s}_lo' for s in names] + [f'{s}_hi' for s in names])
        df.insert(0, 'time', times)
        return df[df['time'] >= t].reset_index(drop=True)

    def to_legacy_csv(self, path: str) -> str:
        """Write the exact column layout the old ROS2Logger produced, so existing notebooks keep working.

        Per control tick the old logger stored 12 reference samples (y_ref, z_ref, yaw_ref) and 12 tube
        rows (save_tube_*); those columns are therefore ~12x longer than the per-tick columns and the
        shorter columns are NaN-padded, as before. The data is reconstructed from the stored plans.
        """
        t = self.ticks
        cols = {
            'time': t['time'], 'x': t['x'], 'y': t['y'], 'z': t['z'], 'yaw': t['yaw'],
            'ctrl_comp_time': t['ctrl_comp_time'], 'rollout_comptime': t['rollout_comptime'],
        }
        y_ref, z_ref, yaw_ref, tube_rows = [], [], [], []
        cache = {}
        for seq, idx in zip(t['plan_seq'].to_numpy(dtype=int), t['traj_idx'].to_numpy(dtype=int)):
            if seq not in cache:
                cache = {seq: self.plan(seq)}
            p = cache[seq]
            rows = slice(idx, idx + _TUBE_EXTENT, _TUBE_SKIP)
            y_ref.append(p['rollout_ref'][rows, 0])
            z_ref.append(p['rollout_ref'][rows, 1])
            yaw_ref.append(p['rollout_ref'][rows, 4])
            start = int(p['latency'] // 0.01) + 1
            tube_rows.append(p['reachable_tube'][start:start + _TUBE_EXTENT:_TUBE_SKIP][:, _TUBE_POS_INDICES])
        cols['y_ref'] = np.concatenate(y_ref) if y_ref else np.array([])
        cols['z_ref'] = np.concatenate(z_ref) if z_ref else np.array([])
        cols['yaw_ref'] = np.concatenate(yaw_ref) if yaw_ref else np.array([])
        for name in ('throttle', 'roll_rate', 'pitch_rate', 'yaw_rate'):
            cols[name] = t[name]
        tube = np.vstack(tube_rows) if tube_rows else np.empty((0, 4))
        # ROS2Logger sorted a vector log's sub-columns alphabetically: pyH, pyL, pzH, pzL
        cols['save_tube_pyH'], cols['save_tube_pyL'] = tube[:, 2], tube[:, 0]
        cols['save_tube_pzH'], cols['save_tube_pzL'] = tube[:, 3], tube[:, 1]
        for name in ('wy', 'wz', 'rollout_latency', 'plan_age', 'plan_expired'):
            cols[name] = t[name]
        n = max(len(v) for v in cols.values())
        out = pd.DataFrame({k: np.pad(np.asarray(v, dtype=float), (0, n - len(v)), constant_values=np.nan)
                            for k, v in cols.items()})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out.to_csv(path, index=False, na_rep='nan')
        return path


def _frame(group) -> pd.DataFrame:
    return pd.DataFrame({k: group[k][()] for k in group.keys()})


def _py(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def main():
    """CLI: summary of a flight log and optional legacy CSV export."""
    import argparse
    parser = argparse.ArgumentParser(description='Inspect a px4_rta_mm_gpr flight log (.h5)')
    parser.add_argument('path')
    parser.add_argument('--csv', help='also write the legacy ROS2Logger CSV to this path')
    args = parser.parse_args()
    with FlightLog(args.path) as log:
        t = log.ticks
        print(f"{args.path}\n  metadata: {log.metadata}")
        if len(t):
            p = t['control_period'].to_numpy()[1:]
            print(f"  ticks: {len(t)}  t=[{t['time'].iloc[0]:.2f}, {t['time'].iloc[-1]:.2f}] s  "
                  f"rate {1 / np.nanmean(p):.1f} Hz  p99 period {1e3 * np.nanpercentile(p, 99):.1f} ms")
        print(f"  plans: {len(log.plan_seqs)}  wind samples: {len(log.wind)}  gain updates: {len(log.gains)}")
        if args.csv:
            print(f"  legacy CSV -> {log.to_legacy_csv(args.csv)}")


if __name__ == '__main__':
    main()
