"""Lightweight, thread-safe timing statistics for periodic callbacks.

For each monitored loop we record, per call:
  * the *period* actually achieved (time since the previous call started), and
  * the *execution time* of the callback body.

A call whose period exceeds ``overrun_factor * nominal_period`` counts as a deadline miss.
These are the numbers that show whether separating control from rollouts worked: with a
single-threaded executor the control loop's max period roughly equals the rollout time.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np


class LoopStats:
    def __init__(self, name: str, nominal_period: float, overrun_factor: float = 1.5,
                 capacity: int = 200_000):
        self.name = name
        self.nominal_period = nominal_period
        self.overrun_factor = overrun_factor
        self._lock = threading.Lock()
        self._periods = np.zeros(capacity)
        self._exec = np.zeros(capacity)
        self._n = 0
        self._last_start = None
        self._capacity = capacity

    def record(self, period: float, exec_time: float) -> None:
        """Add a sample measured elsewhere (e.g. by the C++ fast loop, reported in its ControlTick)."""
        with self._lock:
            if self._n < self._capacity:
                self._periods[self._n] = np.nan if self._n == 0 else period
                self._exec[self._n] = exec_time
                self._n += 1

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            with self._lock:
                if self._n < self._capacity:
                    period = np.nan if self._last_start is None else start - self._last_start
                    self._periods[self._n] = period
                    self._exec[self._n] = end - start
                    self._n += 1
                self._last_start = start

    def periods(self) -> np.ndarray:
        with self._lock:
            return self._periods[1:self._n].copy()

    def exec_times(self) -> np.ndarray:
        with self._lock:
            return self._exec[:self._n].copy()

    def summary(self) -> str:
        p, e = self.periods(), self.exec_times()
        if p.size == 0:
            return f'{self.name:>10}: no samples'
        misses = int(np.sum(p > self.overrun_factor * self.nominal_period))
        return (f'{self.name:>10}: n={e.size:6d}  rate={1.0 / np.mean(p):7.1f} Hz '
                f'(nominal {1.0 / self.nominal_period:.0f})  '
                f'period p50/p99/max = {1e3 * np.median(p):7.2f}/{1e3 * np.percentile(p, 99):7.2f}/'
                f'{1e3 * np.max(p):8.2f} ms  exec p50/max = {1e3 * np.median(e):6.2f}/'
                f'{1e3 * np.max(e):8.2f} ms  misses={misses} ({100.0 * misses / p.size:.1f}%)')
