"""Wind fields for the numerical simulation.

``EvolvingWindField`` reproduces the time-varying disturbance of the original numerical studies
(rta_evanns_GPR2_ANIM.ipynb): a GP-mean profile through fixed anchor points which, within each second, blends from the
current profile f toward sqrt(1 - eps) f + sqrt(eps) g, where g gets fresh random anchor values every second.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Anchor values of the original studies (profiles at heights -2, 0, ..., 12)
PAPER_ANCHORS = np.arange(-2.0, 13.0, 2.0)
PAPER_F_Y = [-0.66005468347438, 1.71910828159421, -2.73909534657961, 3.76474225408493, 3.65733558594356,
             -0.116994810217270, 2.40224375111040, -2.86490929098228]
PAPER_G_Y = [2.80494480741922, -1.65967058757266, -2.21529178685802, -2.41828274757721, -0.976879063072681,
             -2.89172701466510, -3.91337304908981, -3.33944976072828]
PAPER_F_Z = [1.52494480741922, -2.35967058757266, 3.11529178685802, -1.21828274757721, 2.476879063072681,
             -3.19172701466510, 1.81337304908981, -2.53944976072828]
PAPER_G_Z = [-1.90494480741922, 2.85967058757266, -3.41529178685802, 1.71828274757721, -2.276879063072681,
             3.49172701466510, -1.51337304908981, 2.83944976072828]


def _gp_mean_weights(anchors: np.ndarray, sigma_f: float, length: float, sigma_n: float):
    """Precompute a squared-exponential GP mean through anchor points: mean(s) = k(s, anchors) @ alpha."""
    K = sigma_f * np.exp(-0.5 * (anchors[:, None] - anchors[None, :]) ** 2 / length ** 2)
    K_inv = np.linalg.inv(K + sigma_n ** 2 * np.eye(len(anchors)))

    def mean(s, values):
        k = sigma_f * np.exp(-0.5 * (np.asarray(s, dtype=float)[..., None] - anchors) ** 2 / length ** 2)
        return k @ (K_inv @ np.asarray(values, dtype=float))
    return mean


@dataclass
class EvolvingWindField:
    """A 1-D wind profile w(t, s) (s = altitude or lateral position), evolving every second."""
    f: Sequence[float] = field(default_factory=lambda: list(PAPER_F_Y))
    g: Sequence[float] = field(default_factory=lambda: list(PAPER_G_Y))
    anchors: Sequence[float] = field(default_factory=lambda: list(PAPER_ANCHORS))
    epsilon: float = 0.25
    update_magnitude: float = 1.5        # std of the fresh random anchors drawn each second
    scale: float = 1.0                   # multiplies every value (acceleration -> force: scale = mass * factor)
    seed: int = 0

    def __post_init__(self):
        self._anchors = np.asarray(self.anchors, dtype=float)
        self._mean = _gp_mean_weights(self._anchors, sigma_f=5.0, length=2.0, sigma_n=0.01)
        self._rng = np.random.default_rng(self.seed)
        # profiles (f_k, g_k) of every second k seen so far: the field is a pure function of t (any query order)
        self._history = [(np.asarray(self.f, dtype=float), np.asarray(self.g, dtype=float))]

    def _profiles(self, second: int):
        while len(self._history) <= second:
            f, g = self._history[-1]
            # the profile at the end of a second becomes the next f; g gets fresh random anchors
            self._history.append((np.sqrt(1 - self.epsilon) * f + np.sqrt(self.epsilon) * g,
                                  self._rng.normal(scale=self.update_magnitude, size=f.shape)))
        return self._history[second]

    def __call__(self, t: float, s) -> np.ndarray:
        """Wind at time t >= 0 (s) and position(s) s."""
        second = int(np.floor(max(t, 0.0)))
        f, g = self._profiles(second)
        frac = t - second
        values = (1.0 - frac) * f + frac * (np.sqrt(1 - self.epsilon) * f + np.sqrt(self.epsilon) * g)
        return self.scale * self._mean(s, values)


def paper_winds(scale: float = 1.0, seed: int = 0):
    """The two-wind scenario of the original studies: y-wind(altitude) and z-wind(lateral position)."""
    wy = EvolvingWindField(PAPER_F_Y, PAPER_G_Y, scale=scale, seed=seed)
    wz = EvolvingWindField(PAPER_F_Z, PAPER_G_Z, anchors=PAPER_ANCHORS - 6.0, scale=scale, seed=seed + 1)
    return wy, wz


def calm():
    """No wind."""
    zero = EvolvingWindField(np.zeros(8), np.zeros(8), update_magnitude=0.0)
    return zero, EvolvingWindField(np.zeros(8), np.zeros(8), update_magnitude=0.0)
