from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def simpson_weights_like_cpp(n: int) -> np.ndarray:
    """Legacy helper retained for backward compatibility.

    The modern equatorial model uses uniform angular averaging rather than
    the historical 2/4-weight rule inherited from older C++ scripts.
    """
    return np.ones(n, dtype=np.float64)


def mean_radius_equatorial(radius: np.ndarray) -> float:
    """Return equatorial contour mean radius r_avg(t).

    This matches the discrete counterpart of:
        r_avg(t) = (1 / 2pi) * integral r_eq(phi, t) dphi
    """
    r = np.asarray(radius, dtype=np.float64)
    return float(np.mean(r))


def angular_autocorrelation_equatorial(radius: np.ndarray) -> np.ndarray:
    """Compute xi(gamma, t) from equatorial contour data.

    Uses periodic boundary conditions and the discrete estimator:
        xi[j] = mean((r - r_avg) * (roll(r, -j) - r_avg)) / r_avg**2
    """
    r = np.asarray(radius, dtype=np.float64)
    r_avg = mean_radius_equatorial(r)
    centered = r - r_avg

    xi = np.empty(r.size, dtype=np.float64)
    for j in range(r.size):
        shifted = np.roll(r, -j)
        xi[j] = np.mean(centered * (shifted - r_avg)) / (r_avg**2)

    return xi


def mean_radius_like_cpp(radius: np.ndarray) -> float:
    """Backward-compatible wrapper for equatorial average radius."""
    return mean_radius_equatorial(radius)


def angular_autocorrelation_like_cpp(radius: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper for equatorial autocorrelation."""
    return angular_autocorrelation_equatorial(radius)


def compute_angular_autocorrelation_series(r_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute xi(gamma, t) for an equatorial contour time series.

    This module intentionally analyzes only the equatorial contour
    (theta = pi/2), consistent with microscopy-based vesicle experiments.
    """
    r_eq = np.asarray(r_eq, dtype=np.float64)
    n_frames, n_phi = r_eq.shape

    xi_t = np.empty((n_frames, n_phi), dtype=np.float64)
    mean_r = np.empty(n_frames, dtype=np.float64)
    gamma = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    for t in range(n_frames):
        frame = r_eq[t]
        mean_r[t] = mean_radius_equatorial(frame)
        xi_t[t] = angular_autocorrelation_equatorial(frame)

    return gamma, xi_t, mean_r


def time_average_xi(xi_t: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(xi_t, dtype=np.float64), axis=0)


def extract_build_csv(path: str | Path, has_header: bool = True) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if has_header and lines:
        lines = lines[1:]

    rows = []
    expected = None
    for line in lines:
        parts = [p for p in line.split(",") if p != ""]
        if not parts:
            continue
        if expected is None:
            expected = len(parts)
        if len(parts) == expected:
            rows.append(parts)

    return np.asarray(rows, dtype=np.float64)
