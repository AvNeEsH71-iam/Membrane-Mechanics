from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import simpson
from scipy.special import eval_legendre


def project_legendre_spectrum(
    xi_gamma: np.ndarray,
    gamma: np.ndarray,
    l_min: int,
    l_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project equatorial xi(gamma) onto Legendre modes.

    Uses the equivalent form
        b_l = (2l+1)/2 * integral_{-1}^{1} xi(x) P_l(x) dx,
    with x = cos(gamma), which corresponds to
        b_l = integral xi(gamma) P_l(cos(gamma)) d(cos(gamma)).

    This intentionally works with equatorial-contour correlations only.
    """
    xi = np.asarray(xi_gamma, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)

    mask = (g >= 0.0) & (g <= np.pi)
    g_half = g[mask]
    xi_half = xi[mask]

    if g_half.size < 3:
        raise ValueError("Need gamma samples over [0, pi] to project Legendre modes.")

    x = np.cos(g_half)

    # Integrate over x in ascending order for numerical stability.
    order = np.argsort(x)
    x_sorted = x[order]
    xi_sorted = xi_half[order]

    l_values = np.arange(l_min, l_max + 1)
    b_values = np.empty_like(l_values, dtype=np.float64)

    for i, l in enumerate(l_values):
        p_l = eval_legendre(l, x_sorted)
        integrand = xi_sorted * p_l
        b_values[i] = 0.5 * (2 * l + 1) * simpson(y=integrand, x=x_sorted)

    return l_values, b_values


def project_legendre_series(
    xi_t: np.ndarray,
    gamma: np.ndarray,
    l_min: int,
    l_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xi_t = np.asarray(xi_t, dtype=np.float64)
    l_vals = np.arange(l_min, l_max + 1)
    b_t = np.empty((xi_t.shape[0], l_vals.size), dtype=np.float64)

    for t in range(xi_t.shape[0]):
        _, b_t[t] = project_legendre_spectrum(xi_t[t], gamma, l_min=l_min, l_max=l_max)

    return l_vals, b_t
