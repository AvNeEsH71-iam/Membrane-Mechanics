from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares


K_B = 1.380649e-23


@dataclass
class FitResult:
    kappa: float
    sigma: float
    kappa_std: float
    sigma_std: float
    chi2: float
    dof: int
    success: bool
    message: str


def p_l(l: np.ndarray) -> np.ndarray:
    l = np.asarray(l, dtype=np.float64)
    return 4.0 * np.pi * (l - 1.0) * (l + 2.0) * (l + 1.0) * l / (2.0 * l + 1.0)


def q_l(l: np.ndarray) -> np.ndarray:
    l = np.asarray(l, dtype=np.float64)
    return 4.0 * np.pi * (l - 1.0) * (l + 2.0) / (2.0 * l + 1.0)


def theoretical_B_l(l_values: np.ndarray, kappa: float, sigma: float, temperature: float) -> np.ndarray:
    l_vals = np.asarray(l_values, dtype=np.float64)
    j = K_B * temperature / kappa
    return j / (p_l(l_vals) + sigma * q_l(l_vals))


def fit_kappa_sigma(
    l_values: np.ndarray,
    b_values: np.ndarray,
    b_sem: np.ndarray,
    temperature: float,
    kappa0: float,
    sigma0: float,
) -> Tuple[FitResult, np.ndarray]:
    """Fit kappa and sigma using weighted least squares over Legendre modes.

    Constraints:
    - kappa > 0 (via log-parameterization)
    - sigma > -l(l+1) for all fitted l values
    """
    l_vals = np.asarray(l_values, dtype=np.float64)
    b_vals = np.asarray(b_values, dtype=np.float64)
    sem = np.asarray(b_sem, dtype=np.float64)

    if np.any(l_vals < 2):
        raise ValueError("All fitted l values must satisfy l >= 2.")

    fallback_sem = np.max([np.median(np.abs(b_vals)) * 1e-3, 1e-14])
    sem = np.where(~np.isfinite(sem) | (sem <= 0.0), fallback_sem, sem)

    sigma_lower = float(-np.min(l_vals * (l_vals + 1.0)) + 1e-9)
    sigma_upper = 1e4

    def residual(params: np.ndarray) -> np.ndarray:
        log_kappa, sigma = params
        kappa = np.exp(log_kappa)
        model = theoretical_B_l(l_vals, kappa=kappa, sigma=sigma, temperature=temperature)
        return (b_vals - model) / sem

    x0 = np.array([np.log(max(kappa0, 1e-30)), sigma0], dtype=np.float64)
    x0[1] = np.clip(x0[1], sigma_lower + 1e-9, sigma_upper - 1e-9)

    res = least_squares(
        residual,
        x0,
        bounds=([-80.0, sigma_lower], [80.0, sigma_upper]),
        method="trf",
    )

    log_kappa_opt, sigma_opt = res.x
    kappa_opt = float(np.exp(log_kappa_opt))
    chi2 = float(np.sum(res.fun**2))
    dof = max(1, b_vals.size - 2)

    kappa_std = np.nan
    sigma_std = np.nan
    if res.jac is not None and res.jac.size > 0:
        jtj = res.jac.T @ res.jac
        try:
            cov = np.linalg.inv(jtj) * (chi2 / dof)
            log_kappa_std = float(np.sqrt(max(cov[0, 0], 0.0)))
            sigma_std = float(np.sqrt(max(cov[1, 1], 0.0)))
            kappa_std = float(kappa_opt * log_kappa_std)
        except np.linalg.LinAlgError:
            cov = np.full((2, 2), np.nan)
    else:
        cov = np.full((2, 2), np.nan)

    fit = FitResult(
        kappa=kappa_opt,
        sigma=float(sigma_opt),
        kappa_std=float(kappa_std),
        sigma_std=float(sigma_std),
        chi2=chi2,
        dof=dof,
        success=bool(res.success),
        message=str(res.message),
    )
    return fit, cov
