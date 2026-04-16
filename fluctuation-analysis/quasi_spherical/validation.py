from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from scipy.special import eval_legendre

from .fitting import theoretical_B_l
from .generator import GeneratorConfig, generate_equatorial_series
from .legendre import project_legendre_spectrum


@dataclass
class ValidationChecks:
    kappa_reduces_fluctuations: bool
    sigma_suppresses_low_l: bool
    xi_reconstruction_small_error: bool
    kappa_variance_ratio_high_over_low: float
    sigma_suppression_l3: float
    sigma_suppression_l10: float
    xi_reconstruction_rel_rmse: float


def _reconstruct_xi(gamma: np.ndarray, l_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    xi_rec = np.zeros_like(gamma, dtype=np.float64)
    for l, b in zip(l_values, b_values):
        xi_rec += b * eval_legendre(int(l), np.cos(gamma))
    return xi_rec


def compute_validation_checks(
    base_config: GeneratorConfig,
    gamma: np.ndarray,
    xi_avg: np.ndarray,
    l_values: np.ndarray,
    b_values: np.ndarray,
) -> Dict[str, float | bool]:
    """Run lightweight physics and numerical consistency checks.

    Checks implemented:
    - Increasing kappa reduces fluctuation variance
    - Increasing sigma suppresses low-l modes more strongly than high-l
    - Legendre reconstruction error for xi(gamma) is small over [0, pi]
    """
    cfg_lo = GeneratorConfig(
        radius=base_config.radius,
        n_phi=min(base_config.n_phi, 256),
        n_frames=min(base_config.n_frames, 800),
        l_max=base_config.l_max,
        kappa=max(base_config.kappa * 0.5, 1e-30),
        sigma=base_config.sigma,
        temperature=base_config.temperature,
        seed=base_config.seed,
    )
    cfg_hi = GeneratorConfig(
        radius=cfg_lo.radius,
        n_phi=cfg_lo.n_phi,
        n_frames=cfg_lo.n_frames,
        l_max=cfg_lo.l_max,
        kappa=max(base_config.kappa * 2.0, 1e-30),
        sigma=cfg_lo.sigma,
        temperature=cfg_lo.temperature,
        seed=cfg_lo.seed,
    )

    _, _, u_lo = generate_equatorial_series(cfg_lo)
    _, _, u_hi = generate_equatorial_series(cfg_hi)
    var_lo = float(np.var(u_lo))
    var_hi = float(np.var(u_hi))
    kappa_ratio = var_hi / max(var_lo, 1e-30)

    l_ref = np.arange(3, 11)
    b_sigma_low = theoretical_B_l(l_ref, kappa=base_config.kappa, sigma=0.2, temperature=base_config.temperature)
    b_sigma_high = theoretical_B_l(l_ref, kappa=base_config.kappa, sigma=8.0, temperature=base_config.temperature)
    suppression_l3 = float(b_sigma_high[0] / max(b_sigma_low[0], 1e-30))
    suppression_l10 = float(b_sigma_high[-1] / max(b_sigma_low[-1], 1e-30))

    # Use l=0..l_max projection for reconstruction quality.
    # The l=0 baseline term is important for accurately reconstructing xi.
    l_recon_max = int(np.max(l_values)) if l_values.size > 0 else 10
    l_recon, b_recon = project_legendre_spectrum(xi_avg, gamma, l_min=0, l_max=l_recon_max)
    xi_rec = _reconstruct_xi(gamma, l_recon, b_recon)
    mask = (gamma >= 0.0) & (gamma <= np.pi)
    rmse = float(np.sqrt(np.mean((xi_avg[mask] - xi_rec[mask]) ** 2)))
    rel_rmse = rmse / (float(np.sqrt(np.mean(xi_avg[mask] ** 2))) + 1e-30)

    checks = ValidationChecks(
        kappa_reduces_fluctuations=bool(var_hi < var_lo),
        sigma_suppresses_low_l=bool((suppression_l3 < 1.0) and (suppression_l3 < suppression_l10)),
        xi_reconstruction_small_error=bool(rel_rmse < 0.02),
        kappa_variance_ratio_high_over_low=kappa_ratio,
        sigma_suppression_l3=suppression_l3,
        sigma_suppression_l10=suppression_l10,
        xi_reconstruction_rel_rmse=rel_rmse,
    )

    return asdict(checks)
