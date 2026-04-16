from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y

    def sph_harm(m: int, l: int, phi: np.ndarray, theta: float) -> np.ndarray:
        # scipy.special.sph_harm_y uses argument order (l, m, theta, phi).
        return sph_harm_y(l, m, theta, phi)


K_B = 1.380649e-23


@dataclass
class GeneratorConfig:
    radius: float = 10.0
    n_phi: int = 512
    n_frames: int = 1000
    l_max: int = 30
    kappa: float = 8.0e-20
    sigma: float = 0.5
    temperature: float = 300.0
    seed: Optional[int] = 12345
    single_mode: Optional[Tuple[int, int]] = None


def _thermal_variance_l(l: int, kappa: float, sigma: float, temperature: float) -> float:
    denom = (l + 2.0) * (l - 1.0) * (l * (l + 1.0) + sigma)
    denom = max(denom, 1e-14)
    return (K_B * temperature / kappa) / denom


def _sample_mode_amplitudes(
    rng: np.random.Generator,
    l_max: int,
    n_frames: int,
    kappa: float,
    sigma: float,
    temperature: float,
    single_mode: Optional[Tuple[int, int]] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    modes: Dict[Tuple[int, int], np.ndarray] = {}

    if single_mode is not None:
        selected_l, selected_m = single_mode
    else:
        selected_l, selected_m = -1, -1

    for l in range(2, l_max + 1):
        var_l = _thermal_variance_l(l, kappa, sigma, temperature)

        for m in range(0, l + 1):
            if single_mode is not None and (l != selected_l or m != abs(selected_m)):
                amplitude = np.zeros(n_frames, dtype=np.complex128)
            else:
                if m == 0:
                    amplitude = rng.normal(0.0, np.sqrt(var_l), size=n_frames).astype(np.complex128)
                else:
                    # For complex Gaussian, Re/Im each use var/2 so E[|u_lm|^2] = var_l.
                    re = rng.normal(0.0, np.sqrt(var_l / 2.0), size=n_frames)
                    im = rng.normal(0.0, np.sqrt(var_l / 2.0), size=n_frames)
                    amplitude = re + 1j * im

            modes[(l, m)] = amplitude
            if m > 0:
                modes[(l, -m)] = ((-1) ** m) * np.conj(amplitude)

    return modes


def generate_equatorial_series(config: GeneratorConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic equatorial contours r_eq(phi, t).

    Model note:
    - This function intentionally evaluates only theta = pi/2 (equator).
    - This matches contour microscopy where only the vesicle cross-section
      is experimentally observed.
    - Spherical-harmonic mode amplitudes u_lm are projected onto the
      equatorial contour; no full-sphere reconstruction is performed.

    Returns:
        phi: Shape (n_phi,)
        r_eq: Shape (n_frames, n_phi)
        u_eq: Shape (n_frames, n_phi), where r = R * (1 + u)
    """
    rng = np.random.default_rng(config.seed)
    phi = np.linspace(0.0, 2.0 * np.pi, config.n_phi, endpoint=False)
    theta_eq = np.pi / 2.0

    modes = _sample_mode_amplitudes(
        rng=rng,
        l_max=config.l_max,
        n_frames=config.n_frames,
        kappa=config.kappa,
        sigma=config.sigma,
        temperature=config.temperature,
        single_mode=config.single_mode,
    )

    # Cache Y_l^m(theta=pi/2, phi) for speed and numerical consistency.
    ylm_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for l in range(2, config.l_max + 1):
        for m in range(-l, l + 1):
            ylm_cache[(l, m)] = sph_harm(m, l, phi, theta_eq)

    u_eq = np.zeros((config.n_frames, config.n_phi), dtype=np.float64)
    for l in range(2, config.l_max + 1):
        for m in range(-l, l + 1):
            u_eq += np.real(modes[(l, m)][:, None] * ylm_cache[(l, m)][None, :])

    r_eq = config.radius * (1.0 + u_eq)
    r_eq = np.maximum(r_eq, 1e-9)
    return phi, r_eq, u_eq
