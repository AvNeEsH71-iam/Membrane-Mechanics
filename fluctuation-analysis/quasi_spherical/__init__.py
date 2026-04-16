"""Quasi-spherical synthetic fluctuation analysis pipeline.

This package intentionally implements equatorial-contour fluctuation
spectroscopy (theta = pi/2) to mirror microscopy contour measurements.
It does not perform full spherical surface reconstruction.
"""

from .generator import GeneratorConfig, generate_equatorial_series
from .autocorr_wrapper import compute_angular_autocorrelation_series, time_average_xi
from .legendre import project_legendre_spectrum
from .fitting import fit_kappa_sigma
from .validation import compute_validation_checks

__all__ = [
    "GeneratorConfig",
    "generate_equatorial_series",
    "compute_angular_autocorrelation_series",
    "time_average_xi",
    "project_legendre_spectrum",
    "fit_kappa_sigma",
    "compute_validation_checks",
]
