from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .autocorr_wrapper import compute_angular_autocorrelation_series, extract_build_csv, time_average_xi
from .fitting import fit_kappa_sigma, theoretical_B_l
from .generator import GeneratorConfig, generate_equatorial_series
from .legendre import project_legendre_series, project_legendre_spectrum
from .validation import compute_validation_checks
from .visualization import (
    animate_contours,
    ensure_output_dir,
    plot_angular_autocorrelation,
    plot_fit_comparison,
    plot_legendre_spectrum,
)


def _parse_single_mode(single_mode: Optional[str]) -> Optional[Tuple[int, int]]:
    if single_mode is None:
        return None
    parts = single_mode.split(",")
    if len(parts) != 2:
        raise ValueError("single_mode must be formatted as 'l,m'")
    return int(parts[0]), int(parts[1])


def _optional_real_data_mean(build_dir: Path, l_values: np.ndarray) -> Optional[np.ndarray]:
    leg_path = build_dir / "legendre_amplitudes.txt"
    if not leg_path.exists():
        return None

    data = extract_build_csv(leg_path, has_header=True)
    if data.size == 0:
        return None

    means = np.mean(data, axis=0)
    lmin = int(l_values[0])
    lmax = int(l_values[-1])
    if means.size <= lmax:
        return None
    return means[lmin : lmax + 1]


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(Path(args.output_dir))

    config = GeneratorConfig(
        radius=args.radius,
        n_phi=args.n_phi,
        n_frames=args.n_frames,
        l_max=args.l_max,
        kappa=args.kappa,
        sigma=args.sigma,
        temperature=args.temperature,
        seed=args.seed,
        single_mode=_parse_single_mode(args.single_mode),
    )

    phi, r_eq, _ = generate_equatorial_series(config)

    gamma, xi_t, mean_r = compute_angular_autocorrelation_series(r_eq)
    xi_avg = time_average_xi(xi_t)

    # Stable fitting window for equatorial contour spectroscopy.
    l_min = max(3, args.fit_lmin)
    l_max = min(10, args.fit_lmax, config.l_max)
    if l_max < l_min:
        raise ValueError(f"Invalid fit mode window after clamping: l_min={l_min}, l_max={l_max}")

    l_values, b_avg = project_legendre_spectrum(xi_avg, gamma, l_min=l_min, l_max=l_max)
    _, b_t = project_legendre_series(xi_t, gamma, l_min=l_min, l_max=l_max)

    ddof = 1 if b_t.shape[0] > 1 else 0
    b_sem = np.std(b_t, axis=0, ddof=ddof) / np.sqrt(max(b_t.shape[0], 1))
    sem_floor = np.max([np.median(np.abs(b_avg)) * 1e-3, 1e-14])
    b_sem = np.where(~np.isfinite(b_sem) | (b_sem <= 0.0), sem_floor, b_sem)

    fit, _ = fit_kappa_sigma(
        l_values=l_values,
        b_values=b_avg,
        b_sem=b_sem,
        temperature=config.temperature,
        kappa0=args.kappa_guess,
        sigma0=args.sigma_guess,
    )

    b_fit = theoretical_B_l(l_values, kappa=fit.kappa, sigma=fit.sigma, temperature=config.temperature)
    validation = compute_validation_checks(
        base_config=config,
        gamma=gamma,
        xi_avg=xi_avg,
        l_values=l_values,
        b_values=b_avg,
    )

    animate_contours(phi, r_eq[: args.animation_frames], output_dir / "synthetic_contours.gif", fps=args.fps)
    plot_angular_autocorrelation(gamma, xi_avg, output_dir / "angular_autocorrelation.png")
    plot_legendre_spectrum(l_values, b_avg, output_dir / "legendre_spectrum.png")

    real_mean = _optional_real_data_mean(Path(args.real_build_dir), l_values) if args.compare_real else None
    plot_fit_comparison(
        l_values,
        b_avg,
        b_sem,
        b_fit,
        output_dir / "fit_comparison.png",
        real_b=real_mean,
    )

    np.save(output_dir / "phi.npy", phi)
    np.save(output_dir / "r_eq.npy", r_eq)
    np.save(output_dir / "gamma.npy", gamma)
    np.save(output_dir / "xi_t.npy", xi_t)
    np.save(output_dir / "xi_avg.npy", xi_avg)
    np.save(output_dir / "l_values.npy", l_values)
    np.save(output_dir / "b_avg.npy", b_avg)
    np.save(output_dir / "b_sem.npy", b_sem)
    np.save(output_dir / "mean_radius.npy", mean_r)
    (output_dir / "validation_checks.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    summary = [
        "=== Quasi-spherical fluctuation analysis complete ===",
        f"Output directory: {output_dir.resolve()}",
        f"Fit l-window = [{l_min}, {l_max}]",
        f"kappa_fit = {fit.kappa:.6e} +/- {fit.kappa_std:.6e}",
        f"sigma_fit = {fit.sigma:.6e} +/- {fit.sigma_std:.6e}",
        f"chi2/dof  = {fit.chi2:.6e} / {fit.dof}",
        f"optimizer_success = {fit.success} ({fit.message})",
        f"validation.kappa_reduces_fluctuations = {validation['kappa_reduces_fluctuations']}",
        f"validation.sigma_suppresses_low_l = {validation['sigma_suppresses_low_l']}",
        f"validation.xi_reconstruction_small_error = {validation['xi_reconstruction_small_error']}",
        f"validation.xi_reconstruction_rel_rmse = {validation['xi_reconstruction_rel_rmse']:.6e}",
    ]
    (output_dir / "fit_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    for line in summary:
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic quasi-spherical contour generation and fluctuation spectroscopy pipeline"
    )
    parser.add_argument("--radius", type=float, default=10.0, help="Average vesicle radius R")
    parser.add_argument("--n-phi", type=int, default=512, help="Number of equatorial angular samples")
    parser.add_argument("--n-frames", type=int, default=1000, help="Number of generated time frames")
    parser.add_argument("--l-max", type=int, default=30, help="Maximum spherical harmonic mode used for generation")
    parser.add_argument("--kappa", type=float, default=8.0e-20, help="Ground-truth bending rigidity used in synthetic generation")
    parser.add_argument("--sigma", type=float, default=0.5, help="Ground-truth surface tension used in synthetic generation")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducibility")

    parser.add_argument("--fit-lmin", type=int, default=3, help="Minimum l used in fit (clamped to >=3)")
    parser.add_argument("--fit-lmax", type=int, default=10, help="Maximum l used in fit (clamped to <=10)")
    parser.add_argument("--kappa-guess", type=float, default=1.0e-19, help="Initial guess for kappa")
    parser.add_argument("--sigma-guess", type=float, default=0.1, help="Initial guess for sigma")

    parser.add_argument(
        "--single-mode",
        type=str,
        default=None,
        help="Optional single mode isolation as 'l,m' (example: '4,2')",
    )
    parser.add_argument("--animation-frames", type=int, default=200, help="Number of initial frames to include in GIF")
    parser.add_argument("--fps", type=int, default=24, help="Animation frames per second")

    parser.add_argument("--output-dir", type=str, default="quasi_spherical/output", help="Output directory path")
    parser.add_argument(
        "--compare-real",
        action="store_true",
        help="Overlay mean real-data Legendre amplitudes from build/legendre_amplitudes.txt",
    )
    parser.add_argument(
        "--real-build-dir",
        type=str,
        default="build",
        help="Directory containing existing real build outputs",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
