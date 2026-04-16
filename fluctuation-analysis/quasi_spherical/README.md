# Quasi-Spherical Synthetic Fluctuation Pipeline

This module generates synthetic equatorial contours from a quasi-spherical vesicle model and runs the same analysis stages used in the existing fluctuation pipeline.

Important model scope:

- This is intentionally an equatorial contour model only.
- The analysis uses `theta = pi/2`, matching microscopy contour experiments.
- Full spherical surface reconstruction is deliberately not performed.
- Spherical-harmonic fluctuations are projected onto the observed contour and then decomposed in Legendre modes.

Pipeline stages:

1. Generate synthetic `r_eq(phi, t)` from spherical harmonics.
2. Compute angular autocorrelation `xi(gamma, t)` with periodic shifts:

   `xi[j] = mean((r-r_avg) * (roll(r,-j)-r_avg)) / r_avg^2`
3. Time-average `xi`.
4. Project `<xi(gamma)>` onto Legendre polynomials to get `b_l` using integration over `d(cos gamma)`.
5. Fit `kappa` and `sigma` from weighted least squares (SEM weights):

   `B_l = j / (p_l + s q_l)`, where `j = k_B T / kappa` and `s = sigma`.

## Physics model

At the equator (`theta = pi/2`), the generated shape is:

`r(phi,t) = R [1 + u(theta=pi/2, phi, t)]`

with

`u(theta, phi, t) = sum_{l,m} u_lm(t) Y_l^m(theta, phi)`

and thermal mode variance:

`<|u_lm|^2> = (k_B T / kappa) / [(l+2)(l-1)(l(l+1)+sigma)]`

The fitting window is constrained to `3 <= l <= 10` for numerical stability and consistency with equatorial contour spectroscopy.

## Run (one command)

From the `fluctuation-analysis` directory:

```bash
python -m quasi_spherical.main
```

## Useful options

- `--l-max 30`
- `--kappa 8e-20 --sigma 0.5 --temperature 300`
- `--fit-lmin 3 --fit-lmax 10`
- `--single-mode 4,2` (bonus single-mode visualization)
- `--compare-real --real-build-dir build` (bonus synthetic vs real overlay)

## Outputs

By default, files are written to `quasi_spherical/output/`:

- `synthetic_contours.gif`
- `angular_autocorrelation.png`
- `legendre_spectrum.png`
- `fit_comparison.png`
- NumPy arrays: `phi.npy`, `r_eq.npy`, `gamma.npy`, `xi_t.npy`, `xi_avg.npy`, `l_values.npy`, `b_avg.npy`, `b_sem.npy`, `mean_radius.npy`
