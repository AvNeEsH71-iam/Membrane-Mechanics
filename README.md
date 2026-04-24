# Membrane Mechanics

A fully browser-based, interactive tool for analysing thermal shape fluctuations of Giant Unilamellar Vesicles (GUVs). Built on the theoretical framework of Brochard–Lennon / Helfrich, this suite lets you:

- **Visualise** angular auto-correlation functions for any combination of spherical-harmonic modes
- **Compute** Legendre spectrum coefficients *B_l*
- **Fit** bending rigidity *κ* and effective surface tension *Σ* via χ² minimisation
- **Understand** the mathematics and every formula is rendered with MathJax and explained inline

Model scope note:

- The spectroscopy pipeline is intentionally equatorial-contour based (`theta = pi/2`).
- This matches microscopy contour measurements and does not perform full spherical surface reconstruction.

- Website - https://avneesh71-iam.github.io/Membrane-Mechanics/index.html

---

## Repository Structure

```
Membrane-Mechanics/
├── index.html                  # Landing page / dashboard
├── pages/
│   ├── autocorrelation.html    # Angular auto-correlation visualiser
│   ├── legendre.html           # Legendre spectrum & mode amplitudes
│   ├── kappa.html              # κ / Σ extraction via χ² fitting
│   └── theory.html             # Full theory reference with equations
├── js/
│   ├── physics.js              # Core physics: αl, Bl, χ², κ extraction
│   ├── legendre.js             # Legendre polynomial evaluation
│   ├── plot.js                 # Shared charting helpers (Chart.js wrappers)
│   └── ui.js                   # Shared UI utilities
├── css/
│   └── style.css               # Global design system
└── README.md
```

---

## Physics Background

### Helfrich Free Energy
The membrane elastic energy is:

$$F_{Hel} = \int_S \frac{\kappa}{2}(2H - C_0)^2 \, dA + \sigma A$$

### Fluctuation Spectrum
Each spherical-harmonic mode *(l, m)* has mean-square amplitude:

$$\langle |u_{lm}|^2 \rangle = \frac{k_B T}{\kappa (l-1)(l+2)[l(l+1) + \Sigma]}$$

### Angular Auto-Correlation
The equatorial angular auto-correlation is:

$$\xi_{eq}(\gamma, t) = \frac{1}{2\pi r_{avg}^2} \int_0^{2\pi} \left[ r_{eq}(\phi+\gamma, t)\, r_{eq}(\phi, t) - r_{avg}^2 \right] d\phi$$

Expressed as a Legendre series:

$$\langle \xi^{th}(\gamma) \rangle = \sum_{l \geq 2} b_l P_l(\cos\gamma), \quad b_l = \frac{2l+1}{4\pi} \frac{k_B T}{\alpha_l}$$

### Parameter Extraction (χ² Minimisation)
Theoretical Legendre coefficients:

$$\bar{B}_l(j, s) = \frac{j}{p_l + s \, q_l}$$

where $j = k_B T / \kappa$, $s = \Sigma$, and:

$$p_l = \frac{4\pi(l-1)(l+2)(l+1)l}{2l+1}, \quad q_l = \frac{4\pi(l-1)(l+2)}{2l+1}$$

---

## Features by Page

| Page | What you can do |
|------|----------------|
| **Auto-Correlation** | Choose modes l=2…10, set κ and Σ, visualise ξ(γ) and P_l(cos γ) contributions |
| **Legendre Spectrum** | Plot B_l vs l; adjust κ, Σ, R; see how each mode amplitude decays |
| **κ / Σ Extraction** | Paste experimental B_l ± σ_l data, run χ² minimisation, get κ and uncertainties |
| **Theory** | Full derivation reference with live equation rendering |

---

## 🔗 References

1. Brochard & Lennon, *J. de Physique* **36**, 1035–1047 (1975)
2. Helfrich, *Z. Naturforsch. C* **28**, 693–703 (1973)
3. Kumar et al., *Chem. Phys. Lipids* **259**, 105374 (2024) — α-amylase membrane study
4. Dimova & Marques, *The Giant Vesicle Book*, CRC Press (2019)

---

## Made by - 

Avneesh Singh & Ridima Singh : SMBL, IISER Mohali  
Supervisor: Dr. Tripta Bhatia

https://bhattri.github.io/Biophysics/lab.html


