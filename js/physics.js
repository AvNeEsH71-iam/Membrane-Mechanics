/**
 * physics.js
 * Core physics for GUV Vesicle Fluctuation Spectroscopy
 * Based on Helfrich / Brochard-Lennon theory
 * Reference: Kumar et al., Chem. Phys. Lipids 259, 105374 (2024)
 */

const kB = 1.380649e-23; // J/K
const T_DEFAULT = 298.15; // K (25°C)
const kBT_DEFAULT = kB * T_DEFAULT; // ~4.11e-21 J

// Global Chart.js defaults for consistent formal scientific styling.
if (typeof Chart !== 'undefined' && Chart.defaults) {
  Chart.defaults.font.family = 'Times New Roman';
  Chart.defaults.font.size = 14;
  Chart.defaults.color = '#e5e7eb';
  Chart.defaults.plugins.legend.labels.color = '#e5e7eb';
  Chart.defaults.plugins.title.color = '#e5e7eb';
  if (Chart.defaults.scales) {
    Chart.defaults.scales.linear = Chart.defaults.scales.linear || {};
    Chart.defaults.scales.logarithmic = Chart.defaults.scales.logarithmic || {};
    Chart.defaults.scales.category = Chart.defaults.scales.category || {};
    Chart.defaults.scales.linear.grid = { color: 'rgba(148,163,184,0.14)' };
    Chart.defaults.scales.logarithmic.grid = { color: 'rgba(148,163,184,0.14)' };
    Chart.defaults.scales.category.grid = { color: 'rgba(148,163,184,0.14)' };
    Chart.defaults.scales.linear.ticks = { color: '#e5e7eb' };
    Chart.defaults.scales.logarithmic.ticks = { color: '#e5e7eb' };
    Chart.defaults.scales.category.ticks = { color: '#e5e7eb' };
  }
}

// ─── Legendre Polynomials ──────────────────────────────────────────────────

/**
 * Evaluate P_l(x) using Bonnet's recurrence relation.
 * @param {number} l - degree
 * @param {number} x - argument (cosγ)
 * @returns {number}
 */
function legendreP(l, x) {
  if (l === 0) return 1;
  if (l === 1) return x;
  let p_prev = 1, p_curr = x;
  for (let n = 2; n <= l; n++) {
    const p_next = ((2 * n - 1) * x * p_curr - (n - 1) * p_prev) / n;
    p_prev = p_curr;
    p_curr = p_next;
  }
  return p_curr;
}

/**
 * Evaluate P_l(cos γ) for array of γ values.
 * @param {number} l
 * @param {number[]} gammas - array of angles in radians
 * @returns {number[]}
 */
function legendrePArray(l, gammas) {
  return gammas.map(g => legendreP(l, Math.cos(g)));
}

// ─── Mode Stiffness ────────────────────────────────────────────────────────

/**
 * Mode stiffness α_l (in units of κ)
 * α_l = κ * (l-1)(l+2)[l(l+1) + Σ]
 * @param {number} l  - mode number (l >= 2)
 * @param {number} kappa - bending rigidity in kBT units
 * @param {number} Sigma - dimensionless effective surface tension
 * @returns {number} α_l in kBT units
 */
function modeStiffness(l, kappa, Sigma) {
  return kappa * (l - 1) * (l + 2) * (l * (l + 1) + Sigma);
}

/**
 * Mean-square mode amplitude ⟨|u_lm|²⟩ = kBT / α_l
 * @param {number} l
 * @param {number} kappa - in kBT
 * @param {number} Sigma
 * @returns {number}
 */
function modeMeanSquare(l, kappa, Sigma) {
  const alpha = modeStiffness(l, kappa, Sigma);
  if (alpha <= 0) return 0;
  return 1.0 / alpha; // in units of kBT/kBT = dimensionless, since kappa is in kBT
}

// ─── Legendre Coefficients b_l ────────────────────────────────────────────

/**
 * Theoretical Legendre coefficient b_l
 * b_l = (2l+1)/(4π) * ⟨|u_lm|²⟩
 * @param {number} l
 * @param {number} kappa - in kBT units
 * @param {number} Sigma
 * @returns {number}
 */
function bl_theory(l, kappa, Sigma) {
  return ((2 * l + 1) / (4 * Math.PI)) * modeMeanSquare(l, kappa, Sigma);
}

// ─── p_l and q_l factors ──────────────────────────────────────────────────

/**
 * p_l = 4π(l-1)(l+2)(l+1)l / (2l+1)
 */
function pl_factor(l) {
  return (4 * Math.PI * (l - 1) * (l + 2) * (l + 1) * l) / (2 * l + 1);
}

/**
 * q_l = 4π(l-1)(l+2) / (2l+1)
 */
function ql_factor(l) {
  return (4 * Math.PI * (l - 1) * (l + 2)) / (2 * l + 1);
}

/**
 * Theoretical mean B̄_l(j, s) = j / (p_l + s * q_l)
 * where j = kBT/κ, s = Σ
 */
function Bl_bar(l, j, s) {
  const pl = pl_factor(l);
  const ql = ql_factor(l);
  return j / (pl + s * ql);
}

// ─── Angular Auto-Correlation ─────────────────────────────────────────────

/**
 * Compute theoretical angular auto-correlation ξ(γ) as Legendre expansion
 * ξ(γ) = Σ_{l≥2} b_l * P_l(cos γ)
 *
 * @param {number[]} gammas - array of angles in radians [0, 2π]
 * @param {number} kappa - bending rigidity in kBT
 * @param {number} Sigma - effective surface tension
 * @param {number} lMin - minimum mode (default 2)
 * @param {number} lMax - maximum mode (default 10)
 * @returns {number[]} xi values at each γ
 */
function autocorrelation(gammas, kappa, Sigma, lMin = 2, lMax = 10) {
  const xi = new Array(gammas.length).fill(0);
  for (let l = lMin; l <= lMax; l++) {
    const bl = bl_theory(l, kappa, Sigma);
    for (let i = 0; i < gammas.length; i++) {
      xi[i] += bl * legendreP(l, Math.cos(gammas[i]));
    }
  }
  return xi;
}

/**
 * Contribution of a single mode l to ξ(γ)
 * @param {number[]} gammas
 * @param {number} l
 * @param {number} kappa
 * @param {number} Sigma
 * @returns {number[]}
 */
function autocorrSingleMode(gammas, l, kappa, Sigma) {
  const bl = bl_theory(l, kappa, Sigma);
  return gammas.map(g => bl * legendreP(l, Math.cos(g)));
}

// ─── χ² Fitting ───────────────────────────────────────────────────────────

/**
 * Compute j(s) from experimental Bl and sigma_l values
 * j(s) = Σ_l [p_l * B_l / (p_l σ_l)² * (1 + s*q_l/p_l)]
 *         / Σ_l [1/(p_l σ_l)² * (1 + s*q_l/p_l)²]
 *
 * @param {number} s - trial Σ value
 * @param {number[]} Bl_exp - experimental mean Legendre coefficients
 * @param {number[]} sigma_l - standard errors of Bl
 * @param {number[]} modes - array of mode numbers
 * @returns {number} j(s) = kBT/κ
 */
function j_of_s(s, Bl_exp, sigma_l, modes) {
  let num = 0, den = 0;
  for (let i = 0; i < modes.length; i++) {
    const l = modes[i];
    const pl = pl_factor(l);
    const ql = ql_factor(l);
    const sl = sigma_l[i];
    const Bl = Bl_exp[i];
    const factor = 1 + s * ql / pl;
    const weight = 1.0 / (pl * sl) ** 2;
    num += weight * pl * Bl * factor;
    den += weight * factor * factor;
  }
  return den > 0 ? num / den : 0;
}

/**
 * Compute χ²(j(s), s)
 */
function chi2(s, Bl_exp, sigma_l, modes) {
  const j = j_of_s(s, Bl_exp, sigma_l, modes);
  let c2 = 0;
  for (let i = 0; i < modes.length; i++) {
    const l = modes[i];
    const pl = pl_factor(l);
    const ql = ql_factor(l);
    const sl = sigma_l[i];
    const Bl = Bl_exp[i];
    const Bl_th = j / (pl + s * ql);  // = j/(pl*(1 + s*ql/pl))
    const residual = (pl * Bl - pl * Bl_th) / (pl * sl);
    c2 += residual * residual;
  }
  return c2;
}

/**
 * Golden-section search for minimum χ²(s) in [s_min, s_max]
 */
function minimizeChiSquared(Bl_exp, sigma_l, modes, s_min = -50, s_max = 200, tol = 1e-6) {
  const phi = (Math.sqrt(5) - 1) / 2;
  let a = s_min, b = s_max;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  while (Math.abs(b - a) > tol) {
    if (chi2(c, Bl_exp, sigma_l, modes) < chi2(d, Bl_exp, sigma_l, modes)) {
      b = d;
    } else {
      a = c;
    }
    c = b - phi * (b - a);
    d = a + phi * (b - a);
  }
  const s_opt = (a + b) / 2;
  const j_opt = j_of_s(s_opt, Bl_exp, sigma_l, modes);
  return { s_opt, j_opt };
}

/**
 * Compute covariance matrix elements (for error estimation)
 */
function covarianceMatrix(j, s, Bl_exp, sigma_l, modes) {
  let d2c_dj2 = 0, d2c_djds = 0, d2c_ds2 = 0;
  for (let i = 0; i < modes.length; i++) {
    const l = modes[i];
    const pl = pl_factor(l);
    const ql = ql_factor(l);
    const sl = sigma_l[i];
    const Bl = Bl_exp[i];
    const ratio = ql / pl;
    const factor = 1 + s * ratio;
    const weight = 1.0 / (pl * sl) ** 2;
    const residual = pl * Bl - j / factor;

    d2c_dj2 += 2 * weight / (factor * factor);
    d2c_djds += 2 * ratio * weight * (residual - j / factor) / (factor * factor);
    d2c_ds2 += 2 * ratio * ratio * weight * (3 * j / factor - 2 * pl * Bl) / (factor * factor * factor);
  }
  // Invert 2x2 matrix
  const det = d2c_dj2 * d2c_ds2 - d2c_djds * d2c_djds;
  if (Math.abs(det) < 1e-30) return { H11: 0, H22: 0 };
  return {
    H11: d2c_ds2 / det,
    H22: d2c_dj2 / det,
  };
}

/**
 * Full parameter extraction pipeline
 * @param {number[]} Bl_exp
 * @param {number[]} sigma_l
 * @param {number[]} modes
 * @returns {{ kappa, sigma_kappa, Sigma, sigma_Sigma, j, chi2_min }}
 */
function extractParameters(Bl_exp, sigma_l, modes) {
  const { s_opt, j_opt } = minimizeChiSquared(Bl_exp, sigma_l, modes);
  const chi2_min = chi2(s_opt, Bl_exp, sigma_l, modes);
  const { H11, H22 } = covarianceMatrix(j_opt, s_opt, Bl_exp, sigma_l, modes);

  // κ = kBT / j  (in kBT units → just 1/j)
  const kappa = 1.0 / j_opt; // in kBT
  const sigma_kappa = Math.sqrt(H11) * kappa * kappa * j_opt;

  return {
    kappa,          // bending rigidity in kBT units
    sigma_kappa,
    Sigma: s_opt,   // effective surface tension (dimensionless)
    sigma_Sigma: Math.sqrt(H22),
    j: j_opt,
    chi2_min,
  };
}

/**
 * Generate γ array from 0 to 2π
 */
function gammaArray(nPoints = 360) {
  return Array.from({ length: nPoints }, (_, i) => (i / (nPoints - 1)) * 2 * Math.PI);
}

/**
 * Compute χ² curve over range of s values
 */
function chi2Curve(Bl_exp, sigma_l, modes, s_min = -20, s_max = 150, nPoints = 200) {
  const sVals = Array.from({ length: nPoints }, (_, i) => s_min + (i / (nPoints - 1)) * (s_max - s_min));
  const c2Vals = sVals.map(s => chi2(s, Bl_exp, sigma_l, modes));
  return { sVals, c2Vals };
}

// Export for use in other modules
if (typeof module !== 'undefined') {
  module.exports = {
    legendreP, legendrePArray, modeStiffness, modeMeanSquare,
    bl_theory, pl_factor, ql_factor, Bl_bar,
    autocorrelation, autocorrSingleMode,
    j_of_s, chi2, minimizeChiSquared, covarianceMatrix,
    extractParameters, gammaArray, chi2Curve,
    kBT_DEFAULT
  };
}


