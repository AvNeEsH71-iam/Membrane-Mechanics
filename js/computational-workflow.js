/*
 * computational-workflow.js
 * Interactive end-to-end tutorial pipeline for quasi-spherical vesicle analysis.
 */

(function () {
  const TWO_PI = 2 * Math.PI;

  const state = {
    contourChart: null,
    xiChart: null,
    blChart: null,
    chiChart: null,
    fitChart: null,
    latest: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function clampInt(x, lo, hi) {
    return Math.max(lo, Math.min(hi, Math.round(x)));
  }

  function clampFloat(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function readInputs() {
    const lMax = clampInt(parseInt(byId("wfLMax").value, 10), 4, 20);
    let lMinFit = clampInt(parseInt(byId("wfFitLMin").value, 10), 2, 18);
    let lMaxFit = clampInt(parseInt(byId("wfFitLMax").value, 10), lMinFit + 1, 20);

    lMinFit = Math.min(lMinFit, lMax - 1);
    lMaxFit = Math.min(Math.max(lMaxFit, lMinFit + 1), lMax);

    byId("wfLMax").value = String(lMax);
    byId("wfFitLMin").value = String(lMinFit);
    byId("wfFitLMax").value = String(lMaxFit);

    return {
      kappaTrue: clampFloat(parseFloat(byId("wfKappaTrue").value), 2, 200),
      sigmaTrue: clampFloat(parseFloat(byId("wfSigmaTrue").value), -10, 200),
      radius: clampFloat(parseFloat(byId("wfRadius").value), 2, 50),
      lMax,
      nPhi: clampInt(parseInt(byId("wfNPhi").value, 10), 128, 512),
      nFrames: clampInt(parseInt(byId("wfNFrames").value, 10), 30, 400),
      seed: clampInt(parseInt(byId("wfSeed").value, 10), 1, 99999999),
      fitLMin: lMinFit,
      fitLMax: lMaxFit,
    };
  }

  // Deterministic RNG for reproducibility.
  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6D2B79F5;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeNormalSampler(randUniform) {
    let spare = null;
    return function randn() {
      if (spare !== null) {
        const out = spare;
        spare = null;
        return out;
      }
      let u = 0;
      let v = 0;
      while (u <= Number.EPSILON) u = randUniform();
      while (v <= Number.EPSILON) v = randUniform();
      const r = Math.sqrt(-2 * Math.log(u));
      const theta = TWO_PI * v;
      spare = r * Math.sin(theta);
      return r * Math.cos(theta);
    };
  }

  function factorial(n) {
    let out = 1;
    for (let i = 2; i <= n; i += 1) out *= i;
    return out;
  }

  function associatedLegendre(l, m, x) {
    if (m < 0 || m > l) return 0;
    let pmm = 1.0;
    if (m > 0) {
      const somx2 = Math.sqrt(Math.max(0, 1 - x * x));
      let fact = 1.0;
      for (let i = 1; i <= m; i += 1) {
        pmm *= -fact * somx2;
        fact += 2.0;
      }
    }
    if (l === m) return pmm;

    let pmmp1 = x * (2 * m + 1) * pmm;
    if (l === m + 1) return pmmp1;

    let pll = 0.0;
    for (let ll = m + 2; ll <= l; ll += 1) {
      pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
      pmm = pmmp1;
      pmmp1 = pll;
    }
    return pmmp1;
  }

  function sphericalHarmonicAtEquator(l, m, phi) {
    const x = 0.0; // cos(pi/2)
    if (m >= 0) {
      const p = associatedLegendre(l, m, x);
      const norm = Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (factorial(l - m) / factorial(l + m)));
      const angle = m * phi;
      return {
        re: norm * p * Math.cos(angle),
        im: norm * p * Math.sin(angle),
      };
    }

    const mp = -m;
    const y = sphericalHarmonicAtEquator(l, mp, phi);
    const sign = (mp % 2 === 0) ? 1 : -1;
    return {
      re: sign * y.re,
      im: -sign * y.im,
    };
  }

  function precomputeYlm(lMax, phiArr) {
    const ylmRe = new Map();
    const ylmIm = new Map();

    for (let l = 2; l <= lMax; l += 1) {
      for (let m = 0; m <= l; m += 1) {
        const re = new Float64Array(phiArr.length);
        const im = new Float64Array(phiArr.length);
        for (let i = 0; i < phiArr.length; i += 1) {
          const y = sphericalHarmonicAtEquator(l, m, phiArr[i]);
          re[i] = y.re;
          im[i] = y.im;
        }
        const key = `${l},${m}`;
        ylmRe.set(key, re);
        ylmIm.set(key, im);
      }
    }

    return { ylmRe, ylmIm };
  }

  function sampleModeAmplitudes(params, randn) {
    const ampRe = new Map();
    const ampIm = new Map();

    for (let l = 2; l <= params.lMax; l += 1) {
      const denom = (l - 1) * (l + 2) * (l * (l + 1) + params.sigmaTrue);
      const variance = 1.0 / Math.max(params.kappaTrue * denom, 1e-12);

      for (let m = 0; m <= l; m += 1) {
        const key = `${l},${m}`;
        const re = new Float64Array(params.nFrames);
        const im = new Float64Array(params.nFrames);

        if (m === 0) {
          const scale = Math.sqrt(Math.max(variance, 0));
          for (let t = 0; t < params.nFrames; t += 1) {
            re[t] = randn() * scale;
          }
        } else {
          const scale = Math.sqrt(Math.max(variance / 2, 0));
          for (let t = 0; t < params.nFrames; t += 1) {
            re[t] = randn() * scale;
            im[t] = randn() * scale;
          }
        }

        ampRe.set(key, re);
        ampIm.set(key, im);
      }
    }

    return { ampRe, ampIm };
  }

  function generateEquatorialSeries(params) {
    const phi = new Float64Array(params.nPhi);
    for (let i = 0; i < params.nPhi; i += 1) {
      phi[i] = (TWO_PI * i) / params.nPhi;
    }

    const rng = mulberry32(params.seed);
    const randn = makeNormalSampler(rng);

    const { ylmRe, ylmIm } = precomputeYlm(params.lMax, phi);
    const { ampRe, ampIm } = sampleModeAmplitudes(params, randn);

    const uEq = Array.from({ length: params.nFrames }, () => new Float64Array(params.nPhi));

    for (let l = 2; l <= params.lMax; l += 1) {
      for (let m = 0; m <= l; m += 1) {
        const key = `${l},${m}`;
        const yRe = ylmRe.get(key);
        const yIm = ylmIm.get(key);
        const aRe = ampRe.get(key);
        const aIm = ampIm.get(key);

        for (let t = 0; t < params.nFrames; t += 1) {
          const ar = aRe[t];
          const ai = aIm[t];
          const ut = uEq[t];

          if (m === 0) {
            for (let i = 0; i < params.nPhi; i += 1) {
              ut[i] += ar * yRe[i];
            }
          } else {
            for (let i = 0; i < params.nPhi; i += 1) {
              const reProd = ar * yRe[i] - ai * yIm[i];
              ut[i] += 2.0 * reProd;
            }
          }
        }
      }
    }

    const rEq = Array.from({ length: params.nFrames }, () => new Float64Array(params.nPhi));
    for (let t = 0; t < params.nFrames; t += 1) {
      for (let i = 0; i < params.nPhi; i += 1) {
        rEq[t][i] = Math.max(params.radius * (1 + uEq[t][i]), 1e-9);
      }
    }

    return { phi, rEq };
  }

  function simpsonWeightsLikeCpp(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i += 1) w[i] = (i % 2 === 0) ? 2 : 4;
    return w;
  }

  function meanRadiusLikeCpp(r, w) {
    let sum = 0;
    for (let i = 0; i < r.length; i += 1) sum += w[i] * r[i];
    return sum / (3 * r.length);
  }

  function angularAutocorrelationLikeCpp(r, w) {
    const n = r.length;
    const rAvg = meanRadiusLikeCpp(r, w);
    const centered = new Float64Array(n);
    for (let i = 0; i < n; i += 1) centered[i] = r[i] - rAvg;

    const out = new Float64Array(n);
    const denom = 3 * n * rAvg * rAvg;
    for (let j = 0; j < n; j += 1) {
      let acc = 0;
      for (let i = 0; i < n; i += 1) {
        const shifted = r[(i + j) % n] - rAvg;
        acc += w[i] * shifted * centered[i];
      }
      out[j] = acc / denom;
    }
    return { xi: out, rAvg };
  }

  function computeAutocorrelationSeries(rEq) {
    const nFrames = rEq.length;
    const nPhi = rEq[0].length;
    const w = simpsonWeightsLikeCpp(nPhi);
    const gamma = new Float64Array(nPhi);
    for (let i = 0; i < nPhi; i += 1) gamma[i] = (TWO_PI * i) / nPhi;

    const xiT = Array.from({ length: nFrames }, () => new Float64Array(nPhi));
    const meanR = new Float64Array(nFrames);

    for (let t = 0; t < nFrames; t += 1) {
      const { xi, rAvg } = angularAutocorrelationLikeCpp(rEq[t], w);
      xiT[t] = xi;
      meanR[t] = rAvg;
    }

    const xiAvg = new Float64Array(nPhi);
    for (let i = 0; i < nPhi; i += 1) {
      let s = 0;
      for (let t = 0; t < nFrames; t += 1) s += xiT[t][i];
      xiAvg[i] = s / nFrames;
    }

    return { gamma, xiT, xiAvg, meanR };
  }

  function trapz(y, x) {
    let s = 0;
    for (let i = 0; i < y.length - 1; i += 1) {
      s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    return s;
  }

  function projectLegendreSpectrum(xiGamma, gamma, lMin, lMax) {
    const gHalf = [];
    const xiHalf = [];
    for (let i = 0; i < gamma.length; i += 1) {
      if (gamma[i] <= Math.PI + 1e-12) {
        gHalf.push(gamma[i]);
        xiHalf.push(xiGamma[i]);
      }
    }

    const lValues = [];
    const bValues = [];

    for (let l = lMin; l <= lMax; l += 1) {
      const integrand = new Float64Array(gHalf.length);
      for (let i = 0; i < gHalf.length; i += 1) {
        const c = Math.cos(gHalf[i]);
        integrand[i] = xiHalf[i] * legendreP(l, c) * Math.sin(gHalf[i]);
      }
      const bl = 0.5 * (2 * l + 1) * trapz(integrand, gHalf);
      lValues.push(l);
      bValues.push(bl);
    }

    return { lValues, bValues };
  }

  function projectLegendreSeries(xiT, gamma, lMin, lMax) {
    const bT = [];
    for (let t = 0; t < xiT.length; t += 1) {
      const { bValues } = projectLegendreSpectrum(xiT[t], gamma, lMin, lMax);
      bT.push(bValues);
    }
    return bT;
  }

  function computeSem(bT) {
    const nFrames = bT.length;
    const nModes = bT[0].length;
    const sem = new Array(nModes).fill(0);

    for (let m = 0; m < nModes; m += 1) {
      let mean = 0;
      for (let t = 0; t < nFrames; t += 1) mean += bT[t][m];
      mean /= nFrames;

      let v = 0;
      for (let t = 0; t < nFrames; t += 1) {
        const d = bT[t][m] - mean;
        v += d * d;
      }
      const sd = Math.sqrt(v / Math.max(nFrames - 1, 1));
      sem[m] = Math.max(sd / Math.sqrt(Math.max(nFrames, 1)), 1e-12);
    }

    return sem;
  }

  function fmtExp(v, digits = 3) {
    if (!Number.isFinite(v)) return "nan";
    if (Math.abs(v) < 1e-300) return "0";
    return v.toExponential(digits);
  }

  function fmtFix(v, digits = 4) {
    if (!Number.isFinite(v)) return "nan";
    return v.toFixed(digits);
  }

  function buildCalculationLogs(data) {
    const { params, rEq, gamma, xiAvg, lValues, bValues, bSem, fit } = data;

    const step1Rows = [];
    for (let l = 2; l <= Math.min(params.lMax, 7); l += 1) {
      const dL = (l - 1) * (l + 2) * (l * (l + 1) + params.sigmaTrue);
      const vL = 1.0 / Math.max(params.kappaTrue * dL, 1e-12);
      step1Rows.push({ l, dL, vL, sL: Math.sqrt(vL) });
    }
    const lRef1 = 2;
    const dRef1 = (lRef1 - 1) * (lRef1 + 2) * (lRef1 * (lRef1 + 1) + params.sigmaTrue);
    const vRef1 = 1.0 / Math.max(params.kappaTrue * dRef1, 1e-12);
        const step1SummaryTex = `
    \\[
      D_l=(l-1)(l+2)\\left[l(l+1)+\\Sigma\\right],\\qquad
      \\mathrm{var}_l=\\frac{1}{\\kappa D_l}
    \\]`;
    const step1Steps = [
      `Use the selected parameters \\(\\kappa=${fmtFix(params.kappaTrue, 4)}\\) and \\(\\Sigma=${fmtFix(params.sigmaTrue, 4)}\\).`,
      `For reference mode \\(l=${lRef1}\\):
    \\[
    D_${lRef1}=(${lRef1}-1)(${lRef1}+2)\\left[${lRef1}(${lRef1}+1)+${fmtFix(params.sigmaTrue, 4)}\\right]=${fmtFix(dRef1, 4)}
    \\]`,
      `Then
    \\[
    \\mathrm{var}_{${lRef1}}=\\frac{1}{${fmtFix(params.kappaTrue, 4)}\\times ${fmtFix(dRef1, 4)}}=${fmtExp(vRef1, 4)}
    \\]`,
      `Hence \\(\\sqrt{\\mathrm{var}_{${lRef1}}}=${fmtExp(Math.sqrt(vRef1), 4)}\\), used for Gaussian mode sampling.`
    ];

    const frame0 = rEq[0];
    const w = simpsonWeightsLikeCpp(frame0.length);
    const gIdx = Math.floor(frame0.length / 8);
    const gVal = gamma[gIdx];
    const rAvg0 = meanRadiusLikeCpp(frame0, w);
    const step2Rows = [];
    let num2 = 0;
    for (let i = 0; i < frame0.length; i += 1) {
      const c = frame0[i] - rAvg0;
      const s = frame0[(i + gIdx) % frame0.length] - rAvg0;
      const term = w[i] * c * s;
      num2 += term;
      if (i < 8) {
        step2Rows.push({ i, wi: w[i], c, s, term });
      }
    }
    const den2 = 3 * frame0.length * rAvg0 * rAvg0;
    const xi2 = num2 / den2;
        const step2SummaryTex = `
    \\[
      \\xi(\\gamma_j)=\\frac{\\sum_i w_i\\,[r_{i+j}-r_{\\mathrm{avg}}][r_i-r_{\\mathrm{avg}}]}{3N\\,r_{\\mathrm{avg}}^2}
    \\]`;
    const step2Steps = [
      `Compute the weighted mean radius for the chosen frame:
    \\[
    r_{\\mathrm{avg}}=\\frac{\\sum_i w_i r_i}{3N}=${fmtFix(rAvg0, 6)}
    \\]`,
      `Pick shift index \\(j=${gIdx}\\), so \\(\\gamma_j=${fmtFix((gVal * 180) / Math.PI, 2)}^\\circ\\).`,
      `Evaluate numerator and denominator separately:
    \\[
    \\text{Numerator}=${fmtExp(num2, 4)},\\qquad
    \\text{Denominator}=3N\\,r_{\\mathrm{avg}}^2=${fmtExp(den2, 4)}
    \\]`,
      `Final value:
    \\[
    \\xi(\\gamma_j)=\\frac{${fmtExp(num2, 4)}}{${fmtExp(den2, 4)}}=${fmtExp(xi2, 4)}
    \\]`
    ];

    const lRef3 = lValues[0];
    const gHalf = [];
    const xiHalf = [];
    for (let i = 0; i < gamma.length; i += 1) {
      if (gamma[i] <= Math.PI + 1e-12) {
        gHalf.push(gamma[i]);
        xiHalf.push(xiAvg[i]);
      }
    }
    const integ3 = new Float64Array(gHalf.length);
    for (let i = 0; i < gHalf.length; i += 1) {
      integ3[i] = xiHalf[i] * legendreP(lRef3, Math.cos(gHalf[i])) * Math.sin(gHalf[i]);
    }
    const intVal3 = trapz(integ3, gHalf);
    const bRef3 = 0.5 * (2 * lRef3 + 1) * intVal3;
    const step3Rows = [];
    const stride3 = Math.max(1, Math.floor(gHalf.length / 8));
    for (let i = 0; i < gHalf.length && step3Rows.length < 8; i += stride3) {
      step3Rows.push({
        gDeg: gHalf[i] * 180 / Math.PI,
        xi: xiHalf[i],
        pl: legendreP(lRef3, Math.cos(gHalf[i])),
        sg: Math.sin(gHalf[i]),
        in: integ3[i],
      });
    }
        const step3SummaryTex = `
    \\[
      b_l=\\frac{2l+1}{2}\\int_0^{\\pi}\\xi(\\gamma)P_l(\\cos\\gamma)\\sin\\gamma\\,d\\gamma
    \\]`;
    const step3Steps = [
      `Select reference mode \\(l=${lRef3}\\) for explicit substitution.`,
      `Compute the projection integral:
    \\[
    I_l=\\int_0^{\\pi}\\xi(\\gamma)P_l(\\cos\\gamma)\\sin\\gamma\\,d\\gamma=${fmtExp(intVal3, 4)}
    \\]`,
      `Apply prefactor:
    \\[
    b_l=\\frac{2(${lRef3})+1}{2}\\,I_l=${fmtFix((2 * lRef3 + 1) / 2, 4)}\\times ${fmtExp(intVal3, 4)}=${fmtExp(bRef3, 4)}
    \\]`,
      `The sample table lists representative \\(\\gamma\\)-points used inside the numerical integration.`
    ];

    const step4Rows = [];
    let chiSum = 0;
    for (let i = 0; i < lValues.length; i += 1) {
      const bTh = Bl_bar(lValues[i], fit.j, fit.Sigma);
      const res = (bValues[i] - bTh) / bSem[i];
      const c2 = res * res;
      chiSum += c2;
      step4Rows.push({
        l: lValues[i],
        bExp: bValues[i],
        bSem: bSem[i],
        bTh,
        res,
        c2,
      });
    }
        const step4SummaryTex = `
    \\[
      \\bar{B}_l=\\frac{j}{p_l+s q_l},\\qquad
      \\chi^2=\\sum_l\\left(\\frac{B_l^{\\mathrm{exp}}-\\bar{B}_l}{\\sigma_l}\\right)^2
    \\]`;
    const step4Steps = [
      `Use fitted values \\(j=${fmtExp(fit.j, 4)}\\) and \\(s=\\Sigma=${fmtFix(fit.Sigma, 5)}\\) to evaluate each \\(\\bar{B}_l\\).`,
      `For every mode in the fit window, compute residual \((B_l^{exp}-\bar{B}_l)/\sigma_l\) and square it.`,
      `Sum all squared residuals:
    \\[
    \\chi^2=${fmtFix(chiSum, 6)}
    \\]`,
      `Recover membrane rigidity:
    \\[
    \\kappa_{\\mathrm{fit}}=\\frac{1}{j}=${fmtFix(fit.kappa, 5)}
    \\]`
    ];

    return {
      step1: { intro: "Mode stiffness and sampling variance", summaryTex: step1SummaryTex, steps: step1Steps, rows: step1Rows },
      step2: { intro: "Autocorrelation for one angular shift", summaryTex: step2SummaryTex, steps: step2Steps, rows: step2Rows },
      step3: { intro: "Legendre projection from correlation to spectral coefficients", summaryTex: step3SummaryTex, steps: step3Steps, rows: step3Rows },
      step4: { intro: "Chi-square objective and fitted parameters", summaryTex: step4SummaryTex, steps: step4Steps, rows: step4Rows },
    };
  }

  function renderEquationBlock(containerId, log) {
    const el = byId(containerId);
    if (!el || !log) return;
    el.innerHTML = [
      `<p>${log.intro}</p>`,
      log.summaryTex,
    ].join("");
  }

  function renderStepList(listId, steps) {
    const list = byId(listId);
    if (!list) return;
    list.innerHTML = "";
    steps.forEach((stepText) => {
      const li = document.createElement("li");
      li.innerHTML = stepText;
      list.appendChild(li);
    });
  }

  function renderCalculationRows(tbodyId, rows, mapper) {
    const tbody = byId(tbodyId);
    if (!tbody) return;
    tbody.innerHTML = "";
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const vals = mapper(row);
      vals.forEach((v) => {
        const td = document.createElement("td");
        td.textContent = v;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  function renderCalculationLogs() {
    const latest = state.latest;
    if (!latest || !latest.logs) return;

    renderEquationBlock("wfStep1Equation", latest.logs.step1);
    renderStepList("wfStep1StepList", latest.logs.step1.steps);
    renderCalculationRows("wfStep1TableBody", latest.logs.step1.rows, (r) => [
      String(r.l),
      fmtFix(r.dL, 4),
      fmtExp(r.vL, 4),
      fmtExp(r.sL, 4),
    ]);

    renderEquationBlock("wfStep2Equation", latest.logs.step2);
    renderStepList("wfStep2StepList", latest.logs.step2.steps);
    renderCalculationRows("wfStep2TableBody", latest.logs.step2.rows, (r) => [
      String(r.i),
      fmtFix(r.wi, 0),
      fmtExp(r.c, 4),
      fmtExp(r.s, 4),
      fmtExp(r.term, 4),
    ]);

    renderEquationBlock("wfStep3Equation", latest.logs.step3);
    renderStepList("wfStep3StepList", latest.logs.step3.steps);
    renderCalculationRows("wfStep3TableBody", latest.logs.step3.rows, (r) => [
      fmtFix(r.gDeg, 2),
      fmtExp(r.xi, 4),
      fmtExp(r.pl, 4),
      fmtExp(r.sg, 4),
      fmtExp(r.in, 4),
    ]);

    renderEquationBlock("wfStep4Equation", latest.logs.step4);
    renderStepList("wfStep4StepList", latest.logs.step4.steps);
    renderCalculationRows("wfStep4TableBody", latest.logs.step4.rows, (r) => [
      String(r.l),
      fmtExp(r.bExp, 4),
      fmtExp(r.bSem, 4),
      fmtExp(r.bTh, 4),
      fmtExp(r.res, 4),
      fmtExp(r.c2, 4),
    ]);
  }

  function wireStepToggles() {
    const toggles = document.querySelectorAll(".calc-toggle");
    toggles.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-target");
        const panel = byId(targetId);
        if (!panel) return;

        const willOpen = panel.hasAttribute("hidden");
        if (willOpen) {
          panel.removeAttribute("hidden");
          btn.textContent = "Hide Step-by-Step";
          btn.setAttribute("aria-expanded", "true");
        } else {
          panel.setAttribute("hidden", "hidden");
          btn.textContent = "Show Step-by-Step";
          btn.setAttribute("aria-expanded", "false");
        }

        if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
          window.MathJax.typesetPromise([panel]);
        }
      });
    });
  }

  function runWorkflow() {
    const params = readInputs();

    const { phi, rEq } = generateEquatorialSeries(params);
    const { gamma, xiT, xiAvg, meanR } = computeAutocorrelationSeries(rEq);

    const { lValues, bValues } = projectLegendreSpectrum(xiAvg, gamma, params.fitLMin, params.fitLMax);
    const bT = projectLegendreSeries(xiT, gamma, params.fitLMin, params.fitLMax);
    const bSem = computeSem(bT);

    const fit = extractParameters(bValues, bSem, lValues);
    const chi = chi2Curve(bValues, bSem, lValues, -20, 180, 250);

    const bTheory = lValues.map((l) => Bl_bar(l, fit.j, fit.Sigma));

    state.latest = {
      params,
      phi,
      rEq,
      gamma,
      xiAvg,
      lValues,
      bValues,
      bSem,
      fit,
      chi,
      bTheory,
      meanR,
    };

    state.latest.logs = buildCalculationLogs(state.latest);

    renderAll();
  }

  function destroyChart(instanceKey) {
    if (state[instanceKey]) {
      state[instanceKey].destroy();
      state[instanceKey] = null;
    }
  }

  function renderContour() {
    const latest = state.latest;
    if (!latest) return;

    const canvas = byId("wfContourCanvas");
    const ctx = canvas.getContext("2d");

    const frameIdx = Math.floor(latest.rEq.length / 3);
    const frame = latest.rEq[frameIdx];

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#08101a";
    ctx.fillRect(0, 0, width, height);

    const cx = width * 0.5;
    const cy = height * 0.5;

    let maxR = 0;
    for (let i = 0; i < frame.length; i += 1) {
      if (frame[i] > maxR) maxR = frame[i];
    }
    const scale = 0.42 * Math.min(width, height) / maxR;

    ctx.strokeStyle = "rgba(58,79,108,0.30)";
    ctx.lineWidth = 1;
    for (let g = 1; g <= 4; g += 1) {
      const rr = (g / 4) * maxR * scale;
      ctx.beginPath();
      ctx.arc(cx, cy, rr, 0, TWO_PI);
      ctx.stroke();
    }

    ctx.beginPath();
    for (let i = 0; i < frame.length; i += 1) {
      const x = cx + frame[i] * Math.cos(latest.phi[i]) * scale;
      const y = cy + frame[i] * Math.sin(latest.phi[i]) * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();

    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, "rgba(42,72,107,0.25)");
    grad.addColorStop(1, "rgba(58,79,108,0.08)");

    ctx.fillStyle = grad;
    ctx.strokeStyle = "#2a486b";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    byId("wfContourCaption").textContent =
      `Frame ${frameIdx} shown from ${latest.params.nFrames} generated contours. Mean radius ~ ${mean(latest.meanR).toFixed(3)} um.`;
  }

  function renderXiChart() {
    const latest = state.latest;
    if (!latest) return;

    destroyChart("xiChart");

    const labels = Array.from(latest.gamma, (g) => (g * 180 / Math.PI).toFixed(1));
    state.xiChart = new Chart(byId("wfXiChart").getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "ξ(γ) average",
            data: Array.from(latest.xiAvg),
            borderColor: "#2a486b",
            backgroundColor: "rgba(42,72,107,0.08)",
            borderWidth: 2.2,
            pointRadius: 0,
            fill: true,
            tension: 0.15,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#111827",
            borderColor: "#334155",
            borderWidth: 1,
          },
        },
        scales: {
          x: {
            title: { display: true, text: "γ (degrees)", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb", maxTicksLimit: 9 },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
          y: {
            title: { display: true, text: "ξ", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb" },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
        },
      },
    });
  }

  function renderBlChart() {
    const latest = state.latest;
    if (!latest) return;

    destroyChart("blChart");

    state.blChart = new Chart(byId("wfBlChart").getContext("2d"), {
      type: "line",
      data: {
        labels: latest.lValues,
        datasets: [
          {
            label: "Projected B_l",
            data: latest.bValues,
            borderColor: "#3a4f6c",
            backgroundColor: "rgba(58,79,108,0.08)",
            pointRadius: 4,
            pointBackgroundColor: "#3a4f6c",
            pointBorderColor: "#111827",
            pointBorderWidth: 1.2,
            borderWidth: 2,
            fill: true,
            tension: 0.12,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            title: { display: true, text: "mode l", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb" },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
          y: {
            title: { display: true, text: "B_l", color: "#e5e7eb" },
            ticks: {
              color: "#e5e7eb",
              callback: (v) => Number(v).toExponential(1),
            },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
        },
      },
    });
  }

  function renderChiChart() {
    const latest = state.latest;
    if (!latest) return;

    destroyChart("chiChart");

    state.chiChart = new Chart(byId("wfChiChart").getContext("2d"), {
      type: "line",
      data: {
        labels: latest.chi.sVals,
        datasets: [
          {
            label: "χ²(s)",
            data: latest.chi.c2Vals,
            borderColor: "#6a5b42",
            backgroundColor: "rgba(106,91,66,0.08)",
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.18,
          },
          {
            label: "minimum",
            data: latest.chi.sVals.map((s) =>
              Math.abs(s - latest.fit.Sigma) < 0.4 ? latest.fit.chi2_min : null
            ),
            borderColor: "#2a486b",
            backgroundColor: "#2a486b",
            pointRadius: 6,
            showLine: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "s = Σ", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb" },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
          y: {
            title: { display: true, text: "χ²", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb" },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
        },
      },
    });
  }

  function renderFitChart() {
    const latest = state.latest;
    if (!latest) return;

    destroyChart("fitChart");

    state.fitChart = new Chart(byId("wfFitChart").getContext("2d"), {
      type: "line",
      data: {
        labels: latest.lValues,
        datasets: [
          {
            label: "Projected B_l",
            data: latest.bValues,
            borderColor: "#3a4f6c",
            backgroundColor: "#3a4f6c",
            pointRadius: 4,
            borderWidth: 0,
            showLine: false,
          },
          {
            label: "Theory B_l fit",
            data: latest.bTheory,
            borderColor: "#2a486b",
            backgroundColor: "transparent",
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false,
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: "#e5e7eb", boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "mode l", color: "#e5e7eb" },
            ticks: { color: "#e5e7eb" },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
          y: {
            title: { display: true, text: "B_l", color: "#e5e7eb" },
            ticks: {
              color: "#e5e7eb",
              callback: (v) => Number(v).toExponential(1),
            },
            grid: { color: "rgba(148,163,184,0.14)" },
          },
        },
      },
    });
  }

  function updateSummary() {
    const latest = state.latest;
    if (!latest) return;

    const p = latest.params;
    const f = latest.fit;

    byId("wfKappaTrueOut").textContent = p.kappaTrue.toFixed(2);
    byId("wfKappaFitOut").textContent = Number.isFinite(f.kappa) ? f.kappa.toFixed(2) : "nan";
    byId("wfSigmaTrueOut").textContent = p.sigmaTrue.toFixed(2);
    byId("wfSigmaFitOut").textContent = Number.isFinite(f.Sigma) ? f.Sigma.toFixed(2) : "nan";

    const fitMeta = byId("wfFitMeta");
    const kappaErrPct = Number.isFinite(f.kappa) ? (100 * Math.abs(f.kappa - p.kappaTrue) / p.kappaTrue) : NaN;
    const sigmaErrPct = Number.isFinite(f.Sigma) ? (100 * Math.abs(f.Sigma - p.sigmaTrue) / Math.max(1e-12, Math.abs(p.sigmaTrue) || 1)) : NaN;

    fitMeta.innerHTML = "";

    const rows = [
      `χ² minimum: ${f.chi2_min.toFixed(4)}`,
      `j = kBT/κ: ${f.j.toExponential(3)}`,
      `κ relative error: ${Number.isFinite(kappaErrPct) ? kappaErrPct.toFixed(2) : "nan"}%`,
      `Σ relative error: ${Number.isFinite(sigmaErrPct) ? sigmaErrPct.toFixed(2) : "nan"}%`,
    ];

    rows.forEach((txt) => {
      const li = document.createElement("li");
      li.textContent = txt;
      fitMeta.appendChild(li);
    });
  }

  function renderAll() {
    renderContour();
    renderXiChart();
    renderBlChart();
    renderChiChart();
    renderFitChart();
    renderCalculationLogs();
    updateSummary();
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise();
    }
  }

  function mean(arr) {
    let s = 0;
    for (let i = 0; i < arr.length; i += 1) s += arr[i];
    return s / Math.max(1, arr.length);
  }

  function randomizeSeed() {
    byId("wfSeed").value = String(Math.floor(1 + Math.random() * 9_000_000));
  }

  function wireEvents() {
    byId("wfRunBtn").addEventListener("click", () => {
      byId("wfRunBtn").textContent = "Running...";
      byId("wfRunBtn").disabled = true;

      // Let the button state paint before heavy computation starts.
      setTimeout(() => {
        try {
          runWorkflow();
        } catch (err) {
          // eslint-disable-next-line no-alert
          alert(`Workflow failed: ${err.message}`);
          console.error(err);
        } finally {
          byId("wfRunBtn").textContent = "Run Full Workflow";
          byId("wfRunBtn").disabled = false;
        }
      }, 10);
    });

    byId("wfReseedBtn").addEventListener("click", () => {
      randomizeSeed();
    });
  }

  function init() {
    wireEvents();
    wireStepToggles();
    runWorkflow();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();



