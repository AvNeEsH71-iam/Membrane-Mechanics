from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from quasi_spherical.autocorr_wrapper import (
    compute_angular_autocorrelation_series,
    extract_build_csv,
    time_average_xi,
)
from quasi_spherical.fitting import fit_kappa_sigma, theoretical_B_l
from quasi_spherical.generator import GeneratorConfig, generate_equatorial_series
from quasi_spherical.legendre import project_legendre_series, project_legendre_spectrum


st.set_page_config(
    page_title="Vesicle Fluctuation Spectroscopy Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Vesicle Fluctuation Spectroscopy Simulator")
st.caption(
    "Interactive equatorial-contour (theta=pi/2) fluctuation analysis with autocorrelation, Legendre decomposition, and chi-square fitting."
)


@st.cache_data(show_spinner=False)
def cached_generate_contours(
    radius: float,
    n_phi: int,
    n_frames: int,
    l_max: int,
    kappa: float,
    sigma: float,
    temperature: float,
    seed: int,
    single_mode: Optional[str],
):
    parsed_mode = None
    if single_mode:
        l_s, m_s = single_mode.split(",")
        parsed_mode = (int(l_s), int(m_s))

    config = GeneratorConfig(
        radius=radius,
        n_phi=n_phi,
        n_frames=n_frames,
        l_max=l_max,
        kappa=kappa,
        sigma=sigma,
        temperature=temperature,
        seed=seed,
        single_mode=parsed_mode,
    )
    return generate_equatorial_series(config)


@st.cache_data(show_spinner=False)
def cached_autocorrelation(r_eq: np.ndarray):
    return compute_angular_autocorrelation_series(r_eq)


@st.cache_data(show_spinner=False)
def cached_legendre(xi_t: np.ndarray, gamma: np.ndarray, fit_lmin: int, fit_lmax: int):
    l_values, b_avg = project_legendre_spectrum(xi_gamma=np.mean(xi_t, axis=0), gamma=gamma, l_min=fit_lmin, l_max=fit_lmax)
    _, b_t = project_legendre_series(xi_t=xi_t, gamma=gamma, l_min=fit_lmin, l_max=fit_lmax)
    b_sem = np.std(b_t, axis=0, ddof=1) / np.sqrt(max(1, b_t.shape[0]))
    return l_values, b_avg, b_t, b_sem


@st.cache_data(show_spinner=False)
def cached_fit(l_values, b_avg, b_sem, temperature, kappa_guess, sigma_guess):
    fit_result, _ = fit_kappa_sigma(
        l_values=l_values,
        b_values=b_avg,
        b_sem=b_sem,
        temperature=temperature,
        kappa0=kappa_guess,
        sigma0=sigma_guess,
    )
    b_model = theoretical_B_l(l_values, fit_result.kappa, fit_result.sigma, temperature)
    return fit_result, b_model


def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# Sidebar controls
with st.sidebar:
    st.header("Controls")
    data_source = st.radio(
        "Data source",
        options=["Synthetic", "Experimental (future)"],
        index=0,
        help="Synthetic mode is active now. Experimental support can be added later using the same analysis blocks.",
    )

    l_max = st.slider("l_max", min_value=2, max_value=30, value=15, step=1)
    kappa_true = st.slider("kappa (bending rigidity)", min_value=1.0e-21, max_value=5.0e-19, value=8.0e-20, step=1.0e-21, format="%.2e")
    sigma_true = st.slider("Sigma (surface tension)", min_value=-5.0, max_value=20.0, value=0.5, step=0.1)
    temperature = st.slider("Temperature (T)", min_value=250.0, max_value=350.0, value=300.0, step=1.0)
    n_frames = st.slider("Number of frames", min_value=20, max_value=2000, value=300, step=10)

    show_individual_modes = st.checkbox("Show individual modes", value=False)
    add_noise = st.checkbox("Add noise", value=False)

    noise_level = 0.0
    if add_noise:
        noise_level = st.slider("Noise std (fraction of radius)", min_value=0.0, max_value=0.05, value=0.005, step=0.001)

    st.subheader("Fit setup")
    fit_lmin = st.slider("Fit l_min", min_value=3, max_value=10, value=3, step=1)
    fit_lmax = st.slider("Fit l_max", min_value=3, max_value=10, value=min(10, l_max), step=1)
    fit_lmax = max(fit_lmax, fit_lmin)
    kappa_guess = st.slider("Initial kappa guess", min_value=1.0e-21, max_value=5.0e-19, value=1.0e-19, step=1.0e-21, format="%.2e")
    sigma_guess = st.slider("Initial Sigma guess", min_value=-5.0, max_value=20.0, value=0.1, step=0.1)

    seed = st.number_input("Random seed", min_value=0, max_value=2_000_000_000, value=12345, step=1)

    with st.expander("Optional mode isolation"):
        isolate_mode = st.checkbox("Isolate a single (l,m) mode", value=False)
        selected_mode = None
        if isolate_mode:
            l_mode = st.slider("Single mode l", min_value=2, max_value=l_max, value=min(4, l_max), step=1)
            m_mode = st.slider("Single mode m", min_value=-l_mode, max_value=l_mode, value=min(2, l_mode), step=1)
            selected_mode = f"{l_mode},{m_mode}"

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

if data_source != "Synthetic":
    st.info("Experimental mode is a placeholder in this version. Switch to Synthetic to run the full simulation pipeline.")


if run_btn and data_source == "Synthetic":
    with st.spinner("Running quasi-spherical simulation and spectral analysis..."):
        phi, r_eq, _ = cached_generate_contours(
            radius=10.0,
            n_phi=512,
            n_frames=n_frames,
            l_max=l_max,
            kappa=kappa_true,
            sigma=sigma_true,
            temperature=temperature,
            seed=int(seed),
            single_mode=selected_mode,
        )

        if add_noise and noise_level > 0.0:
            rng = np.random.default_rng(int(seed) + 1001)
            noise = rng.normal(0.0, noise_level * 10.0, size=r_eq.shape)
            r_eq = np.maximum(r_eq + noise, 1e-9)

        gamma, xi_t, _ = cached_autocorrelation(r_eq)
        xi_avg = time_average_xi(xi_t)

        l_values, b_avg, b_t, b_sem = cached_legendre(
            xi_t=xi_t,
            gamma=gamma,
            fit_lmin=fit_lmin,
            fit_lmax=min(fit_lmax, l_max),
        )

        fit_result, b_model = cached_fit(
            l_values=l_values,
            b_avg=b_avg,
            b_sem=b_sem,
            temperature=temperature,
            kappa_guess=kappa_guess,
            sigma_guess=sigma_guess,
        )

        st.session_state["sim"] = {
            "phi": phi,
            "r_eq": r_eq,
            "gamma": gamma,
            "xi_avg": xi_avg,
            "l_values": l_values,
            "b_avg": b_avg,
            "b_sem": b_sem,
            "b_model": b_model,
            "b_t": b_t,
            "fit": fit_result,
        }


if "sim" in st.session_state:
    sim = st.session_state["sim"]

    st.markdown("### Main Visualization Panel")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("A) Generated contour")
        frame_idx = st.slider("Frame index", min_value=0, max_value=sim["r_eq"].shape[0] - 1, value=0, step=1)

        fig_contour, ax_contour = plt.subplots(figsize=(6, 4))
        ax_contour.plot(sim["phi"], sim["r_eq"][frame_idx], color="black", lw=1.6)
        ax_contour.set_xlabel("phi (rad)")
        ax_contour.set_ylabel("r(phi)")
        ax_contour.set_title(f"Equatorial contour at frame {frame_idx}")
        ax_contour.grid(alpha=0.15)
        st.pyplot(fig_contour)

        st.download_button(
            "Download contour plot (PNG)",
            data=figure_to_png_bytes(fig_contour),
            file_name="contour_plot.png",
            mime="image/png",
        )
        plt.close(fig_contour)

    with col_b:
        st.subheader("B) Angular autocorrelation")
        fig_xi, ax_xi = plt.subplots(figsize=(6, 4))
        ax_xi.plot(sim["gamma"], sim["xi_avg"], color="black", lw=1.8)
        ax_xi.set_xlabel("gamma (rad)")
        ax_xi.set_ylabel("xi(gamma)")
        ax_xi.set_title("Time-averaged angular autocorrelation")
        ax_xi.grid(alpha=0.15)
        st.pyplot(fig_xi)

        st.download_button(
            "Download autocorrelation plot (PNG)",
            data=figure_to_png_bytes(fig_xi),
            file_name="angular_autocorrelation.png",
            mime="image/png",
        )
        plt.close(fig_xi)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("C) Legendre spectrum")
        fig_leg, ax_leg = plt.subplots(figsize=(6, 4))
        ax_leg.plot(sim["l_values"], sim["b_avg"], "o-", color="black", lw=1.4, label="projected")
        ax_leg.set_xlabel("l")
        ax_leg.set_ylabel("b_l")
        ax_leg.set_title("Legendre decomposition")
        ax_leg.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_leg.grid(alpha=0.15)
        ax_leg.legend(frameon=False)
        st.pyplot(fig_leg)

        st.download_button(
            "Download Legendre spectrum (PNG)",
            data=figure_to_png_bytes(fig_leg),
            file_name="legendre_spectrum.png",
            mime="image/png",
        )
        plt.close(fig_leg)

    with col_d:
        st.subheader("D) Fit comparison")
        fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
        ax_fit.errorbar(
            sim["l_values"],
            sim["b_avg"],
            yerr=sim["b_sem"],
            fmt="o",
            color="black",
            capsize=3,
            label="computed B_l",
        )
        ax_fit.plot(sim["l_values"], sim["b_model"], "--", color="dimgray", lw=1.8, label="theory B_l")

        # Optional future compatibility check if real data exists in build.
        try:
            real_leg = extract_build_csv("build/legendre_amplitudes.txt", has_header=True)
            if real_leg.size > 0 and real_leg.shape[1] > int(sim["l_values"][-1]):
                real_mean = np.mean(real_leg, axis=0)
                lo = int(sim["l_values"][0])
                hi = int(sim["l_values"][-1]) + 1
                ax_fit.plot(sim["l_values"], real_mean[lo:hi], ":", color="gray", lw=1.5, label="real mean (preview)")
        except Exception:
            pass

        ax_fit.set_xlabel("l")
        ax_fit.set_ylabel("B_l")
        ax_fit.set_title("Computed vs theoretical spectrum")
        ax_fit.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_fit.grid(alpha=0.15)
        ax_fit.legend(frameon=False)
        st.pyplot(fig_fit)

        st.download_button(
            "Download fit comparison (PNG)",
            data=figure_to_png_bytes(fig_fit),
            file_name="fit_comparison.png",
            mime="image/png",
        )
        plt.close(fig_fit)

    if show_individual_modes:
        with st.expander("Individual mode amplitudes over time", expanded=False):
            l_choice = st.selectbox("Mode l to inspect", options=list(sim["l_values"]), index=0)
            l_idx = int(np.where(sim["l_values"] == l_choice)[0][0])

            fig_modes, ax_modes = plt.subplots(figsize=(10, 3.5))
            ax_modes.plot(sim["b_t"][:, l_idx], color="black", lw=1.2)
            ax_modes.set_xlabel("Frame")
            ax_modes.set_ylabel(f"b_{l_choice}(t)")
            ax_modes.set_title(f"Temporal evolution of mode l = {l_choice}")
            ax_modes.grid(alpha=0.15)
            st.pyplot(fig_modes)
            plt.close(fig_modes)

    st.markdown("### Results Panel")
    with st.container(border=True):
        fit = sim["fit"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated kappa", f"{fit.kappa:.3e}")
        m2.metric("Estimated Sigma", f"{fit.sigma:.3e}")
        m3.metric("Chi-square", f"{fit.chi2:.3e}")
        st.caption(
            f"Uncertainty: kappa +/- {fit.kappa_std:.3e}, Sigma +/- {fit.sigma_std:.3e}; dof = {fit.dof}; optimizer success = {fit.success}"
        )

    st.markdown("### Run Instructions")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation' to generate results.")
    st.markdown("### Run Instructions")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
