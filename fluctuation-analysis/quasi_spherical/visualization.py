from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_angular_autocorrelation(gamma: np.ndarray, xi_avg: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gamma, xi_avg, color="black", lw=2)
    ax.set_xlabel(r"$\gamma$ (rad)")
    ax.set_ylabel(r"$\langle \xi(\gamma) \rangle$")
    ax.set_title("Time-averaged angular autocorrelation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_legendre_spectrum(l_values: np.ndarray, b_values: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(l_values, b_values, "o-", color="tab:blue", lw=1.5)
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$b_l$")
    ax.set_title("Legendre spectrum from synthetic contours")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_fit_comparison(
    l_values: np.ndarray,
    b_values: np.ndarray,
    b_sem: np.ndarray,
    b_theory: np.ndarray,
    output_path: Path,
    real_b: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        l_values,
        b_values,
        yerr=b_sem,
        fmt="o",
        capsize=3,
        color="tab:blue",
        ecolor="black",
        label="synthetic projected",
    )
    ax.plot(l_values, b_theory, "--", color="tab:red", lw=2, label="theory fit")

    if real_b is not None and real_b.size == l_values.size:
        ax.plot(l_values, real_b, "-.", color="tab:green", lw=1.8, label="real-data mean")

    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$B_l$")
    ax.set_title("Theoretical vs computed spectrum")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def animate_contours(phi: np.ndarray, r_eq: np.ndarray, output_path: Path, fps: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")

    max_r = float(np.max(r_eq) * 1.1)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic equatorial contour evolution")

    x = r_eq[0] * np.cos(phi)
    y = r_eq[0] * np.sin(phi)
    (line,) = ax.plot(x, y, color="tab:purple", lw=2)

    def update(frame: int):
        x_t = r_eq[frame] * np.cos(phi)
        y_t = r_eq[frame] * np.sin(phi)
        line.set_data(x_t, y_t)
        return (line,)

    anim = FuncAnimation(fig, update, frames=r_eq.shape[0], interval=1000 / fps, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
