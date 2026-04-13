from __future__ import annotations

import pathlib
from statistics import NormalDist

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


OUT_DPI = 220


N_VEC = np.array([1000, 5000, 10000, 15000, 20000], dtype=float)
VALS = np.array(
    [
        -0.17546927855215824,
        -0.1906119630309043,
        -0.17924501776041424,
        -0.1826805612086024,
        -0.181184208323609,
    ],
    dtype=float,
)
SD_VALS = np.array(
    [
        0.04106466473642666,
        0.018296399941889575,
        0.013174287295527257,
        0.01086721462174894,
        0.009383218078206898,
    ],
    dtype=float,
)


def make_quantization_quantiles(out_path: pathlib.Path, n_bins: int = 6) -> None:
    """Illustrate equiprobable quantization of a normal distribution."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    normal = NormalDist(mu=0.0, sigma=1.0)

    # Use finite tails for display while keeping equal-probability bin logic.
    x_min = normal.inv_cdf(0.001)
    x_max = normal.inv_cdf(0.999)
    x = np.linspace(x_min, x_max, 1200)
    pdf = np.array([normal.pdf(float(xx)) for xx in x])

    # Equiprobable intervals I_i = [a_i, a_{i+1}] with P(I_i)=1/n_bins.
    p_edges = np.linspace(0.0, 1.0, n_bins + 1)
    p_nodes = (np.arange(1, n_bins + 1) - 0.5) / n_bins

    finite_edge_probs = p_edges[1:-1]
    finite_edges = np.array([normal.inv_cdf(float(p)) for p in finite_edge_probs])
    nodes = np.array([normal.inv_cdf(float(p)) for p in p_nodes])

    edges_for_plot = np.concatenate(([x_min], finite_edges, [x_max]))

    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=OUT_DPI)

    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.8, n_bins))
    for i in range(n_bins):
        left, right = edges_for_plot[i], edges_for_plot[i + 1]
        mask = (x >= left) & (x <= right)
        ax.fill_between(x[mask], pdf[mask], color=colors[i], alpha=0.36, linewidth=0)

    ax.plot(x, pdf, color="#0f172a", lw=2.2, label="Standard normal density")

    for edge in finite_edges:
        ax.axvline(edge, color="#475569", lw=1.0, ls="--", alpha=0.8)

    node_heights = np.array([normal.pdf(float(v)) for v in nodes])
    ax.scatter(nodes, node_heights, color="#b91c1c", s=34, zorder=6, label="Nodes $x_i$")

    for node, h in zip(nodes, node_heights):
        ax.plot([node, node], [0.0, h], color="#b91c1c", lw=1.0, alpha=0.9)

    ax.text(
        0.03,
        0.95,
        f"$N={n_bins}$ equiprobable bins, $\\mathbb{{P}}([x_i, x_{{i+1}}])=1/N$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#0f172a",
    )

    ax.set_xlabel("$\\epsilon$")
    ax.set_ylabel("Density")
    ax.set_title("Quantization by normal quantiles")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, float(pdf.max()) * 1.1)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def _style_estimate_axis(ax: Axes) -> None:
    ax.plot(N_VEC, VALS, color="#0f172a", marker="o", lw=2.0, ms=5)
    ax.set_xticks(N_VEC)
    ax.set_xticklabels([f"{int(n)}" for n in N_VEC])
    ax.set_ylabel("Estimate of E[X]")
    ax.grid(True, alpha=0.22)


def make_mc_estimates_layout(out_path: pathlib.Path) -> None:
    """2x1 layout: estimate curve on top, empty panel below."""

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=OUT_DPI)

    _style_estimate_axis(ax_top)
    ax_top.tick_params(axis="x", labelbottom=True)
    ax_top.set_title("Monte-Carlo estimate as a function of draws N")

    ax_bottom.axis("off")
    ax_top.set_xlabel("Number of draws N")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_mc_estimates_with_ci_and_sd_decay(out_path: pathlib.Path) -> None:
    """2x1 layout: estimate+/-1sd on top and sd decay on bottom."""

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=OUT_DPI, sharex=True)

    _style_estimate_axis(ax_top)
    ax_top.fill_between(
        N_VEC,
        VALS - SD_VALS,
        VALS + SD_VALS,
        color="#2563eb",
        alpha=0.18,
        label=r"$\pm 1\,\sigma$ interval",
    )
    ax_top.scatter(N_VEC, VALS, color="#0f172a", s=24, zorder=4)
    ax_top.set_title("Estimate with standard-deviation intervals")
    ax_top.legend(loc="upper right", frameon=True, fontsize=9)

    ax_bottom.plot(N_VEC, SD_VALS, color="#b91c1c", marker="s", lw=2.0, ms=5)
    ax_bottom.set_xticks(N_VEC)
    ax_bottom.set_xticklabels([f"{int(n)}" for n in N_VEC])
    ax_bottom.set_ylabel("Std. deviation")
    ax_bottom.set_xlabel("Number of draws N")
    ax_bottom.set_title("Decay of standard deviation with N")
    ax_bottom.grid(True, alpha=0.22)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_mc_estimates_with_ci_and_sd_decay_logx(out_path: pathlib.Path) -> None:
    """2x1 layout: estimate on log-x top panel and log-log sd decay bottom panel."""

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=OUT_DPI, sharex=True)

    _style_estimate_axis(ax_top)
    ax_top.fill_between(
        N_VEC,
        VALS - SD_VALS,
        VALS + SD_VALS,
        color="#2563eb",
        alpha=0.18,
        label=r"$\pm 1\,\sigma$ interval",
    )
    ax_top.scatter(N_VEC, VALS, color="#0f172a", s=24, zorder=4)
    ax_top.set_title("Estimate with standard-deviation intervals (log-x)")
    ax_top.legend(loc="upper right", frameon=True, fontsize=9)

    ax_bottom.plot(N_VEC, SD_VALS, color="#b91c1c", marker="s", lw=2.0, ms=5)
    ax_bottom.set_ylabel("Std. deviation")
    ax_bottom.set_xlabel("Number of draws N (log scale)")
    ax_bottom.set_title("Decay of standard deviation with N (log-log)")
    ax_bottom.grid(True, alpha=0.22)

    for ax in (ax_top, ax_bottom):
        ax.set_xscale("log")
        ax.set_xticks(N_VEC)
        ax.set_xticklabels([f"{int(n)}" for n in N_VEC])

    ax_bottom.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_gauss_hermite_integration(out_path: pathlib.Path, n_nodes: int = 10) -> None:
    """Illustrate Gauss-Hermite nodes/weights for normal expectations."""

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")

    # Hermite nodes u_i and weights w_i integrate \int f(u) exp(-u^2) du.
    u_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_nodes)

    # Single-panel chart with actual Gauss-Hermite abscissae u_i and weights w_i.
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=OUT_DPI)

    ax.vlines(u_nodes, 0.0, w_nodes, color="#1d4ed8", lw=2.0, alpha=0.9)
    ax.scatter(u_nodes, w_nodes, color="#1d4ed8", s=45, zorder=5)
    ax.set_title("Actual hermgauss points and weights: $(u_i, w_i)$")
    ax.set_xlabel(r"$u_i$ (roots of $H_N$)")
    ax.set_ylabel(r"$w_i$")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo_root / "lectures" / "discretization"

    quant_file = out_dir / "quantization_quantiles.png"
    mc_file = out_dir / "mc_estimates_layout.png"
    mc_ci_file = out_dir / "mc_estimates_ci_sd_layout.png"
    mc_ci_logx_file = out_dir / "mc_estimates_ci_sd_layout_logx.png"
    gh_file = out_dir / "gauss_hermite_nodes.png"

    make_quantization_quantiles(quant_file, n_bins=6)
    make_mc_estimates_layout(mc_file)
    make_mc_estimates_with_ci_and_sd_decay(mc_ci_file)
    make_mc_estimates_with_ci_and_sd_decay_logx(mc_ci_logx_file)
    make_gauss_hermite_integration(gh_file, n_nodes=10)

    print(f"Wrote {quant_file}")
    print(f"Wrote {mc_file}")
    print(f"Wrote {mc_ci_file}")
    print(f"Wrote {mc_ci_logx_file}")
    print(f"Wrote {gh_file}")


if __name__ == "__main__":
    main()
