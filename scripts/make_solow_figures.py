from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt


def _style_axes(ax):
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("$k_t$")
    ax.set_ylabel("$k_{t+1}$")


def make_standard_solow(out_path: pathlib.Path) -> None:
    # Standard Solow mapping: k_{t+1} = (1-δ)k_t + s k_t^α
    # Parameters here are intentionally NOT realistic: we want the curvature
    # to be visually obvious in slides.
    alpha = 0.5
    delta = 0.85
    s = 4.0

    def solow_map(x: np.ndarray) -> np.ndarray:
        return (1.0 - delta) * x + s * (x**alpha)

    # Start with a moderate range and expand until we bracket a fixed point
    # (so the crossing is visible on the slide).
    k_max = 12.0
    k = np.linspace(0.0, k_max, 800)
    f = solow_map(k)
    diff = f - k
    for _ in range(8):
        # Use a strict sign change to avoid picking up the trivial root at k=0
        # due to sign(0)=0.
        if np.any(diff[1:] * diff[:-1] < 0):
            break
        k_max *= 1.6
        k = np.linspace(0.0, k_max, 1200)
        f = solow_map(k)
        diff = f - k

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    ax.plot(k, k, color="black", lw=1.6, label="$k_{t+1}=k_t$")
    ax.plot(k, f, color="#1f77b4", lw=2.2, label="$k_{t+1}=(1-\\delta)k_t + s k_t^\\alpha$")

    # Mark the economically relevant steady state: the positive fixed point.
    # With typical Solow-style maps, k=0 is always a fixed point; we skip it.
    crossings = np.where(diff[1:] * diff[:-1] < 0)[0]
    if len(crossings) > 0:
        # Choose the *last* crossing (usually the positive steady state).
        i = int(crossings[-1])
        k0, k1 = float(k[i]), float(k[i + 1])
        d0, d1 = float(diff[i]), float(diff[i + 1])
        k_star = k0 - d0 * (k1 - k0) / (d1 - d0)
    else:
        # Fallback: pick the best positive candidate (avoid k=0).
        mask = k > 1e-6
        idx = int(np.argmin(np.abs(diff[mask])))
        k_star = float(k[mask][idx])
    ax.scatter([k_star], [k_star], color="#d62728", s=35, zorder=5)
    ax.annotate(
        "$k^*$",
        xy=(k_star, k_star),
        xytext=(k_star + 0.8, k_star - 1.2),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#d62728"),
        fontsize=12,
        color="#d62728",
    )

    ax.set_title("Solow map (illustrative)")
    _style_axes(ax)

    # Keep the cobweb geometry readable by using a square plot range that
    # contains both the x-range (including k*) and the generated y-values.
    y_max = float(np.nanmax(f))
    m = max(float(k_max), y_max, k_star) * 1.02
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")


def make_multiple_fixed_points(out_path: pathlib.Path) -> None:
    # Illustrative nonlinear map with 3 fixed points.
    # We build it so that f(k) - k has three roots.
    a, b, c = 1.5, 4.5, 8.5
    eps = 0.012

    k = np.linspace(0.0, 12.0, 800)
    f = k + eps * (k - a) * (k - b) * (k - c)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    ax.plot(k, k, color="black", lw=1.6, label="$k_{t+1}=k_t$")
    ax.plot(k, f, color="#1f77b4", lw=2.2, label="$k_{t+1}=f(k_t)$ (nonlinear) ")

    # Mark the three fixed points (by construction, approx at a,b,c)
    for kk in (a, b, c):
        ax.scatter([kk], [kk], color="#d62728", s=25, zorder=5)

    ax.set_title("Multiple fixed points (illustrative)")
    _style_axes(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo_root / "lectures" / "convergence"

    make_standard_solow(out_dir / "solow_generated.png")
    make_multiple_fixed_points(out_dir / "solow_multiple_fixed_points.png")

    print(f"Wrote {out_dir / 'solow_generated.png'}")
    print(f"Wrote {out_dir / 'solow_multiple_fixed_points.png'}")


if __name__ == "__main__":
    main()
