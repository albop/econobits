from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np


OUT_DPI = 200


def make_newton_success_failure(out_path: pathlib.Path) -> None:
    """Illustrate Newton on the same function with good vs bad initial guesses."""

    def f(x: np.ndarray) -> np.ndarray:
        return x**3 - 2.0 * x + 2.0

    def fp(x: np.ndarray) -> np.ndarray:
        return 3.0 * x**2 - 2.0

    def iterate(x0: float, n: int = 6) -> list[float]:
        xs = [float(x0)]
        x = float(x0)
        for _ in range(n):
            d = float(fp(np.array(x)))
            if abs(d) < 1e-10:
                break
            x = x - float(f(np.array(x))) / d
            xs.append(float(x))
        return xs

    xgrid = np.linspace(-3.0, 2.2, 900)
    ygrid = f(xgrid)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=OUT_DPI, sharey=True)

    setups = [
        ("Good initial guess", -1.8, "#2ca02c"),
        ("Bad initial guess (cycles)", 0.0, "#d62728"),
    ]

    for ax, (title, x0, color) in zip(axes, setups):
        ax.plot(xgrid, ygrid, color="#1f77b4", lw=2.0, label="f(x) = x^3 - 2x + 2")
        ax.axhline(0.0, color="black", lw=1.0)

        xs = iterate(x0)
        for i, x in enumerate(xs[:-1]):
            y = float(f(np.array(x)))
            xn = xs[i + 1]
            # Vertical to curve, then tangent projection to x-axis.
            ax.plot([x, x], [0, y], color=color, lw=1.5, alpha=0.9)
            ax.plot([x, xn], [y, 0], color=color, lw=1.5, alpha=0.9)

        ax.scatter(xs[0], 0.0, s=45, color=color, zorder=5)
        ax.text(xs[0] + 0.06, 0.24, "x0", color=color, fontsize=10)

        ax.set_title(title, fontsize=12)
        ax.set_xlim(-3.0, 2.2)
        ax.set_ylim(-6.5, 7.5)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x")

    axes[0].set_ylabel("f(x)")
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    fig.suptitle("Newton method: local speed vs global fragility", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_newton_overshoot(out_path: pathlib.Path) -> None:
    """Illustrate a Newton step that points to the root but goes too far."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.arctan(4.0 * x)

    def fp(x: np.ndarray) -> np.ndarray:
        return 4.0 / (1.0 + (4.0 * x) ** 2)

    x0 = 0.5
    y0 = float(f(np.array(x0)))
    slope0 = float(fp(np.array(x0)))
    x1 = x0 - y0 / slope0

    xgrid = np.linspace(-1.4, 1.4, 700)
    ygrid = f(xgrid)
    y_tangent = y0 + slope0 * (xgrid - x0)

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0), dpi=OUT_DPI)

    ax.plot(xgrid, ygrid, color="#1f77b4", lw=2.2, label="f(x) = arctan(4x)")
    ax.plot(xgrid, y_tangent, color="#ff7f0e", lw=2.0, ls="--", label="Tangent at x0")
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.3)

    # Newton geometry: from x0 to curve, then tangent projection to x-axis.
    ax.plot([x0, x0], [0.0, y0], color="#2ca02c", lw=2.0)
    ax.plot([x0, x1], [y0, 0.0], color="#d62728", lw=2.0)

    ax.scatter([x0], [0.0], color="#2ca02c", s=55, zorder=5)
    ax.scatter([x1], [0.0], color="#d62728", s=55, zorder=5)
    ax.scatter([0.0], [0.0], color="#111111", s=45, zorder=5)

    ax.annotate(
        "x0",
        xy=(x0, 0.0),
        xytext=(x0 + 0.04, -0.16),
        color="#2ca02c",
        fontsize=11,
    )
    ax.annotate(
        "Newton step x1",
        xy=(x1, 0.0),
        xytext=(x1 - 0.55, 0.18),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        color="#d62728",
        fontsize=10,
    )
    ax.annotate(
        "root x*",
        xy=(0.0, 0.0),
        xytext=(0.07, 0.11),
        color="#111111",
        fontsize=10,
    )
    ax.text(
        0.02,
        0.94,
        "Starts moving left toward x* but overshoots past the root",
        transform=ax.transAxes,
        fontsize=10,
        color="#444444",
        va="top",
    )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Newton overshooting in one step", fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_bisection_progress(out_path: pathlib.Path) -> None:
    """Show shrinking brackets for bisection on cos(x)-x."""

    def f(x: np.ndarray) -> np.ndarray:
        return np.cos(x) - x

    a, b = 0.0, 1.0
    brackets: list[tuple[float, float, float]] = []

    for _ in range(7):
        c = 0.5 * (a + b)
        brackets.append((a, b, c))
        if float(f(np.array(a))) * float(f(np.array(c))) <= 0:
            b = c
        else:
            a = c

    xgrid = np.linspace(0.0, 1.0, 500)
    ygrid = f(xgrid)

    fig, (ax_curve, ax_interval) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        dpi=OUT_DPI,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    ax_curve.plot(xgrid, ygrid, color="#1f77b4", lw=2.0, label="f(x)=cos(x)-x")
    ax_curve.axhline(0.0, color="black", lw=1.0)
    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(-0.5, 1.1)
    ax_curve.set_ylabel("f(x)")
    ax_curve.grid(True, alpha=0.2)
    ax_curve.legend(loc="upper right", frameon=True, fontsize=9)

    for i, (left, right, mid) in enumerate(brackets):
        y = len(brackets) - i
        ax_interval.plot([left, right], [y, y], color="#d62728", lw=3.0)
        ax_interval.scatter([mid], [y], color="#2ca02c", s=28, zorder=5)
        ax_interval.text(right + 0.012, y, f"n={i}", va="center", fontsize=9)

    ax_interval.set_xlim(0.0, 1.0)
    ax_interval.set_ylim(0.3, len(brackets) + 1.2)
    ax_interval.set_xlabel("x")
    ax_interval.set_ylabel("interval")
    ax_interval.set_yticks([])
    ax_interval.grid(True, axis="x", alpha=0.2)

    fig.suptitle("Bisection: bracket shrinks every iteration", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def make_kkt_geometry(out_path: pathlib.Path) -> None:
    """Two-panel KKT geometry: interior and corner optimum."""

    p1, p2, b = 1.0, 1.0, 4.0
    x = np.linspace(0.001, b - 0.001, 500)

    # Interior case with Cobb-Douglas utility U=x1^0.5 x2^0.5.
    x2_budget = (b - p1 * x) / p2
    u_star = 2.0  # At (2,2), U=sqrt(4)=2.
    x2_indiff = (u_star**2) / x

    # Corner case for linear utility U=2x1+x2 (slope -2): optimum at (4,0).
    x2_linear = (8.0 - 2.0 * x)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=OUT_DPI, sharex=True, sharey=True)

    ax = axes[0]
    ax.plot(x, x2_budget, color="#1f77b4", lw=2.2, label="Budget line")
    ax.plot(x, x2_indiff, color="#ff7f0e", lw=1.8, label="Indifference curve")
    ax.scatter([2.0], [2.0], s=45, color="#2ca02c", zorder=5)
    ax.annotate(
        "interior optimum",
        xy=(2.0, 2.0),
        xytext=(2.35, 2.55),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#2ca02c"),
        color="#2ca02c",
        fontsize=10,
    )
    ax.set_title("Binding constraint, interior solution", fontsize=11)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(x, x2_budget, color="#1f77b4", lw=2.2, label="Budget line")
    valid = x2_linear >= 0
    ax.plot(x[valid], x2_linear[valid], color="#ff7f0e", lw=1.8, label="Linear indifference")
    ax.scatter([4.0], [0.0], s=45, color="#d62728", zorder=5)
    ax.annotate(
        "corner optimum",
        xy=(4.0, 0.0),
        xytext=(2.5, 0.75),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#d62728"),
        color="#d62728",
        fontsize=10,
    )
    ax.set_title("Binding + inequality gives corner", fontsize=11)
    ax.grid(True, alpha=0.2)

    for ax in axes:
        ax.set_xlim(0, 4.2)
        ax.set_ylim(0, 4.2)
        ax.set_xlabel("x1")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    axes[1].legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle("KKT geometry: tangency vs corner", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo_root / "lectures" / "optimization"

    make_newton_success_failure(out_dir / "newton_success_failure.png")
    make_newton_overshoot(out_dir / "newton_overshoot.png")
    make_bisection_progress(out_dir / "bisection_progress.png")
    make_kkt_geometry(out_dir / "kkt_geometry.png")

    print(f"Wrote {out_dir / 'newton_success_failure.png'}")
    print(f"Wrote {out_dir / 'newton_overshoot.png'}")
    print(f"Wrote {out_dir / 'bisection_progress.png'}")
    print(f"Wrote {out_dir / 'kkt_geometry.png'}")


if __name__ == "__main__":
    main()
