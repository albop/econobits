from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt


def cobweb(ax, f, x0: float, n: int = 12, xmin: float = 0.0, xmax: float = 1.5, title: str = ""):
    xs = np.linspace(xmin, xmax, 400)
    ax.plot(xs, xs, color="black", lw=1.5, label="y = x")
    ax.plot(xs, f(xs), color="#1f77b4", lw=2, label="y = f(x)")

    x = float(x0)
    for _ in range(n):
        y = float(f(x))
        ax.plot([x, x], [x, y], color="#d62728", lw=1.5)
        ax.plot([x, y], [y, y], color="#d62728", lw=1.5)
        x = y

    ax.scatter([x0], [0], s=25, color="#d62728", zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)


def main() -> None:
    # Two simple affine maps to illustrate monotone vs oscillatory convergence
    # (Keeps the picture clean and focuses on the sign/magnitude of f'(x*)).
    f1 = lambda x: 0.55 * x + 0.25  # 0 < f'(x*) < 1
    f2 = lambda x: 0.9 - 0.55 * x  # -1 < f'(x*) < 0

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=200)

    cobweb(
        axes[0],
        f1,
        x0=0.15,
        n=10,
        xmin=0,
        xmax=1.3,
        title="Monotone convergence\n$0<f'(x^*)<1$",
    )

    cobweb(
        axes[1],
        f2,
        x0=0.15,
        n=12,
        xmin=0,
        xmax=1.3,
        title="Oscillatory convergence\n$-1<f'(x^*)<0$",
    )

    for ax in axes:
        ax.legend(loc="lower right", fontsize=9, frameon=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Cobweb diagrams (fixed-point iteration $x_{n+1}=f(x_n)$)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out = repo_root / "lectures" / "convergence" / "cobweb_intuition.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
