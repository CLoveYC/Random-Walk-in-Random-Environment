from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "rwre_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L = 180
X = np.arange(-L, L + 1)

def potential_from_p(p: np.ndarray) -> np.ndarray:
    if p.shape != (len(X) - 1,):
        raise ValueError(f"Expected p to have shape {(len(X)-1,)}, got {p.shape}.")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("All transition probabilities must lie strictly between 0 and 1.")

    increments = np.log((1.0 - p) / p)
    V = np.concatenate(([0.0], np.cumsum(increments)))
    zero_index = np.where(X == 0)[0][0]
    return V - V[zero_index]

def make_environments():
    rng_transient = np.random.default_rng(12)
    rng_recurrent = np.random.default_rng(31)

    p_rw = np.full(len(X) - 1, 0.5)

    p_transient = rng_transient.choice(
        np.array([0.9, 0.4]),
        size=len(X) - 1,
        p=np.array([0.2, 0.8]),
    )

    p_recurrent = rng_recurrent.uniform(0.1, 0.9, size=len(X) - 1)

    return {
        "ordinary_rw": (p_rw, potential_from_p(p_rw)),
        "transient_rwre": (p_transient, potential_from_p(p_transient)),
        "recurrent_rwre": (p_recurrent, potential_from_p(p_recurrent)),
    }

def save_potential(filename_stem, V, title, subtitle, annotate=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.1))

    ax.plot(X, V, linewidth=2.2)
    ax.fill_between(X, V, 0, alpha=0.08)
    ax.axhline(0, linewidth=0.8, alpha=0.45)
    ax.axvline(0, linewidth=0.8, alpha=0.35)

    ax.set_xlim(X[0], X[-1])
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"potential $V(x)$")
    ax.set_title(title, fontsize=15, pad=14)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.5)

    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate:
        if np.allclose(V, 0):
            ax.annotate(
                "flat potential",
                xy=(0, 0),
                xytext=(55, 2.5),
                arrowprops={"arrowstyle": "->"},
                fontsize=10,
            )
        else:
            valley_index = int(np.argmin(V))
            valley_x = int(X[valley_index])
            valley_y = float(V[valley_index])

            ax.scatter([valley_x], [valley_y], zorder=4)
            ax.annotate(
                "deep valley",
                xy=(valley_x, valley_y),
                xytext=(valley_x + 45, valley_y + 8),
                arrowprops={"arrowstyle": "->"},
                fontsize=10,
            )

            left_mean = float(np.mean(V[: len(V)//3]))
            right_mean = float(np.mean(V[-len(V)//3 :]))
            if right_mean < left_mean - 5:
                ax.annotate(
                    "macroscopic downward tilt",
                    xy=(X[-45], V[-45]),
                    xytext=(X[-150], V[-45] + 10),
                    arrowprops={"arrowstyle": "->"},
                    fontsize=10,
                )

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.82)
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

def main():
    envs = make_environments()
    p_rw, V_rw = envs["ordinary_rw"]
    p_tr, V_tr = envs["transient_rwre"]
    p_rec, V_rec = envs["recurrent_rwre"]

    print("Finite-sample diagnostics")
    print(f"ordinary RW: mean log rho = 0")
    print(f"transient RWRE: mean p = {p_tr.mean():.4f}, "
          f"mean log rho = {np.mean(np.log((1-p_tr)/p_tr)):.4f}")
    print(f"recurrent RWRE: mean p = {p_rec.mean():.4f}, "
          f"mean log rho = {np.mean(np.log((1-p_rec)/p_rec)):.4f}")

    save_potential(
        "01_ordinary_rw_clean", V_rw,
        "Environment A", r"$p_x \equiv 1/2$", False
    )
    save_potential(
        "02_transient_rwre_clean", V_tr,
        "Environment B", r"$\mathbb{E}[p_x]=1/2$", False
    )
    save_potential(
        "03_recurrent_rwre_clean", V_rec,
        "Environment C", r"$p_x \sim \mathrm{Unif}(0.1,0.9)$", False
    )

    save_potential(
        "04_ordinary_rw_reveal", V_rw,
        "Ordinary symmetric random walk",
        r"$V(x)\equiv 0$: recurrent and diffusive", True
    )
    save_potential(
        "05_transient_rwre_reveal", V_tr,
        "A fair-on-average but transient RWRE",
        r"$\mathbb{E}[p_x]=1/2$ but $\mathbb{E}[\log\rho_x]<0$", True
    )
    save_potential(
        "06_recurrent_rwre_reveal", V_rec,
        "A recurrent Sinai-type RWRE",
        r"$\mathbb{E}[\log\rho_x]=0$: valleys create trapping", True
    )

if __name__ == "__main__":
    main()
