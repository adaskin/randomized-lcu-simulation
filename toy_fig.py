"""Toy example for the paper: six Pauli terms on three qubits in a product state (Fig. 'toy').
python toy_fig.py [--out figs]"""

import argparse, os, itertools
import numpy as np
import matplotlib.pyplot as plt
from figstyle import W, FS_BASE, FS_SMALL, apply

apply()
K_ = (
    W / 10.4
)  # the layout was drawn on a 10.4-inch canvas; sizes below are scaled to the printed width

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
P = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def pauli(s):
    M = np.array([[1.0 + 0j]])
    for c in s:
        M = np.kron(M, P[c])
    return M


names = ["XII", "IXI", "IIX", "ZZI", "IZZ", "YYI"]
coef = np.array([1.0, 0.8, 0.9, -0.7, 0.6, 0.5])
lab = [r"$X_1$", r"$X_2$", r"$X_3$", r"$-Z_1Z_2$", r"$Z_2Z_3$", r"$Y_1Y_2$"]
th = [2.107, 0.728, 2.225]
ph = [2.949, 0.19, 0.281]  # product state (fixed; chosen to make the contrast visible)
a = np.abs(coef) / np.abs(coef).sum()
U = [np.sign(c) * pauli(s) for c, s in zip(coef, names)]
psi = np.array([1.0 + 0j])
for t, p in zip(th, ph):
    psi = np.kron(psi, [np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)])
x = np.array([(u @ psi) - (psi.conj() @ u @ psi) * psi for u in U])
K = np.real(x.conj() @ x.T)
L = 6


def price(parts):
    return sum(a[b] @ K[np.ix_(b, b)] @ a[b] / a[b].sum() for b in parts) - a @ K @ a


good, bad = [[0, 1, 2], [3, 4, 5]], [[0, 2, 3], [1, 4, 5]]
E1, Eg, Eb = price([[i] for i in range(L)]), price(good), price(bad)
y = x - a @ x
Y = np.concatenate([y.real, y.imag], 1)
C = (Y * a[:, None]).T @ Y
w, V = np.linalg.eigh(C)
V = V[:, ::-1]
P2 = Y @ V[:, :2]
cap = w[::-1][:2].sum() / w.sum()
print(f"E1={E1:.3f} good={Eg:.3f} poor={Eb:.3f} variance in projection={cap:.2f}")

fig, ax = plt.subplots(
    1, 3, figsize=(W, 4.5 * K_ * 1.1), gridspec_kw=dict(width_ratios=[0.9, 1, 1])
)
ang = np.pi / 2 - np.arange(L) * 2 * np.pi / L
pos = np.c_[np.cos(ang), np.sin(ang)]
for g in range(L):
    for h in range(g + 1, L):
        k = K[g, h]
        if abs(k) < 0.02:
            continue
        ax[0].plot(
            *zip(pos[g], pos[h]),
            color="#2563a8" if k < 0 else "#c0392b",
            lw=0.4 + 4.5 * abs(k),
            alpha=0.85,
            zorder=1,
            solid_capstyle="round",
        )
for g in range(L):
    ax[0].scatter(
        *pos[g], s=230 * K_ * a[g], color="white", edgecolor="k", lw=1.0, zorder=2
    )
    ax[0].text(*(pos[g] * 1.38), lab[g], ha="center", va="center", fontsize=FS_BASE)
ax[0].plot([], [], color="#c0392b", lw=3, label=r"$K_{gh}>0$: move together")
ax[0].plot([], [], color="#2563a8", lw=3, label=r"$K_{gh}<0$: compensate")
ax[0].plot(
    [], [], " ", label="circle area $\\propto a_g$, edge width $\\propto |K_{gh}|$"
)
ax[0].legend(
    frameon=False,
    fontsize=FS_SMALL,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=1,
)
ax[0].set_xlim(-1.7, 1.7)
ax[0].set_ylim(-1.7, 1.7)
ax[0].set_aspect("equal")
ax[0].axis("off")
ax[0].set_title("(a) term correlation graph", loc="left", fontsize=FS_BASE)
cols = ["#e08e0b", "#2a9d55"]
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import matplotlib.patheffects as pe

halo = [pe.withStroke(linewidth=2, foreground="white")]
# label offsets (fractions of the panel half-width) chosen so that labels stay clear of the stars and of each other
OFF = {
    "good": {
        0: (0.0, 0.2),
        1: (-0.24, 0.0),
        2: (0.3, -0.12),
        3: (0.0, 0.2),
        4: (0.0, -0.2),
        5: (0.0, 0.2),
    },
    "bad": {
        0: (0.0, 0.2),
        1: (-0.25, -0.02),
        2: (0.1, -0.22),
        3: (0.0, 0.2),
        4: (0.0, -0.2),
        5: (0.0, 0.2),
    },
}
for i, (Pt, E, tt, key) in enumerate(
    [(good, Eg, "(b) good grouping", "good"), (bad, Eb, "(c) poor grouping", "bad")]
):
    A = ax[i + 1]
    l0 = np.abs(P2).max()
    lim = l0 * 1.4
    for c, b in enumerate(Pt):
        pts = P2[b]
        hull = pts[ConvexHull(pts).vertices]
        A.add_patch(
            Polygon(
                hull,
                closed=True,
                facecolor=cols[c],
                edgecolor=cols[c],
                alpha=0.16,
                lw=0,
                zorder=1,
            )
        )
        A.add_patch(
            Polygon(
                hull,
                closed=True,
                facecolor="none",
                edgecolor=cols[c],
                alpha=0.7,
                lw=1.0,
                ls="--",
                zorder=1,
            )
        )
        m = (a[b] @ pts) / a[b].sum()
        for g in b:
            A.plot(
                [m[0], P2[g, 0]],
                [m[1], P2[g, 1]],
                color=cols[c],
                lw=0.9,
                alpha=0.8,
                zorder=2,
            )
            A.scatter(
                *P2[g],
                s=520 * K_ * a[g],
                color="white",
                edgecolor=cols[c],
                lw=1.2,
                zorder=3,
            )
            dx, dy = OFF[key][g]
            A.text(
                P2[g, 0] + dx * l0,
                P2[g, 1] + dy * l0,
                lab[g],
                fontsize=FS_BASE,
                ha="center",
                va="center",
                zorder=5,
                path_effects=halo,
            )
        A.annotate(
            "",
            xy=m,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=cols[c],
                lw=1.5,
                mutation_scale=9,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=4,
        )
        A.scatter(
            *m, marker="*", color=cols[c], s=150 * K_, edgecolor="k", lw=0.7, zorder=6
        )
    A.scatter(0, 0, marker="+", color="k", s=120 * K_, lw=1.8, zorder=7)
    A.text(
        0.07 * l0,
        -0.06 * l0,
        r"$\mu$",
        fontsize=FS_BASE + 1.5,
        ha="left",
        va="top",
        zorder=7,
        path_effects=halo,
    )
    A.set_title(tt, loc="left", fontsize=FS_BASE)
    A.text(
        0.5,
        -0.03,
        rf"$E={E:.3f}$   (qDRIFT: $E_1={E1:.3f}$)",
        transform=A.transAxes,
        fontsize=FS_BASE,
        ha="center",
        va="top",
    )
    A.set_xticks([])
    A.set_yticks([])
    A.set_aspect("equal")
    for sp in A.spines.values():
        sp.set_visible(True)
        sp.set_color("0.6")
    A.set_xlim(-lim, lim)
    A.set_ylim(-lim, lim)
from matplotlib.lines import Line2D

fig.legend(
    handles=[
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            color="0.75",
            mec="k",
            ms=10,
            label=r"block mean $\mu_c$",
        ),
        Line2D(
            [],
            [],
            marker="+",
            ls="",
            color="k",
            ms=8,
            mew=1.8,
            label=r"overall mean $\mu$",
        ),
        Line2D(
            [],
            [],
            color="0.4",
            lw=2.2,
            marker=">",
            ms=0,
            label=r"$\mu\to\mu_c$ (weighted squared length $=E$)",
        ),
    ],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=FS_SMALL,
    bbox_to_anchor=(0.66, 0.0),
    columnspacing=1.2,
    handletextpad=0.4,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(args.out, "fig_toy.pdf"), bbox_inches="tight")
print("ok")
