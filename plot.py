"""Make the paper figures from saved cases (no simulation here).

python plot.py [--root runs] [--out figs] [--n 15]

Reads the saved cases of configs/grid.py under <root>/cases/ (analysis.grid_cases). The main systems of the figures are model A
and model B at --n qubits (default 15): B at its largest L, and at L = DRIFT_L (the drift and block-number figures need the
per-time graphs, which are only kept up to L = 4096). Style: figstyle.py.
"""
import argparse, os
import numpy as np, json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator
from analysis import *
from figstyle import W, W_HALF, FS_BASE, FS_SMALL, C, DASH, apply

apply()
TAU = 100.0
S = 0.62  # line and marker scale relative to the 7.6-inch layout these figures were designed on
SAVE = dict(bbox_inches="tight")



def thin(fig, keep=()):
    """few ticks: at most ~4 per axis, no minor ticks"""
    for A in fig.axes:
        for nm, axis in (("x", A.xaxis), ("y", A.yaxis)):
            if (id(A), nm) in keep:
                continue
            if axis.get_scale() == "log":
                axis.set_major_locator(LogLocator(base=10, numticks=5))
            else:
                axis.set_major_locator(MaxNLocator(nbins=4, integer=False))
            axis.set_minor_locator(NullLocator())


FS = (W, 3.0)
LAB = dict(
    qDRIFT="qDRIFT",
    full="full $H$ (LCU)",
    random="random blocks",
    magnitude="magnitude composite",
    graph="graph blocks ($t=0$)",
    oracle="oracle blocks",
)
MS = ("full", "qDRIFT", "magnitude", "random", "graph")
MK = dict(qDRIFT="s", full="D", random="v", magnitude="^", graph="o", oracle="P")
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--out", default="figs")
ap.add_argument("--n", type=int, default=15, help="qubit number of the main systems")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
by = grid_cases(root=args.root)
NQ = args.n
DRIFT_L = 2048 if 2048 in study_tags(by, "B", NQ) else max(study_tags(by, "B", NQ))  # a smaller n has no L = 2048
mainA = by[f"A n={NQ}"]
BSER = study_tags(by, "B", NQ)  # {L: [instances]}
Ls = sorted(BSER)
SERB = [r for L in Ls for r in BSER[L]]
mainB = BSER[Ls[-1]][0]
driftB = BSER[DRIFT_L][0]
TA = f"(a) model A, $n={NQ}$, $L={mainA['L']}$"
TB = f"(b) model B, $n={NQ}$, $L={mainB['L']}$"


def sty(m, every=3, stagger=0):
    """line style of method m: the proposed method (graph) is the thick one, hollow markers for the two that overlap.
    stagger>0: markers every `every` points, shifted per method so that coinciding curves still show every marker"""
    return dict(
        marker=MK[m],
        markevery=(2 + stagger * MS.index(m), every) if stagger else (1, every),
        mfc="white" if m in ("qDRIFT", "magnitude") else None,
        color=C[m],
        ls=DASH[m],
        zorder=5 if m == "graph" else 4 if m == "full" else 3,
    )


def handles(ms=MS):
    return [Line2D([], [], label=LAB[m], **{k: v for k, v in sty(m).items() if k not in ("markevery", "zorder")}) for m in ms]


def legend_below(fig, hs, ncol, y=-0.02):
    fig.legend(handles=hs, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), columnspacing=1.4, handletextpad=0.5)


def tag_box(ax, text, color, loc=(0.5, 0.94), ha="center"):
    ax.text(*loc, text, transform=ax.transAxes, ha=ha, va="top", fontsize=FS_BASE, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=10)


# ---------------------------------------------------------------- Fig: cost vs accuracy
fig, ax = plt.subplots(1, 2, figsize=FS, sharey=True)
eps = np.logspace(np.log10(0.5), -4, 120)
for A, rec, t in zip(ax, (mainA, mainB), (TA, TB)):
    curves = {m: np.array([N_eps(rec, e, m)[0] for e in eps]) for m in MS}
    for m in MS:
        A.loglog(eps, curves[m], **sty(m, every=17, stagger=3))
    A.set_xlabel(r"target infidelity $\varepsilon$  (tighter $\rightarrow$)")
    A.set_title(t, loc="left")
    A.invert_xaxis()
    # where the cheapest randomized scheme (graph blocks) beats full H
    xs = eps_cross(rec)
    if xs is None:
        tag_box(A, "full $H$ cheapest at every $\\varepsilon$", C["full"], loc=(0.97, 0.26), ha="right")
    else:
        A.axvspan(eps[0], xs, color=C["graph"], alpha=0.10, lw=0)
        A.axvline(xs, color=C["graph"], lw=1.2, ls="--")
        tag_box(A, f"graph blocks cheapest\nfor $\\varepsilon\\gtrsim{xs:.2g}$", C["graph"], loc=(0.04, 0.95), ha="left")
        tag_box(A, "full $H$ cheapest", C["full"], loc=(0.97, 0.22), ha="right")
ax[0].set_ylabel("T gates to reach $\\varepsilon$\n(noiseless, $\\tau=100$)")
ax[0].set_ylim(1e4, 5e9)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
legend_below(fig, handles(), 5)
thin(fig, keep={(id(A), "x") for A in ax})
for A in ax:
    A.set_xticks([1e-1, 1e-2, 1e-3, 1e-4]); A.minorticks_off()
fig.savefig(os.path.join(args.out, "fig_cost.pdf"), **SAVE)
plt.close()

# ---------------------------------------------------------------- Fig: noise
fig, ax = plt.subplots(1, 2, figsize=FS, sharey=True)
ps = np.logspace(-13, -6, 140)
TARGET = 1e-2
for A, rec, t in zip(ax, (mainA, mainB), (TA, TB)):
    for m in MS:
        y = np.array([min(eps_noisy(rec, p, m)[0], 1) for p in ps])
        A.loglog(ps, y, **sty(m, every=17, stagger=3))
        if m in ("full", "qDRIFT", "graph"):  # largest error rate at which the target is still reachable
            pstar = cross(ps, y, np.full_like(y, TARGET))
            if pstar:
                A.plot([pstar], [TARGET], marker="o", ms=9, mfc="white", mec=C[m], mew=1.6, zorder=8, ls="")
    A.axhline(TARGET, color="k", lw=1.2, ls=":")
    A.set_xlabel("error rate per T gate $p$  (noisier $\\rightarrow$)")
    A.set_title(t, loc="left")
    A.set_ylim(1e-8, 3)
ax[0].set_ylabel("smallest reachable infidelity\n(steps optimized)")
ax[0].text(1.5e-13, TARGET * 1.5, "target $10^{-2}$", fontsize=FS_SMALL, va="bottom")
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
hs = handles() + [Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec="0.2", mew=1.4, label="largest $p$ reaching $10^{-2}$")]
legend_below(fig, hs, 3, y=-0.07)
fig.subplots_adjust(bottom=0.34)
thin(fig, keep={(id(A), "x") for A in ax})
for A in ax:
    A.set_xticks([1e-12, 1e-10, 1e-8, 1e-6]); A.minorticks_off()
fig.savefig(os.path.join(args.out, "fig_noise.pdf"), **SAVE)
plt.close()

# ---------------------------------------------------------------- Fig: scaling with problem size
SC = ("full", "qDRIFT", "magnitude", "random", "graph")
ns = sorted(r["n"] for r in by.values() if r["tag"].startswith("A n=") and r["init"]["kind"] == "neel")
EPS_ROWS = (1e-1, 1e-2)
fig, ax = plt.subplots(2, 2, figsize=(W, 4.9), sharex="col", sharey="row")
for row, e in enumerate(EPS_ROWS):
    means = {}
    for m in SC:
        st = sty(m, every=1)
        ax[row, 0].plot(ns, [N_eps(by[f"A n={n}"], e, m)[0] for n in ns], **st)
        v = [[N_eps(r, e, m)[0] for r in BSER[L]] for L in Ls]
        means[m] = np.array([np.mean(x) for x in v])
        ax[row, 1].errorbar(Ls, means[m], [np.std(x) for x in v], capsize=2.5, elinewidth=1.0, **st)
    Lx = cross(Ls, means["graph"], means["full"])
    if Lx:
        ax[row, 1].axvspan(Lx, Ls[-1] * 1.15, color=C["graph"], alpha=0.10, lw=0)
        ax[row, 1].axvline(Lx, color=C["graph"], lw=1.0, ls="--")
        tag_box(ax[row, 1], f"graph blocks cheaper\nthan full $H$ for $L\\gtrsim{Lx / 10 ** int(np.log10(Lx)):.1f}\\times10^{{{int(np.log10(Lx))}}}$", C["graph"], loc=(0.96, 0.06), ha="right")
        ax[row, 1].texts[-1].set_va("bottom")
    ax[row, 0].set_ylabel(f"T gates for $\\varepsilon={e:g}$")
    for col in (0, 1):
        ax[row, col].set_yscale("log")
ax[1, 0].set_xlabel("qubits $n$ (model A)"); ax[1, 0].set_xticks(ns)
ax[1, 1].set_xlabel("number of terms $L$ (model B)")
ax[1, 1].set_xscale("log")
ax[0, 0].set_title("(a) model A, $L=3n(n-1)/2+2n$", loc="left")
ax[0, 1].set_title(f"(b) model B, $n={NQ}$", loc="left")
fig.tight_layout(rect=(0, 0.09, 1, 1))
legend_below(fig, handles(SC), 3)
thin(fig, keep={(id(A), "x") for A in ax.flat})
short = lambda L: f"{L // 1024}Ki" if L >= 1024 else str(L)
for row in (0, 1):
    ax[row, 0].set_xticks(ns[::2] if len(ns) > 6 else ns)
    tk = Ls[::2] if len(Ls) > 6 else Ls
    ax[row, 1].set_xticks(tk); ax[row, 1].set_xticklabels([short(t) for t in tk]); ax[row, 1].minorticks_off()
    ax[row, 1].set_xlim(Ls[0] * 0.85, Ls[-1] * 1.18)
fig.savefig(os.path.join(args.out, "fig_scaling.pdf"), **SAVE)
plt.close()

# ---------------------------------------------------------------- Fig: error per step, along the evolution and against k
fig, ax = plt.subplots(2, 2, figsize=(W, 4.9), sharey="row")
cases = ((mainA, TA), (driftB, f"(b) model B, $n={NQ}$, $L={driftB['L']}$"))
RB = ("random", "graph", "oracle")
for col, (rec, ttl) in enumerate(cases):
    d = rec["drift"]; ts = np.array(d["ts"]); E1t = np.array(d["E1"])
    A0 = ax[0, col]
    A0.plot(ts, E1t, **sty("qDRIFT", every=2))
    A0.plot(ts, np.array(d["random"]) * E1t, **sty("random", every=2))
    A0.plot(ts, np.array(d["graph"]) * E1t, **sty("graph", every=2))
    A0.set_yscale("log"); A0.set_ylim(0.02, 3)
    A0.set_xlabel(r"evolution time $\tau=\lambda t$"); A0.set_xticks([0, 50, 100])
    A0.set_title(ttl + f";  $k={d['k']}$ blocks", loc="left", fontsize=FS_BASE)
    A0.text(0.03, 0.04, "full LCU: $E=0$ (off the scale)", transform=A0.transAxes, fontsize=FS_SMALL, color=C["full"], )
    A1 = ax[1, col]
    L = rec["L"]
    for m in RB:
        pl = rec["plans"][m]
        ks = sorted({p["k"] for p in pl if p["k"] > 1})
        y = [np.mean([p["Eb"] for p in pl if p["k"] == k]) for k in ks]
        A1.plot(ks, y, **sty(m, every=2))
    A1.plot([L], [rec["E1"]], **{**sty("qDRIFT"), "ls": "", "ms": 8, "markevery": None})
    A1.annotate("qDRIFT ($k=L$)", (L, rec["E1"]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=FS_SMALL, color=C["qDRIFT"])
    A1.axvline(d["k"], color="0.35", lw=1.2, ls="--", zorder=1)
    A1.text(d["k"] * 1.12, 1.0e-3, "$k$ used\nabove", fontsize=FS_SMALL, color="0.25", va="bottom", ha="left")
    A1.annotate("", xy=(1.0, 4.5e-4), xytext=(1.0, 1.6e-3), arrowprops=dict(arrowstyle="-|>", color=C["full"], lw=2.2), annotation_clip=False)
    A1.text(1.25, 4.8e-4, "full LCU $k=1$: $\\bar E=0$", fontsize=FS_SMALL, color=C["full"], va="bottom", ha="left")
    A1.set_xscale("log"); A1.set_yscale("log")
    A1.set_xlabel("number of blocks $k$  (more blocks $\\rightarrow$)")
    A1.set_xlim(0.8, L * 2.2); A1.set_ylim(3e-4, 4)
ax[0, 0].set_ylabel("error per step $E(\\tau)$")
ax[1, 0].set_ylabel("average error per step $\\bar E$")
fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.4)
legend_below(fig, handles(("qDRIFT", "random", "graph", "oracle")), 4)
thin(fig, keep={(id(A), "x") for A in ax.flat})
for col in (0, 1):
    ax[1, col].xaxis.set_major_locator(LogLocator(base=10, numticks=5)); ax[1, col].xaxis.set_minor_locator(NullLocator())
fig.savefig(os.path.join(args.out, "fig_robust.pdf"), **SAVE)
plt.close()

# ---------------------------------------------------------------- Fig: validation of the error formula
V = list(grid_cases("validate", args.root))
if V:
    fig, ax = plt.subplots(figsize=(W_HALF, 3.3))
    z = np.logspace(-3.3, -0.4)
    ax.fill_between(z, 0.9 * z, 1.1 * z, color="0.88", lw=0, label=r"$\pm10\%$")
    ax.loglog(z, z, "k:", lw=1.3, label="perfect agreement")
    MKV = {"A": "o", "B": "s"}
    CP = dict(qDRIFT=C["qDRIFT"], graph=C["graph"], random=C["random"])
    for x in V:
        long = "tau=100" in x["tag"]  # filled: tau = 100 (the paper's evolution time); hollow: tau = 8
        ax.errorbar(x["pred"], x["mc"], x["se"], fmt=MKV[x["tag"][0]], mfc=CP[x["plan"]] if long else "white", mec=CP[x["plan"]],
                    ecolor=CP[x["plan"]], capsize=2.5, elinewidth=1.0, zorder=4)
    h = [Line2D([], [], ls="", marker=MKV[t], mfc="white", mec="0.2", label=f"model {t}") for t in MKV]
    h += [Line2D([], [], ls="", marker="o", mfc="white", mec="0.2", label=r"$\tau=8$"),
          Line2D([], [], ls="", marker="o", mfc="0.2", mec="0.2", label=r"$\tau=100$")]
    h += [Line2D([], [], ls="", marker="s", mfc=CP[p], mec=CP[p], label=LAB.get(p, p).replace(" ($t=0$)", "")) for p in CP]
    ax.set_xlabel(r"predicted infidelity $\theta^2\sum_sE_{\psi_s}$")
    ax.set_ylabel("measured infidelity $1-F$ (Monte Carlo)")
    ax.set_xlim(z[0], z[-1]); ax.set_ylim(z[0], z[-1]); ax.set_aspect("equal")
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=FS_SMALL, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    thin(fig); fig.savefig(os.path.join(args.out, "fig_val2.pdf"), **SAVE)
    plt.close()

# ---------------------------------------------------------------- Fig: cost against the number of blocks
def Ncost(G, E, eps):  # T gates to reach eps: cost per step x number of steps (theta <= 1)
    return G * max(TAU, TAU**2 * E / eps)


fig, ax = plt.subplots(2, 2, figsize=(W, 5.0), sharex="col", sharey="row")
for col, (r, ttl) in enumerate(cases):
    L = r["L"]
    for row, eps in enumerate((1e-1, 1e-2)):
        A = ax[row, col]
        for m, mk in (("random", "v"), ("oracle", "P"), ("graph", "o")):
            ks = ks_of(r, m)
            y = [np.mean([Ncost(p["G_T"], p["Eb"], eps) for p in r["plans"][m] if p["k"] == k]) for k in ks]
            A.plot(ks, y, color=C[m], marker=mk, ls=DASH[m], zorder=5 if m == "graph" else 3)
        Nq = Ncost(r["G1_T"], r["E1"], eps)
        Nf = (TAU + np.log(1 / eps)) * r["GamL_T"]
        A.plot([L], [Nq], "s", color=C["qDRIFT"], mfc="white", zorder=6)
        A.plot([1], [Nf], "D", color=C["full"], zorder=6)
        A.axhline(Nf, color=C["full"], lw=1, ls="--", zorder=1)
        A.axhline(Nq, color=C["qDRIFT"], lw=1, ls="--", zorder=1)
        A.set_xscale("log"); A.set_yscale("log")
        A.xaxis.set_major_locator(LogLocator(base=10, numticks=5)); A.xaxis.set_minor_locator(NullLocator())
        A.yaxis.set_major_locator(LogLocator(base=10, numticks=5)); A.yaxis.set_minor_locator(NullLocator())
        if row == 0:
            A.set_title(ttl, loc="left")
        if row == 1:
            A.set_xlabel("number of blocks $k$  (block size $L/k$)")
ax[0, 0].set_ylabel(r"T gates, $\varepsilon=0.1$"); ax[1, 0].set_ylabel(r"T gates, $\varepsilon=10^{-2}$")
hs = [Line2D([], [], color=C["full"], marker="D", ls="--", lw=1, label="full $H$ ($k=1$)"),
      Line2D([], [], color=C["qDRIFT"], marker="s", mfc="white", ls="--", lw=1, label="qDRIFT ($k=L$)"),
      Line2D([], [], color=C["random"], marker="v", ls=DASH["random"], label="random blocks"),
      Line2D([], [], color=C["graph"], marker="o", ls=DASH["graph"], label="graph blocks ($t=0$)"),
      Line2D([], [], color=C["oracle"], marker="P", ls=DASH["oracle"], label="oracle blocks")]
fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(args.out, "fig_blocks.pdf"), **SAVE)
plt.close()

# ---------------------------------------------------------------- Fig: block size against the number of terms (model B, instance 0)
from matplotlib.ticker import FixedLocator, FixedFormatter

EPS = 1e-2
N1 = lambda G, E: Ncost(G, E, EPS)


def plan(r, m, k):
    ps = [p for p in r["plans"][m] if p["k"] == k]
    return np.mean([p["G_T"] for p in ps]), np.mean([p["Eb"] for p in ps])


B0 = [BSER[L][0] for L in Ls]
fig, ax = plt.subplots(1, 2, figsize=(W, 3.5))
cm = plt.get_cmap("viridis")
for i, r in enumerate(B0):
    L = r["L"]; col = cm(i / (len(B0) - 1) * 0.9)
    for m, ls, mk in (("graph", "-", "o"), ("random", "--", "v")):
        ks = sorted(ks_of(r, m), reverse=True)
        ax[0].plot([L / k for k in ks], [L / k * plan(r, m, k)[1] / r["E1"] for k in ks], ls=ls, marker=mk, color=col, markevery=2)
ax[0].axhline(1, color="k", lw=0.8, ls=":")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("block size $m=L/k$"); ax[0].set_ylabel(r"design factor $D=m\bar E/\bar E_1$")
ax[0].set_title("(a) error is $\\bar E=\\bar E_1D/m$", loc="left")
step = max(1, len(B0) // 6)
ax[0].legend(handles=[Line2D([], [], color=cm(i / (len(B0) - 1) * 0.9), lw=2, label=f"$L={r['L']}$") for i, r in list(enumerate(B0))[::step]]
             + [Line2D([], [], color="0.3", marker="o", label="graph"), Line2D([], [], color="0.3", ls="--", marker="v", label="random")],
             ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=FS_SMALL, columnspacing=0.8, handlelength=1.4)


def nearest(r, m=None, k=None):
    return min(ks_of(r), key=lambda q: abs(np.log(q / (k if k else r["L"] / m))))


Lsx = [r["L"] for r in B0]
series = [("qDRIFT", [N1(r["G1_T"], r["E1"]) for r in B0], dict(color=C["qDRIFT"], marker="s", mfc="white", ls=(0, (5, 2)))),
          ("full $H$", [(TAU + np.log(1 / EPS)) * r["GamL_T"] for r in B0], dict(color=C["full"], marker="D")),
          ("graph, fixed $m=16$", [N1(*plan(r, "graph", nearest(r, m=16))) for r in B0], dict(color=C["graph"], marker="o")),
          ("graph, fixed $k=32$", [N1(*plan(r, "graph", nearest(r, k=32))) for r in B0], dict(color=C["graph"], marker="^", ls=(0, (5, 2)))),
          ("graph, fixed $k=2$", [N1(*plan(r, "graph", 2)) for r in B0], dict(color=C["graph"], marker="v", ls=(0, (1.2, 1.5)))),
          ("graph, best $k$", [min(N1(*plan(r, "graph", k)) for k in ks_of(r)) for r in B0], dict(color=C["oracle"], marker="P", ls=(0, (6, 2, 1.5, 2))))]
for lab, y, st in series:
    ax[1].plot(Lsx, y, label=lab, **st)
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
tk = [t for t in Lsx if t in (16, 64, 256, 1024, 4096, 16384)]
ax[1].xaxis.set_major_locator(FixedLocator(tk)); ax[1].xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk])); ax[1].xaxis.set_minor_locator(NullLocator())
ax[1].set_xlabel(f"number of terms $L$ (model B, $n={NQ}$)"); ax[1].set_ylabel(r"T gates, $\varepsilon=10^{-2}$")
ax[1].set_title("(b) block size against $L$", loc="left")
ax[1].legend(fontsize=FS_SMALL, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, columnspacing=0.8, handlelength=1.6)
for A in ax:
    A.yaxis.set_major_locator(LogLocator(base=10, numticks=5)); A.yaxis.set_minor_locator(NullLocator())
ax[0].xaxis.set_major_locator(LogLocator(base=10, numticks=5)); ax[0].xaxis.set_minor_locator(NullLocator())
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_regimes.pdf"), **SAVE)
plt.close()

print("figures written to", args.out)
