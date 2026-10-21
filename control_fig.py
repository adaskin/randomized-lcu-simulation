"""Control figure: error per step of groupings planned on the graph of the initial state or on the oracle graph, relative to the
weights-only grouping, at t=0 and averaged along the trajectory (the information of tables/table_control.tex).

python control_fig.py [--root runs] [--out paper] [--n 15]"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from analysis import grid_cases, kmean, ks_of
from figstyle import W, FS_SMALL, C, DASH, apply

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--out", default="figs")
ap.add_argument("--n", type=int, default=15)
args = ap.parse_args()
apply()
by = grid_cases(root=args.root)
N = args.n
SYS = [
    ("model A, $n=%d$, $L=345$" % N, "A n=%d" % N),
    ("model B, $n=%d$, $L=1024$" % N, "B n=%d L=1024 s=0" % N),
    ("model B, $n=%d$, $L=32768$" % N, "B n=%d L=32768 s=0" % N),
]
fig, axs = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True)
for ax, (title, tag) in zip(axs, SYS):
    r = by[tag]
    ks = sorted(
        {
            min(ks_of(r), key=lambda q: abs(np.log(q / (r["L"] / m))))
            for m in (64, 16, 4)
            if m <= r["L"] / 2
        }
    )
    ms = [r["L"] / k for k in ks]
    g0 = [kmean(r, "graph", k, "E0") / kmean(r, "weights", k, "E0") for k in ks]
    gb = [kmean(r, "graph", k) / kmean(r, "weights", k) for k in ks]
    ob = [kmean(r, "oracle", k) / kmean(r, "weights", k) for k in ks]
    x = np.arange(len(ks))
    ax.axhline(1, color=C["weights"], ls=DASH["weights"], lw=1.6)
    ax.plot(
        x,
        g0,
        color=C["graph"],
        ls="-",
        marker="o",
        mfc="white",
        label=r"graph, at $t=0$",
    )
    ax.plot(
        x,
        gb,
        color=C["graph"],
        ls=DASH["qDRIFT"],
        marker="o",
        label="graph, trajectory average",
    )
    ax.plot(
        x,
        ob,
        color=C["oracle"],
        ls=DASH["oracle"],
        marker="s",
        label="oracle, trajectory average",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:.0f}" for m in ms])
    ax.set_xlim(-0.35, len(ks) - 0.65)
    ax.set_xlabel("block size $m$")
    ax.set_title(title, fontsize=8.5)
    ax.text(
        0.97,
        0.04,
        "weights only = 1",
        color=C["weights"],
        transform=ax.transAxes,
        ha="right",
        fontsize=FS_SMALL,
    )
axs[0].set_ylabel("error per step / weights only")
axs[0].set_ylim(0.3, 1.62)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout()
os.makedirs(args.out, exist_ok=True)
fig.savefig(os.path.join(args.out, "fig_control.pdf"), bbox_inches="tight")
print("wrote fig_control.pdf")
