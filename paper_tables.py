"""Writes the data tables of the paper (tables/table_*.tex) from the saved grid runs; no simulation.

    python paper_tables.py [--root runs] [--dir tables] [--n 15]

table_planners    alternative ways of choosing the grouping, E at t=0 (task planners)
table_cost        T-gate cost to reach 0.1 and 0.01 for the main systems
table_ndep        cost against n and L (best graph grouping, crossover with full H)
table_weights     graph vs weights-only vs oracle groupings: cost-error product and break-even L*
table_control     error per step of groupings planned on different information (t=0 / trajectory average)
table_breakeven   break-even number of terms against full H for three accuracies, estimated and measured
table_pairs       blocks of two terms: exact matching, swaps, commuting pairs (task pairs)
table_sampling    block probabilities (task sampling)
table_freeblocks  unequal block sizes (task freeblocks)
"""

import argparse
import os

import numpy as np

from analysis import (
    Lstar,
    N_eps,
    cross,
    eps_cross,
    grid_cases,
    kmean,
    ks_of,
    phi_best,
    study_tags,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--dir", default="tables")
ap.add_argument("--n", type=int, default=15, help="qubit number of the main systems")
args = ap.parse_args()
os.makedirs(args.dir, exist_ok=True)
by = grid_cases(root=args.root)
NQ = args.n
WIDE = r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
TAB_END = "\n\\bottomrule\n\\end{tabular*}\n"


def sci(x):
    e = int(np.floor(np.log10(x)))
    return rf"${x / 10**e:.1f}\times10^{{{e}}}$"


def write(name, tex):
    open(f"{args.dir}/{name}", "w").write(tex)
    print(f"--- {name}\n{tex}")


def rec(model, n, L=None, s=0):
    return by["A n=%d" % n] if model == "A" else by["B n=%d L=%d s=%d" % (n, L, s)]


def have(model, n, L=None):
    return ("A n=%d" % n if model == "A" else "B n=%d L=%d s=0" % (n, L)) in by


_SIDE = {}


def side(task, model, n, L=None):
    tag = "A n=%d" % n if model == "A" else "B n=%d L=%d s=0" % (n, L)
    if task not in _SIDE:
        _SIDE[task] = grid_cases(task, args.root)
    return _SIDE[task].get(tag)


def label(model, n, L):
    return (
        rf"model A, $n={n}$, $L={L}$" if model == "A" else rf"model B, $n={n}$, $L={L}$"
    )


# ---------- alternative planners
rows = [
    ("qDRIFT ($k=L$)", None),
    ("random blocks", "random"),
    ("cluster, then deal", "cluster+deal"),
    ("sort-and-deal (one direction)", "svd-snake"),
    ("greedy (all directions)", "greedy"),
    ("swaps (Algorithm~\\ref{alg:anti})", "swap"),
    ("cluster, then deal, then swaps", "cluster+deal -> swap"),
    ("greedy, then swaps", "greedy -> swap"),
    ("full LCU ($k=1$)", 0),
]
cols = [("A", NQ, None, 16), ("B", NQ, 1024, 8), ("B", NQ, 1024, 16)]
data = []
for model, n, L, m in cols:
    d = side("planners", model, n, L)
    r = d["results"]
    data.append(min(r, key=lambda x: abs(np.log(x["m"] / m))) if d else None)
if all(data):
    head = " & ".join(
        rf"{'A' if mo == 'A' else 'B'}, $L={dd['L']}$, $k={x['k']}$"
        for (mo, n, L, m), x, dd in zip(
            cols, data, [side("planners", c[0], c[1], c[2]) for c in cols]
        )
    )
    lines = []
    for name, key in rows:
        cells = []
        for x in data:
            v = x["E1"] if key is None else 0.0 if key == 0 else x[key]
            cells.append(f"{v:.3f}" if key != 0 else "0")
        best = key == "swap"
        lines.append(
            name
            + " & "
            + " & ".join(rf"\textbf{{{c}}}" if best else c for c in cells)
            + r"\\"
        )
    write(
        "table_planners.tex",
        WIDE
        + "lccc}\n\\toprule\ngrouping & "
        + head
        + "\\\\\n\\midrule\n"
        + "\n".join(lines)
        + TAB_END,
    )

# ---------- cost of the main systems
sysl = [("A", NQ, None)] + [
    ("B", NQ, L) for L in (2048, 8192, 32768) if have("B", NQ, L)
]
lines = []
for model, n, L in sysl:
    r = rec(model, n, L)
    for j, e in enumerate((0.1, 0.01)):
        v = {m: N_eps(r, e, m) for m in ("qDRIFT", "full", "random", "graph", "oracle")}
        practical = {m: v[m][0] for m in ("qDRIFT", "full", "random", "graph")}
        best = min(practical, key=practical.get)

        def f(m, k=False):
            s = f"{v[m][0] / 1e6:.2f}" if v[m][0] < 1e7 else f"{v[m][0] / 1e6:.1f}"
            s = rf"\textbf{{{s}}}" if m == best else s
            return s + (f" ({v[m][1]})" if k else "")

        lead = f"{model} & {r['L']}" if j == 0 else "&"
        lines.append(
            f"{lead} & {e:g} & {f('qDRIFT')} & {f('full')} & {f('random')} & {f('graph', True)} & {f('oracle', True)}"
            + r"\\"
        )
write(
    "table_cost.tex",
    WIDE + r"""lrrccccc}
\toprule
model & $L$ & $\varepsilon$ & qDRIFT & full $H$ & random & graph ($k$) & oracle ($k$)\\
\midrule
""" + "\n".join(lines) + TAB_END,
)

# ---------- n- and L-dependence
ROWS = (
    [("A", n, None) for n in (5, 10, 15, 20)]
    + [("B", n, 256) for n in (5, 10, 15, 20)]
    + [("B", n, 1024) for n in (10, 15, 20)]
    + [("B", n, 8192) for n in (10, 15)]
    + [("B", 15, 32768)]
)
ROWS = [x for x in ROWS if have(*x)]
lines = []
for model, n, L in ROWS:
    r = rec(model, n, L)
    pg, kg = phi_best(r, "graph")
    p1 = r["G1_T"] * r["E1"]
    x = eps_cross(r)
    lines.append(
        f"{model} & {n} & {r['L']} & {kg} ({r['L'] / kg:.0f}) & {N_eps(r, 1e-2, 'graph')[0] / 1e6:.1f} & "
        f"{N_eps(r, 1e-2, 'qDRIFT')[0] / 1e6:.1f} & {N_eps(r, 1e-2, 'full')[0] / 1e6:.2f} & {pg / p1:.2f} & "
        + (f"{x:.3f}" if x else "never")
        + r" \\"
    )
write(
    "table_ndep.tex",
    WIDE + r"""llrrccccc}
\toprule
& & & & \multicolumn{3}{c}{T gates ($\times10^6$), $\varepsilon=10^{-2}$} & & \\
\cmidrule(lr){5-7}
model & $n$ & $L$ & $k$ ($m$) & graph & qDRIFT & full $H$ & $\Phi_g/\Phi_1$ & graph cheapest for $\varepsilon>$\\
\midrule
""" + "\n".join(lines) + TAB_END,
)

# ---------- graph vs weights-only vs oracle
lines = []
for model, n, L in [x for x in ROWS if x[1] != 5]:
    r = rec(model, n, L)
    p1 = r["G1_T"] * r["E1"]
    v = {m: phi_best(r, m)[0] for m in ("graph", "weights", "oracle")}
    lines.append(
        f"{model} & {n} & {r['L']} & "
        + " & ".join(f"{v[m] / p1:.2f}" for m in ("graph", "weights", "oracle"))
        + " & "
        + " & ".join(sci(Lstar(v[m], 1e-2)) for m in ("graph", "weights"))
        + r" \\"
    )
write(
    "table_weights.tex",
    WIDE + r"""llrccccc}
\toprule
model & $n$ & $L$ & $\Phi_g/\Phi_1$ & $\Phi_w/\Phi_1$ & $\Phi_o/\Phi_1$ & $L^\ast$ graph & $L^\ast$ weights only\\
\midrule
""" + "\n".join(lines) + TAB_END,
)

# ---------- break-even from the largest instance, estimated and measured
Bs = study_tags(by, "B", NQ)
Ls = sorted(Bs)
r = Bs[Ls[-1]][0]
p1 = r["G1_T"] * r["E1"]
pg, po = phi_best(r, "graph")[0], phi_best(r, "oracle")[0]
lines = []
for e in (1e-1, 1e-2, 1e-3):
    mean = lambda m: [np.mean([N_eps(x, e, m)[0] for x in Bs[L]]) for L in Ls]
    xc = cross(Ls, mean("graph"), mean("full"))
    meas = sci(xc) if xc else rf"beyond ${Ls[-1]}$"
    lines.append(
        f"$10^{{{int(np.log10(e))}}}$ & "
        + " & ".join(sci(Lstar(p, e)) for p in (p1, pg, po))
        + f" & {meas}"
        + r" \\"
    )
write(
    "table_breakeven.tex",
    WIDE + r"""lcccc}
\toprule
$\varepsilon$ & qDRIFT & graph blocks & oracle blocks & measured crossing (graph)\\
\midrule
""" + "\n".join(lines) + TAB_END,
)

# ---------- control: groupings planned on different information
groups = []
CTRL = [("A", NQ, None), ("B", NQ, 1024), ("B", NQ, 32768)]
for model, n, L in CTRL:
    if not have(model, n, L):
        continue
    r = rec(model, n, L)
    ks = sorted(
        {
            min(ks_of(r), key=lambda q: abs(np.log(q / (r["L"] / m))))
            for m in (64, 16, 4)
            if m <= r["L"] / 2
        }
    )
    rr = []
    for k in ks:
        cell = [
            f"{kmean(r, m, k, 'E0'):.3f} / {kmean(r, m, k):.3f}"
            for m in ("graph", "variances", "weights", "random")
        ]
        rr.append(
            f"{k} ({r['L'] / k:.0f}) & "
            + " & ".join(cell)
            + f" & {kmean(r, 'oracle', k):.3f}"
        )
    groups.append((label(model, n, r["L"]), r["E1_0"], r["E1"], rr))
body = []
for lab, e0, eb, rr in groups:
    body.append(
        rf"\multicolumn{{6}}{{l}}{{\emph{{{lab}}}; qDRIFT: {e0:.3f} / {eb:.3f}}}\\"
    )
    body += [x + r"\\" for x in rr]
    body.append(r"\addlinespace")
write(
    "table_control.tex",
    WIDE + r"""rccccc}
\toprule
$k$ ($m$) & graph & variances only & weights only & random & oracle ($\bar E$)\\
\midrule
""" + "\n".join(body[:-1]) + TAB_END,
)

# ---------- blocks of two
lines = []
for model, n, L in [("A", 10, None), ("A", 15, None), ("B", 10, 256), ("B", 15, 256)]:
    d = side("pairs", model, n, L)
    if not d:
        continue
    p, Lr = d["results"], d["L"]
    q, cm = p["qDRIFT"], p["commuting pairs: matching"]
    D0, Db = 2 * cm["E0"] / q["E0"], 2 * cm["Eb"] / q["Eb"]
    lines.append(
        f"{model[0]}, $n={n}$, $L={Lr}$ & {p['pairs: matching']['E0']:.4f} & {p['pairs: swaps']['E0']:.4f} & {D0:.2f} & {Db:.2f} & "
        f"{q['N'] / 1e7:.2f} & {cm['N'] / 1e7:.2f} & {p['commuting pairs: random']['N'] / 1e7:.2f}"
        + r" \\"
    )
if lines:
    write(
        "table_pairs.tex",
        WIDE + r"""lccccccc}
\toprule
 & \multicolumn{2}{c}{$E$ at $t=0$} & \multicolumn{2}{c}{$D$, commuting pairs} & \multicolumn{3}{c}{T gates ($\times10^7$), $\varepsilon=10^{-2}$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
model & exact & swaps & $t=0$ & average & qDRIFT & graph pairs & random pairs\\
\midrule
""" + "\n".join(lines) + TAB_END,
    )

# ---------- block probabilities: alpha (ours) vs Hilbert-Schmidt vs variance-optimal at t=0 vs with hindsight
RULES = ("alpha", "hs", "var0", "varbar")
body = []
for model, n, L in [("A", NQ, None), ("B", NQ, 1024), ("B", NQ, 4096)]:
    d = side("sampling", model, n, L)
    if not d:
        continue
    res, Lr = d["results"], d["L"]
    P1 = res[f"qDRIFT|{Lr}|alpha"]["Phi"]
    body.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label(model, n, Lr)}}}}}\\")
    for pl, name in (
        ("graph", "graph blocks"),
        ("weights", "weights only"),
        ("random", "random blocks"),
        ("oracle", "oracle blocks"),
    ):
        cells = []
        for ru in RULES:
            v = {
                int(x.split("|")[1]): res[x]["Phi"] / P1
                for x in res
                if x.startswith(pl + "|") and x.endswith("|" + ru)
            }
            kb = min(v, key=v.get)
            cells.append(f"{v[kb]:.2f} ({kb})")
        body.append(f"{name} & " + " & ".join(cells) + r"\\")
    q = [1.0, 1.0] + [res[f"qDRIFT|{Lr}|{ru}"]["Phi"] / P1 for ru in ("var0", "varbar")]
    body.append("qDRIFT & " + " & ".join(f"{x:.2f}" for x in q) + r"\\")
    body.append(r"\addlinespace")
if body:
    write(
        "table_sampling.tex",
        WIDE + r"""lcccc}
\toprule
grouping & $p_c=\alpha_c$ & Hilbert--Schmidt & optimal at $t=0$ & optimal on average\\
\midrule
""" + "\n".join(body[:-1]) + TAB_END,
    )

# ---------- unequal block sizes: equal sizes (Algorithm 1) vs free sizes (moves on the predicted cost) vs weight-balanced dealing
body = []
for model, n, L in [("B", NQ, 1024), ("B", NQ, 4096)]:
    F = side("freeblocks", model, n, L)
    if not F:
        continue
    res, Nq = F["results"], F["Nq"]
    body.append(
        rf"\multicolumn{{5}}{{l}}{{\emph{{{label(model, n, F['L'])}}}; full $H$: {F['Nf'] / Nq:.2f}}}\\"
    )
    for mode, mname in (
        ("equal", "equal sizes (Algorithm~\\ref{alg:anti})"),
        ("free", "free sizes"),
        ("clw", "weight-balanced dealing"),
        ("clw+free", "weight-balanced, then free"),
    ):
        cells = []
        for pl in ("graph", "weights"):
            for ru in ("alpha", "hs"):
                v = {
                    x: res[x][f"N_{ru}"] / Nq
                    for x in res
                    if x.startswith(f"{pl}|{mode}|")
                }
                cells.append(f"{min(v.values()):.2f}")
        body.append(f"{mname} & " + " & ".join(cells) + r"\\")
    body.append(r"\addlinespace")
if body:
    write(
        "table_freeblocks.tex",
        WIDE + r"""lcccc}
\toprule
 & \multicolumn{2}{c}{planned on the graph} & \multicolumn{2}{c}{planned on the weights}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
block sizes & $p_c=\alpha_c$ & Hilbert--Schmidt & $p_c=\alpha_c$ & Hilbert--Schmidt\\
\midrule
""" + "\n".join(body[:-1]) + TAB_END,
    )
