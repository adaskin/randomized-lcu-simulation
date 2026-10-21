"""Every number quoted in the paper's numerical section, from the saved grid runs (no simulation).

python paper_numbers.py [--root runs] [--n 15]      prints sections; redirect to a file to diff against the paper

Sections: cost (per Hamiltonian: cost ratios, Phi, best block size, crossover accuracy), breakeven (L* from Phi at several L,
and direct crossings), design factor D(m, L), fixed block size vs fixed k, n-dependence, robustness (initial states, instances),
control planners, validation, graph vs oracle, noise, side experiments.
"""

import argparse

import numpy as np

from analysis import (
    Lstar,
    N_eps,
    TAU,
    cands,
    cross,
    eps_cross,
    eps_noisy,
    grid_cases,
    kmean,
    ks_of,
    phi_best,
    study_tags,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs")
ap.add_argument("--n", type=int, default=15, help="qubit number of the main systems")
args = ap.parse_args()
by = grid_cases(root=args.root)
V = list(grid_cases("validate", args.root))
NQ = args.n
EPS = (1e-1, 1e-2, 1e-3)
isneel = lambda r: r["init"]["kind"] == "neel"
Bs = study_tags(by, "B", NQ)
Ls = sorted(Bs)
B0 = [Bs[L][0] for L in Ls]


def line(r):
    pg, kg = phi_best(r, "graph")
    po, _ = phi_best(r, "oracle")
    p1 = r["G1_T"] * r["E1"]
    row = f"{r['tag']:26s} L={r['L']:6d} Phig/Phi1={pg / p1:.3f} (oracle {po / p1:.3f}) k={kg} m={r['L'] / kg:.0f}"
    for e in EPS[:2]:
        g, q, f = (N_eps(r, e, m)[0] for m in ("graph", "qDRIFT", "full"))
        row += f" | eps={e:g}: g/q={g / q:.2f} f/g={f / g:.2f}"
    x = eps_cross(r)
    row += f" | graph<full for eps>{x:.3g}" if x else " | full H cheapest"
    return row, pg, p1, po


print("== cost: per record (T-gate model R=40, kappa=2, c=5, tau=100)")
rows = {}
for tag in sorted(
    [t for t in by if isneel(by[t])], key=lambda t: (t[0], by[t]["n"], by[t]["L"], t)
):
    s, pg, p1, po = line(by[tag])
    rows[tag] = (pg, p1, po)
    print(s)

print(
    f"\n== breakeven: L* (graph, oracle, qDRIFT) from Phi measured at each B n={NQ} s=0 record"
)
for r in B0:
    pg, p1, po = rows[r["tag"]]
    print(
        f"{r['tag']:26s} "
        + "  ".join(
            f"eps={e:g}: g {Lstar(pg, e):.3g} o {Lstar(po, e):.3g} q {Lstar(p1, e):.3g}"
            for e in EPS
        )
    )

print(
    f"\n== measured crossing of graph blocks and full H in L (mean over instances), B n={NQ}"
)
for e in EPS:
    mean = lambda m: [np.mean([N_eps(x, e, m)[0] for x in Bs[L]]) for L in Ls]
    xc = cross(Ls, mean("graph"), mean("full"))
    xq = cross(Ls, mean("qDRIFT"), mean("full"))
    print(
        f"eps={e:g}: graph/full crossing L={xc if xc is None else round(xc)}  qDRIFT/full crossing L={xq if xq is None else round(xq)}"
    )

print(f"\n== design factor D = m Ebar/E1, graph blocks (B n={NQ}, s=0)")
for r in B0:
    out = [
        f"m={r['L'] / k:.0f}:{r['L'] / k * kmean(r, 'graph', k) / r['E1']:.2f}"
        for k in ks_of(r)
    ]
    print(f"{r['tag']:26s} E1bar={r['E1']:.3f}  " + " ".join(out))

print("\n== fixed block size vs fixed k (eps=1e-2, graph blocks): T gates")
Nc = lambda G, E, e=1e-2: G * max(TAU, TAU**2 * E / e)


def plan(r, k):
    ps = [p for p in r["plans"]["graph"] if p["k"] == k]
    return np.mean([p["G_T"] for p in ps]), np.mean([p["Eb"] for p in ps])


near = lambda r, m: min(ks_of(r), key=lambda q: abs(np.log(q / (r["L"] / m))))
for r in B0:
    ks = ks_of(r)
    best = min(ks, key=lambda k: Nc(*plan(r, k)))
    print(
        f"L={r['L']:6d} qDRIFT {Nc(r['G1_T'], r['E1']):.3g} full {(TAU + np.log(100)) * r['GamL_T']:.3g} fixed m=16 (k={near(r, 16)}) {Nc(*plan(r, near(r, 16))):.3g} "
        f"k=2 {Nc(*plan(r, 2)):.3g} best k={best} (m={r['L'] / best:.0f}) {Nc(*plan(r, best)):.3g}"
    )

print("\n== exponents: fixed k=2 cost ~ L^a, fixed m=16 cost ~ L^a")
for lab, f in (
    ("k=2", lambda r: Nc(*plan(r, 2))),
    ("m=16", lambda r: Nc(*plan(r, near(r, 16)))),
):
    for a, b in ((Ls[-3], Ls[-1]), (Ls[-5], Ls[-1])):
        ra, rb = Bs[a][0], Bs[b][0]
        print(f"{lab} L {a}->{b}: exponent {np.log(f(rb) / f(ra)) / np.log(b / a):.2f}")

print(
    "\n== n-dependence: Phi_g/Phi_1, best k (m) and eps=1e-2 costs at fixed L, over n"
)
for L in sorted(
    {
        r["L"]
        for r in by.values()
        if r["tag"].startswith("B") and isneel(r) and r["hamiltonian"]["seed"] == 0
    }
):
    out = []
    for n in (5, 10, 15, 20):
        t = f"B n={n} L={L} s=0"
        if t in by:
            r = by[t]
            pg, kg = phi_best(r, "graph")
            out.append(f"n={n}: {pg / (r['G1_T'] * r['E1']):.2f} (m={L / kg:.0f})")
    if len(out) > 1:
        print(f"L={L:6d} " + "  ".join(out))
print(
    "model A:",
    "  ".join(
        f"n={n} (L={by[f'A n={n}']['L']}): {rows[f'A n={n}'][0] / rows[f'A n={n}'][1]:.2f}"
        for n in (5, 10, 15, 20)
        if f"A n={n}" in by
    ),
)

print("\n== robustness: initial states (n=10, 15) and instances")
for tag in sorted(t for t in by if "init=" in t):
    base = tag.split(" init=")[0]
    if base in by or base + " s=0" in by:
        b = by[base] if base in by else by[base + " s=0"]
        r = by[tag]
        print(
            f"{tag:34s} Phig/Phi1={phi_best(r, 'graph')[0] / (r['G1_T'] * r['E1']):.3f} (neel {phi_best(b, 'graph')[0] / (b['G1_T'] * b['E1']):.3f}); "
            f"g/q at 1e-2 {N_eps(r, 1e-2, 'graph')[0] / N_eps(r, 1e-2, 'qDRIFT')[0]:.3f} (neel {N_eps(b, 1e-2, 'graph')[0] / N_eps(b, 1e-2, 'qDRIFT')[0]:.3f})"
        )
for n in (5, 10, 15, 20):
    sp = []
    for L, rs in sorted(study_tags(by, "B", n).items()):
        if len(rs) > 1:
            x = [
                phi_best(r, "graph")[0] / (r["G1_T"] * r["E1"]) for r in rs if isneel(r)
            ]
            sp.append(f"L={L}: {np.mean(x):.3f}+-{np.std(x):.3f}")
    if sp:
        print(f"instance spread of Phig/Phi1, n={n}: " + "; ".join(sp))

print(
    "\n== control planners: E0 / Ebar of the swap heuristic on the graph, its diagonal, the identity; random; oracle Ebar"
)
for tag in sorted(
    t
    for t in by
    if t.startswith(("A n", "B n"))
    and isneel(by[t])
    and (t.startswith("A") or " s=0" in t)
    and by[t]["n"] == NQ
):
    r = by[tag]
    print(tag, f"L={r['L']}  qDRIFT E1_0={r['E1_0']:.3f} E1bar={r['E1']:.3f}")
    for k in ks_of(r):
        g = lambda m: (kmean(r, m, k, "E0"), kmean(r, m, k))
        print(
            f"   k={k:5d} "
            + "  ".join(
                f"{m} {g(m)[0]:.3f}/{g(m)[1]:.3f}"
                for m in ("graph", "variances", "weights", "random")
            )
            + f"  oracle {g('oracle')[1]:.3f}"
        )

print(
    "\n== validation of the error formula: MC / predicted, entries with predicted < 0.1"
)
for tau in (8, 100):
    for n in (5, 10, 15):
        sel = [
            v
            for v in V
            if f"tau={tau}" == v["tag"].split()[-1]
            and f"n={n} " in v["tag"]
            and v["pred"] < 0.1
        ]
        if sel:
            q = np.array([v["mc"] / v["pred"] for v in sel])
            print(
                f"tau={tau} n={n}: {len(sel)} settings, MC/pred in [{q.min():.2f}, {q.max():.2f}], mean {q.mean():.2f}"
            )
if V:
    q = np.array([v["mc"] / v["pred"] for v in V if v["pred"] < 0.1])
    z = np.array([(v["mc"] - v["pred"]) / v["se"] for v in V if v["pred"] < 0.1])
    print(
        f"all: {len(q)} settings with pred<0.1, MC/pred in [{q.min():.2f}, {q.max():.2f}]; {np.sum(np.abs(z) < 2)} within 2 standard errors"
    )
    big = [v for v in V if v["pred"] >= 0.1]
    if big:
        print(
            "pred>=0.1:",
            len(big),
            "settings, MC/pred in [%.2f, %.2f]"
            % (
                min(v["mc"] / v["pred"] for v in big),
                max(v["mc"] / v["pred"] for v in big),
            ),
        )

print(
    "\n== graph vs oracle and random vs graph, over all records and block counts (same k)"
)
rat, rr, rr8 = [], [], []
for r in by.values():
    for k in ks_of(r):
        g, o = kmean(r, "graph", k), kmean(r, "oracle", k)
        rat.append(g / o)
        if "random" in r["plans"] and [p for p in r["plans"]["random"] if p["k"] == k]:
            x = kmean(r, "random", k) / g
            rr.append(x)
            if r["L"] / k >= 8:
                rr8.append(x)
print(
    f"graph/oracle Ebar over {len(rat)} (record, k): median {np.median(rat):.2f}, mean {np.mean(rat):.2f}, p90 {np.percentile(rat, 90):.2f}, max {np.max(rat):.2f}"
)
print(
    f"random/graph Ebar: median {np.median(rr):.1f}; blocks of >=8 terms: median {np.median(rr8):.1f}"
)

print(f"\n== noise: largest error rate p at which infidelity 1e-2 is reachable")
ps = np.logspace(-14, -5, 1800)
for tag in [f"A n={NQ}"] + [f"B n={NQ} L={L} s=0" for L in Ls[-3:]]:
    r = by[tag]
    out = []
    for m in ("full", "graph", "qDRIFT"):
        ok = [p for p in ps if min(eps_noisy(r, p, m)[0], 1) <= 1e-2]
        out.append(f"{m} {max(ok):.1e}" if ok else f"{m} none")
    print(f"{tag:30s}", "  ".join(out))
for m in ("full", "graph"):
    f = lambda r: max(
        [p for p in ps if min(eps_noisy(r, p, m)[0], 1) <= 1e-2] or [np.nan]
    )
    y = [f(r) for r in B0[-6:]]
    print(
        m,
        "largest tolerable p by L:",
        " ".join(f"{r['L']}:{v:.1e}" for r, v in zip(B0[-6:], y)),
    )
print("noise crossing (graph vs full) in L from the largest tolerable p:", end=" ")
fg = [
    max([p for p in ps if min(eps_noisy(r, p, "graph")[0], 1) <= 1e-2] or [np.nan])
    for r in B0
]
ff = [
    max([p for p in ps if min(eps_noisy(r, p, "full")[0], 1) <= 1e-2] or [np.nan])
    for r in B0
]
print(cross(Ls, fg, ff))

print(
    "\n== weights-only planner as a competitor: Phi/Phi_1 (plan chosen on its own t=0 error) and break-even L*"
)
for tag in sorted(
    t
    for t in by
    if isneel(by[t]) and by[t]["n"] in (10, NQ) and (t.startswith("A") or " s=0" in t)
):
    r = by[tag]
    p1 = r["G1_T"] * r["E1"]
    out = []
    for m in ("graph", "weights", "oracle"):
        pg, kk = phi_best(r, m)
        out.append(f"{m} {pg / p1:.3f} (k={kk})")
    pw, _ = phi_best(r, "weights")
    print(f"{tag:26s}", "  ".join(out), f" L*(eps=1e-2) weights {Lstar(pw, 1e-2):.3g}")

print("\n== side experiments")
for t in ("planners", "pairs", "sampling", "freeblocks"):
    d = grid_cases(t, args.root)
    print(f"{t}: {len(d)} cases:", ", ".join(sorted(d)))
for tag, d in sorted(grid_cases("pairs", args.root).items()):
    p = d["results"]
    sw, ma = p["pairs: swaps"]["E0"], p["pairs: matching"]["E0"]
    q, cm, cr = (
        p["qDRIFT"],
        p["commuting pairs: matching"],
        p["commuting pairs: random"],
    )
    print(
        f"pairs {tag}: swaps/exact {sw / ma:.4f}; commuting fraction {p['commuting_fraction_of_pairs']:.2f}; N commuting/qDRIFT {cm['N'] / q['N']:.3f}, random commuting/qDRIFT {cr['N'] / q['N']:.3f}; "
        f"paired weight {cm['frac_paired']:.2f}; D0 {2 * cm['E0'] / q['E0']:.2f} Dbar {2 * cm['Eb'] / q['Eb']:.2f}"
    )

print(
    "\n== numbers quoted in the text of the control, block-size and sampling paragraphs"
)
for n in (10, NQ):
    Bn = study_tags(by, "B", n)
    Ln = sorted(Bn)
    for e in EPS[:2]:
        mean = lambda m: [np.mean([N_eps(x, e, m)[0] for x in Bn[L]]) for L in Ln]
        xc = cross(Ln, mean("graph"), mean("full"))
        print(
            f"n={n}: measured graph/full crossing at eps={e:g}: {xc if xc is None else round(xc)}; N=2^n={2**n}; estimated L* (graph, largest L) {Lstar(phi_best(Bn[Ln[-1]][0], 'graph')[0], e):.3g}"
        )
ratio_w, ratio_o = {"A": [], "B": []}, {"A": [], "B": []}
for tag, r in by.items():
    if (
        not isneel(r)
        or r["n"] not in (10, 15)
        or (tag[0] == "B" and " s=0" not in tag)
        or "weights" not in r["plans"]
    ):
        continue
    for k in ks_of(r):
        ratio_w[tag[0]].append(
            kmean(r, "weights", k, "E0") / max(kmean(r, "graph", k, "E0"), 1e-12)
        )
        ratio_o[tag[0]].append(1 - kmean(r, "oracle", k) / kmean(r, "weights", k))
for m in "AB":
    print(
        f"model {m}: weights/graph error at t=0: max {max(ratio_w[m]):.2f}; oracle gain over weights-only on average: max {max(ratio_o[m]):.2f}, min {min(ratio_o[m]):.2f}"
    )
print(
    "random-block design factor, max over records:",
    max(
        r["L"] / k * kmean(r, "random", k) / r["E1"]
        for r in by.values()
        if r["L"] >= 256 and "random" in r["plans"]
        for k in ks_of(r)
    ),
)
print("graph design factor slope D ~ m^a for model B n=15 L=32768, m=2..256:", end=" ")
r = Bs[Ls[-1]][0]
ms_ = np.array([r["L"] / k for k in ks_of(r) if r["L"] / k <= 256])
Ds = np.array(
    [r["L"] / k * kmean(r, "graph", k) / r["E1"] for k in ks_of(r) if r["L"] / k <= 256]
)
a = np.polyfit(np.log(ms_), np.log(Ds), 1)[0]
print(f"a={a:.2f} (error ~ m^{-(1 - a):.2f})")
print(
    "noise: p_full(L) * L at the largest L, p_graph; extrapolated crossing L = p_full*L / p_graph:",
    end=" ",
)
r = B0[-1]
pf = max(p for p in ps if min(eps_noisy(r, p, "full")[0], 1) <= 1e-2)
pgq = max(p for p in ps if min(eps_noisy(r, p, "graph")[0], 1) <= 1e-2)
print(f"{pf * r['L'] / pgq:.3g}")
for tag, d in sorted(grid_cases("freeblocks", args.root).items()):
    res = d["results"]
    ws = [
        res[x]["weight_single"]
        for x in res
        if x.split("|")[1] == "clw" and res[x]["n_single"] > 0
    ]
    if ws:
        print(
            f"freeblocks {tag}: weight-balanced dealing isolates single terms carrying {min(ws):.2f}..{max(ws):.2f} of the weight"
        )
    for pl in ("graph", "weights"):
        for ru in ("alpha", "hs"):
            eq = min(res[x][f"N_{ru}"] for x in res if x.startswith(f"{pl}|equal|"))
            fr = min(res[x][f"N_{ru}"] for x in res if x.startswith(f"{pl}|free|"))
            sz = [res[x]["sizes"] for x in res if x.startswith(f"{pl}|free|")]
            print(
                f"freeblocks {tag}: plan {pl} {ru}: free/equal best cost {fr / eq:.3f}; free sizes min/max over k: {min(s[0] for s in sz)}..{max(s[2] for s in sz)}"
            )
print("sampling: HS vs alpha, hindsight vs HS, break-even with weights+HS")
for tag, d in sorted(grid_cases("sampling", args.root).items()):
    res, Lr = d["results"], d["L"]
    P1 = res[f"qDRIFT|{Lr}|alpha"]["Phi"]
    for pl in ("graph", "weights", "oracle"):
        b = {
            ru: min(
                res[x]["Phi"]
                for x in res
                if x.startswith(pl + "|") and x.endswith("|" + ru)
            )
            for ru in ("alpha", "hs", "var0", "varbar")
        }
        print(
            f"{tag:22s} {pl:8s} HS/alpha {b['hs'] / b['alpha']:.3f}  varbar/HS {b['varbar'] / b['hs']:.3f}  var0/alpha {b['var0'] / b['alpha']:.2f}   L*(1e-2) with HS {Lstar(b['hs'] * (1.0), 1e-2):.3g}"
        )
    print(f"{tag:22s} qDRIFT var0/alpha {res[f'qDRIFT|{Lr}|var0']['Phi'] / P1:.2f}")
