import numpy as np, json

TH0 = 1.0


def cost_of(rec, pl, model):  # (Gamma, E0, Eb)
    return pl["G_" + model[0].lower() if model == "T" else "G_g"], pl["E0"], pl["Eb"]


def G(pl, m):
    return pl["G_T"] if m == "T" else pl["G_g"]


def cands(rec, method, m):
    L = rec["L"]
    q = dict(k=L, E0=rec["E1_0"], Eb=rec["E1"], G_T=rec["G1_T"], G_g=rec["G1_g"])
    return [p for p in rec["plans"][method] if p["k"] > 1] + [q]


def N_eps(rec, eps, method, m="T", sel="pred"):
    """total cost to reach infidelity eps (noiseless). method: qDRIFT, full, graph, oracle, random, magnitude"""
    tau = rec["tau"]
    rmin = tau / TH0
    if method == "full":
        return (tau + np.log(1 / eps)) * (
            rec["GamL_T"] if m == "T" else rec["GamL_g"]
        ), 1
    if method == "qDRIFT":
        pl = dict(
            k=rec["L"], E0=rec["E1_0"], Eb=rec["E1"], G_T=rec["G1_T"], G_g=rec["G1_g"]
        )
        return G(pl, m) * max(rmin, tau**2 * pl["Eb"] / eps), rec["L"]
    cs = cands(rec, method, m)

    def cost(pl, E):
        return G(pl, m) * max(rmin, tau**2 * E / eps)

    if method == "graph" or method == "magnitude":
        pick = min(cs, key=lambda p: cost(p, p["E0"]))  # chosen from the t=0 graph
    elif method == "random":  # generous: best k on true average, mean over draws
        ks = sorted({p["k"] for p in rec["plans"]["random"]})
        best = None
        for k in ks:
            ps = [p for p in rec["plans"]["random"] if p["k"] == k]
            c = np.mean([cost(p, p["Eb"]) for p in ps])
            if best is None or c < best[0]:
                best = (c, k)
        q = cost(cands(rec, "graph", m)[-1], rec["E1"])
        return min(best[0], q), best[1]
    else:
        pick = min(cs, key=lambda p: cost(p, p["Eb"]))  # oracle
    return cost(pick, pick["Eb"]), pick["k"]


def eps_noisy(rec, p, method, m="T"):
    """smallest achievable infidelity at error rate p per unit cost"""
    tau = rec["tau"]
    rmin = tau / TH0
    if method == "full":
        return p * (tau + 3) * (rec["GamL_T"] if m == "T" else rec["GamL_g"]), 1

    def best(pl, E):
        g = G(pl, m)
        r = max(rmin, tau * np.sqrt(E / (p * g)))
        return tau**2 * E / r + p * r * g

    if method == "qDRIFT":
        pl = dict(
            k=rec["L"], E0=rec["E1_0"], Eb=rec["E1"], G_T=rec["G1_T"], G_g=rec["G1_g"]
        )
        return best(pl, pl["Eb"]), rec["L"]
    cs = cands(rec, method, m)
    if method == "random":
        ks = sorted({x["k"] for x in rec["plans"]["random"]})
        v = [
            (
                np.mean(
                    [best(x, x["Eb"]) for x in rec["plans"]["random"] if x["k"] == k]
                ),
                k,
            )
            for k in ks
        ]
        return min(v + [(best(cs[-1], cs[-1]["Eb"]), rec["L"])])
    key = (
        (lambda x: x["E0"]) if method in ("graph", "magnitude") else (lambda x: x["Eb"])
    )
    pick = min(cs, key=lambda x: best(x, key(x)))
    return best(pick, pick["Eb"]), pick["k"]


# ------------------------------------------------------------------ shared by the figure, table and number scripts
TAU, R0, CC = 100.0, 40.0, 5.0  # evolution time and the T-gate cost model of the paper (sim.DEFAULTS["cost"])


def phi_best(rec, method):
    """(Gamma * Ebar, k) of the plan the planner would choose: lowest Gamma * E0 on its own t=0 error, priced on the true average"""
    p = min(cands(rec, method, "T"), key=lambda x: x["G_T"] * x["E0"])
    return p["G_T"] * p["Eb"], p["k"]


def Lstar(Phi, eps):
    """break-even number of terms against full H: tau^2 Phi/eps = (tau + ln 1/eps) * 2 (R + c L)"""
    return ((TAU**2 * Phi / eps) / (TAU + np.log(1 / eps)) / 2 - R0) / CC


def eps_cross(rec, method="graph"):
    """largest accuracy eps at which `method` is still cheaper than full H (None: never; eps_min: at every eps tested)"""
    es = np.logspace(np.log10(0.5), -4, 400)
    d = [N_eps(rec, e, method)[0] - N_eps(rec, e, "full")[0] for e in es]
    for i in range(len(es) - 1):
        if d[i] < 0 <= d[i + 1]:
            return es[i]
    return None if d[0] > 0 else es[-1]


def cross(x, f, g):
    """x where f first crosses g (log-linear interpolation); x is increasing; None if it does not"""
    d = np.log(np.asarray(f)) - np.log(np.asarray(g))
    i = next((i for i in range(len(d) - 1) if d[i] > 0 >= d[i + 1] or d[i] < 0 <= d[i + 1]), None)
    if i is None:
        return None
    w = d[i] / (d[i] - d[i + 1])
    return float(np.exp(np.log(x[i]) + w * (np.log(x[i + 1]) - np.log(x[i]))))


def kmean(rec, method, k, key="Eb"):
    """mean over the partitions of one k (random plans have several draws per k)"""
    return float(np.mean([p[key] for p in rec["plans"][method] if p["k"] == k]))


def ks_of(rec, method="graph"):
    return sorted({p["k"] for p in rec["plans"][method] if p["k"] > 1})


def study_tags(by, model, n, init="neel"):
    """records of one series: {L: [records over instances]} for model B, [record] for model A"""
    out = {}
    for r in by.values():
        h = r["hamiltonian"]
        if h["model"] != model or h["n"] != n or r["init"]["kind"] != init:
            continue
        out.setdefault(r["L"], []).append(r)
    for L in out:
        out[L].sort(key=lambda r: r["hamiltonian"].get("seed", 0))
    return out


def grid_cases(task="study", root="runs"):
    """the saved cases of configs/grid.py (only those: stale runs of earlier code in <root>/cases are ignored)"""
    import os, sys

    from sim import load_cases, resolve

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"))
    import grid

    hs = {h for _, c, _, h in resolve(grid.config) if c["task"] == task}
    return load_cases(root, task, hashes=hs)
