"""The side experiments of the paper as functions of a Hamiltonian, called by the tasks of sim.py (so that they are configured,
cached and run exactly like the main study):

    planners_exp    alternative ways of choosing the grouping (cluster-then-deal, sort-and-deal, greedy) against swaps
    pairs_exp       blocks of two terms: exact matching, swaps, commuting pairs
    sampling_exp    block probabilities: weight-proportional, Hilbert-Schmidt, variance-optimal at t=0 and with hindsight
    freeblocks_exp  unequal block sizes: moves on the predicted cost, weight-balanced dealing

Each returns a JSON-serialisable dict. Costs use core.COST (set by the caller from the case's `cost` entry).
"""

import numpy as np

import core
from core import (
    anticluster,
    graph,
    graphs,
    labels_to_blocks,
    price,
    price_blocks,
    random_labels,
    trajectory,
)

KAP_R_C = lambda: (core.COST["kq"], core.COST["R"], core.COST["c"])


def kgrid(L, ms):
    """block counts k = ceil(L/m) for the target block sizes m <= L/2 (same rule as sim._kgrid, without the extra k=2)"""
    return sorted({int(np.ceil(L / m)) for m in ms if m <= L / 2})


# ------------------------------------------------------------------ alternative planners
def _embed(K, r):
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 0, None)
    o = np.argsort(-w)
    return V[:, o[:r]] * np.sqrt(w[o[:r]])


def _cluster_deal(a, K, k, seed=0):
    from scipy.cluster.vq import kmeans2

    L = len(a)
    Y = _embed(K, 20)
    m = int(np.ceil(L / k))  # m clusters of ~k terms -> each block takes one from each
    _, cl = kmeans2(Y, m, seed=seed, minit="++")
    lab = np.empty(L, int)
    for i, g in enumerate(np.argsort(cl, kind="stable")):
        lab[g] = i % k
    return lab


def _svd_snake(a, K, k):
    Y = _embed(K, 1)[:, 0]
    lab = np.empty(len(a), int)
    for i, g in enumerate(np.argsort(Y)):
        r, p = divmod(i, k)
        lab[g] = p if r % 2 == 0 else k - 1 - p
    return lab


def _greedy(a, K, k):
    L = len(a)
    Y = _embed(K, L)
    o = np.argsort(-(a * np.linalg.norm(Y, axis=1)))
    s = np.zeros((k, Y.shape[1]))
    al = np.full(k, 1e-12)
    lab = np.empty(L, int)
    cnt = np.zeros(k, int)
    cap = int(np.ceil(L / k))
    for g in o:
        v = a[g] * Y[g]
        d = ((s + v) ** 2).sum(1) / (al + a[g]) - (s**2).sum(1) / al
        d[cnt >= cap] = np.inf
        c = int(np.argmin(d))
        lab[g] = c
        s[c] += v
        al[c] += a[g]
        cnt[c] += 1
    return lab


def planners_exp(H, psi0, ms, rng, sweeps=6):
    """error per step E at t=0 of each planner, for each k = ceil(L/m)"""
    a, L = H.a, H.L
    K = graph(H, psi0)
    E1 = price_blocks([[g] for g in range(L)], a, K)
    out = []
    for k in kgrid(L, ms):
        P = lambda lab: price(lab, a, K, k)
        r = dict(k=k, m=L / k, E1=E1)
        r["random"] = P(random_labels(L, k, rng))
        r["swap"] = P(anticluster(a, K, k, sweeps=sweeps))
        lab = _cluster_deal(a, K, k)
        r["cluster+deal"] = P(lab)
        r["cluster+deal -> swap"] = P(anticluster(a, K, k, sweeps=sweeps, init=lab))
        lab = _svd_snake(a, K, k)
        r["svd-snake"] = P(lab)
        r["svd-snake -> swap"] = P(anticluster(a, K, k, sweeps=sweeps, init=lab))
        lab = _greedy(a, K, k)
        r["greedy"] = P(lab)
        r["greedy -> swap"] = P(anticluster(a, K, k, sweeps=sweeps, init=lab))
        out.append(r)
    return dict(L=L, results=out)


# ------------------------------------------------------------------ blocks of two
def _commute(H, g, h):
    s = (int(H.xs[g]) & int(H.zs[h])) ^ (int(H.zs[g]) & int(H.xs[h]))
    return bin(s).count("1") % 2 == 0


def _match(a, K, edges, perfect):
    """max total reduction of E over pairs; reduction of pair {g,h} = a_g a_h Var(U_g-U_h)/(a_g+a_h)"""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(a)))
    for g, h in edges:
        r = a[g] * a[h] * (K[g, g] + K[h, h] - 2 * K[g, h]) / (a[g] + a[h])
        G.add_edge(g, h, weight=r + (10.0 if perfect else 0.0))  # offset forces a perfect matching
    M = nx.max_weight_matching(G, maxcardinality=perfect)
    paired = {x for e in M for x in e}
    return [list(e) for e in M] + [[g] for g in range(len(a)) if g not in paired]


def _pair_cost_step(blocks, a, H):
    """average T gates per step: single term R; commuting pair 2R (two rotations);
    otherwise a coherent pair, min(LCU kappa(R+2c), 3R by conjugating one rotation)"""
    KAP, R, C = KAP_R_C()
    G = 0.0
    for b in blocks:
        al = a[b].sum()
        if len(b) == 1:
            G += al * R
        elif len(b) == 2 and _commute(H, *b):
            G += al * 2 * R
        elif len(b) == 2:
            G += al * min(KAP * (R + 2 * C), 3 * R)
        else:  # the one block of three that an odd L leaves when L // 2 blocks are dealt
            G += al * KAP * (R + len(b) * C)
    return G


def pairs_exp(H, psi0, tau, nt, eps, rng):
    KAP, R, C = KAP_R_C()
    ts, Psi = trajectory(H, psi0, tau, nt)
    Ks = [graph(H, p) for p in Psi]
    a, L = H.a, H.L
    K0 = Ks[0]
    Ebar = lambda bl: float(np.mean([price_blocks(bl, a, K) for K in Ks]))
    allE = [(g, h) for g in range(L) for h in range(g + 1, L)]
    comE = [(g, h) for g, h in allE if _commute(H, g, h)]
    plans = {"qDRIFT": [[g] for g in range(L)]}
    plans["pairs: matching"] = _match(a, K0, allE, True)
    kp = (L + 1) // 2  # pairs (and one single term when L is odd, as the matching leaves it)
    plans["pairs: swaps"] = labels_to_blocks(anticluster(a, K0, kp), kp)
    plans["pairs: random"] = labels_to_blocks(random_labels(L, kp, rng), kp)
    plans["commuting pairs: matching"] = _match(a, K0, comE, False)
    used, rb = set(), []  # random maximal matching on commuting edges
    for i in rng.permutation(len(comE)):
        g, h = comE[i]
        if g not in used and h not in used:
            used |= {g, h}
            rb.append([g, h])
    plans["commuting pairs: random"] = rb + [[g] for g in range(L) if g not in used]
    res = {}
    for nm, bl in plans.items():
        E0, Eb, G = price_blocks(bl, a, K0), Ebar(bl), _pair_cost_step(bl, a, H)
        res[nm] = dict(
            E0=E0,
            Eb=Eb,
            G=G,
            N=G * max(tau, tau**2 * Eb / eps),
            npairs=sum(len(b) == 2 for b in bl),
            frac_paired=float(sum(a[b].sum() for b in bl if len(b) == 2)),
        )
    res["full"] = dict(N=(tau + np.log(1 / eps)) * KAP * (R + C * L))
    res["commuting_fraction_of_pairs"] = len(comE) / len(allE)
    return dict(L=L, eps=eps, results=res)


# ------------------------------------------------------------------ block probabilities
def _block_stats(blocks, a, K):
    """alpha_c^2 Var(A_c) = a_c^T K a_c for every block, and Var(Hbar)"""
    return np.array([a[b] @ K[np.ix_(b, b)] @ a[b] for b in blocks]), a @ K @ a


def _probs(rule, blocks, a, W0, Wbar, floor):
    al = np.array([a[b].sum() for b in blocks])
    if rule == "alpha":
        p = al
    elif rule == "hs":
        p = np.array([np.linalg.norm(a[b]) for b in blocks])
    elif rule == "var0":
        p = np.sqrt(np.maximum(W0, floor * W0.max()))
    else:  # varbar
        p = np.sqrt(np.maximum(Wbar, 1e-15))
    return p / p.sum()


def sampling_exp(H, psi0, tau, nt, ms, rng, floor=1e-3):
    """Fixed-state error per step for arbitrary block probabilities p (unbiased: sum_c p_c (alpha_c/p_c) A_c = Hbar):
        E_p = sum_c alpha_c^2 Var(A_c) / p_c - Var(Hbar),   optimum p_c ~ alpha_c sqrt(Var(A_c)).
    Rules: alpha (p_c = alpha_c), hs (p_c ~ ||a_c||_2), var0 (optimum at t=0, floored), varbar (optimum for the trajectory
    average, hindsight). Groupings planned on the graph at t=0, the identity ("weights"), at random, and the averaged graph."""
    KAP, R0, CC = KAP_R_C()
    a, L = H.a, H.L
    ts, Psi = trajectory(H, psi0, tau, nt)
    K0, Kbar, _ = graphs(H, Psi, keep_all=False)
    ks = kgrid(L, ms)
    plans = {}
    for k in ks:
        plans[("graph", k)] = labels_to_blocks(anticluster(a, K0, k), k)
        plans[("weights", k)] = labels_to_blocks(anticluster(a, np.eye(L), k), k)
        plans[("random", k)] = labels_to_blocks(random_labels(L, k, rng), k)
        plans[("oracle", k)] = labels_to_blocks(anticluster(a, Kbar, k), k)
    del K0, Kbar
    plans[("qDRIFT", L)] = [[g] for g in range(L)]
    W = {pk: [] for pk in plans}
    VH = []
    for p in Psi:  # per-time block statistics, one K(t) at a time
        K = graph(H, p)
        for pk, bl in plans.items():
            W[pk].append(_block_stats(bl, a, K)[0])
        VH.append(a @ K @ a)
        del K
    VH = np.array(VH)
    res = {}
    for (pl, k), bl in plans.items():
        Wt = np.array(W[(pl, k)])  # nt x blocks
        W0, Wbar = Wt[0], Wt.mean(0)
        sizes = np.array([len(b) for b in bl])
        Gam = np.where(sizes == 1, R0, KAP * (R0 + CC * sizes))
        for rule in ("alpha", "hs", "var0", "varbar"):
            if pl == "qDRIFT" and rule == "hs":
                continue  # HS sampling of single terms is qDRIFT itself
            p = _probs(rule, bl, a, W0, Wbar, floor)
            Et = (Wt / p).sum(1) - VH
            G = float(p @ Gam)
            res[f"{pl}|{k}|{rule}"] = dict(
                E0=float(Et[0]),
                Eb=float(Et.mean()),
                G=G,
                Phi=G * float(Et.mean()),
                Phi0=G * float(Et[0]),
                maxratio=float(np.max(np.array([a[b].sum() for b in bl]) / p)),
            )
    return dict(L=L, ks=ks, results=res)


# ------------------------------------------------------------------ unequal block sizes
def freeblocks_exp(H, psi0, tau, nt, ms, eps, rng=None):
    """Groupings with unequal block sizes. Algorithm 1 deals the terms round-robin and then swaps, so every block has L/k
    terms. Here the sizes are free: starting from the equal-size grouping, a local search moves single terms to another
    block or to a new block of their own whenever that lowers the predicted cost to reach accuracy eps on the planning
    graph, N = Gamma(P) * max(tau, tau^2 E(P)/eps); swaps between blocks are interleaved. Planners: "graph" (K at t=0) and
    "weights" (identity); each grouping is evaluated along the trajectory with weight-proportional (alpha) and
    Hilbert-Schmidt (hs) block probabilities."""
    KAP, R0, CC = KAP_R_C()
    a, L = H.a, H.L

    def gam(n):
        n = np.asarray(n)
        return np.where(n <= 1, R0, KAP * (R0 + CC * n))

    def ncost(G, E):
        return G * np.maximum(tau, tau**2 * np.asarray(E) / eps)

    ts, Psi = trajectory(H, psi0, tau, nt)
    K0 = graph(H, Psi[0])
    plans = {}
    for pl, KK in (("graph", K0), ("weights", np.eye(L))):
        for k in kgrid(L, ms):
            eq = anticluster(a, KK, k)
            plans[(pl, "equal", k)] = labels_to_blocks(eq, k)
            fr = _move_search(a, KK, eq.copy(), gam, ncost, R0)
            plans[(pl, "free", k)] = labels_to_blocks(fr, fr.max() + 1)
            cw = _cluster_deal_w(a, KK, k)
            plans[(pl, "clw", k)] = labels_to_blocks(cw, cw.max() + 1)
            cf = _move_search(a, KK, cw.copy(), gam, ncost, R0)
            plans[(pl, "clw+free", k)] = labels_to_blocks(cf, cf.max() + 1)
    del K0
    Ks = [graph(H, p) for p in Psi]
    VH = np.array([a @ K @ a for K in Ks])
    E1 = np.array([a @ np.diag(K) for K in Ks]) - VH
    Nq = ncost(R0, E1.mean())
    Nf = (tau + np.log(1 / eps)) * KAP * (R0 + CC * L)
    res = {}
    for (pl, mode, k), bl in plans.items():
        ev = _evaluate(bl, a, Ks, VH, gam)
        sz = np.array([len(b) for b in bl])
        al = np.array([a[b].sum() for b in bl])
        res[f"{pl}|{mode}|{k}"] = dict(
            k=len(bl),
            sizes=[int(sz.min()), float(np.median(sz)), int(sz.max())],
            n_single=int((sz == 1).sum()),
            weight_single=float(al[sz == 1].sum()),
            **{f"{r}_{q}": v for r, d in ev.items() for q, v in d.items()},
            **{f"N_{r}": float(ncost(ev[r]["G"], ev[r]["Eb"])) for r in ev},
        )
    return dict(L=L, eps=eps, E1bar=float(E1.mean()), Nq=float(Nq), Nf=float(Nf), results=res)


def _move_search(a, K, lab, gam, ncost, R0, rounds=4, sweeps=6, seed=0):
    """local search on N = Gamma * max(tau, tau^2 E/eps) with single-term moves (to any block or a new singleton) and swaps"""
    rng = np.random.default_rng(seed)
    L = len(a)
    dK = np.diag(K)
    aKa = a @ K @ a
    for _ in range(rounds):
        k = lab.max() + 1
        W = np.zeros((k, L))
        W[lab, np.arange(L)] = a
        u = W @ K  # u[c, g] = sum_{h in c} a_h K_hg
        al = W.sum(1)
        n = np.bincount(lab, minlength=k).astype(float)
        S = (u * W).sum(1)  # a_c^T K a_c
        moved = 0
        for _ in range(sweeps):
            changed = 0
            for g in rng.permutation(L):
                i = lab[g]
                ag = a[g]
                live = n > 0
                E = (S[live] / al[live]).sum() - aKa
                G = (al * gam(n)).sum()
                Si = S[i] - 2 * ag * u[i, g] + ag**2 * dK[g]
                ai = al[i] - ag
                ni = n[i] - 1
                Sj = S + 2 * ag * u[:, g] + ag**2 * dK[g]
                aj = al + ag
                nj = n + 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    oldj = np.where(n > 0, S / al, 0.0)
                    Enew = E - S[i] / al[i] + (Si / ai if ni > 0 else 0.0) - oldj + Sj / aj
                Gnew = G - al[i] * gam(n[i]) + (ai * gam(ni) if ni > 0 else 0.0) - al * gam(n) + aj * gam(nj)
                Phi = ncost(G, E)
                PhiNew = ncost(Gnew, Enew)
                PhiNew[i] = np.inf
                # move to a new singleton block
                Es = E - S[i] / al[i] + (Si / ai if ni > 0 else 0.0) + dK[g] * ag
                Gs = G - al[i] * gam(n[i]) + (ai * gam(ni) if ni > 0 else 0.0) + ag * R0
                j = int(np.argmin(PhiNew))
                best = PhiNew[j]
                new = ncost(Gs, Es) < best and n[i] > 1
                if (ncost(Gs, Es) if new else best) < Phi * (1 - 1e-12):
                    if new:  # open a new block
                        j = len(al)
                        u = np.vstack([u, np.zeros(L)])
                        al = np.append(al, 0.0)
                        n = np.append(n, 0.0)
                        S = np.append(S, 0.0)
                    du = ag * K[g]
                    S[i] = Si
                    al[i] = ai
                    n[i] = ni
                    u[i] -= du
                    S[j] = S[j] + 2 * ag * u[j, g] + ag**2 * dK[g]
                    al[j] += ag
                    n[j] += 1
                    u[j] += du
                    lab[g] = j
                    changed += 1
            moved += changed
            if changed == 0:
                break
        # compact labels, then improve E at fixed sizes with swaps
        _, lab = np.unique(lab, return_inverse=True)
        lab = anticluster(a, K, lab.max() + 1, init=lab)
        if moved == 0:
            break
    return lab


def _cluster_deal_w(a, K, k, r=20, seed=0):
    """cluster the terms in an r-dimensional embedding of K (top eigenvectors), then deal each cluster's members,
    heaviest first, to the block with the smallest total weight: blocks of similar weight and mixed content,
    whose sizes differ (a heavy term ends up in a small block). K = identity: one cluster, pure weight balancing."""
    from scipy.cluster.vq import kmeans2

    L = len(a)
    if np.allclose(K, np.diag(np.diag(K))):
        cl = np.zeros(L, int)
    else:
        w, V = np.linalg.eigh(K)
        o = np.argsort(-w)[:r]
        Y = V[:, o] * np.sqrt(np.clip(w[o], 0, None))
        _, cl = kmeans2(Y, max(1, L // k), seed=seed, minit="++")
    lab = np.empty(L, int)
    al = np.zeros(k)
    for c in sorted(set(cl), key=lambda c: -a[cl == c].sum()):
        for g in sorted(np.where(cl == c)[0], key=lambda g: -a[g]):
            j = int(np.argmin(al))
            lab[g] = j
            al[j] += a[g]
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def _evaluate(blocks, a, Ks, VH, gam):
    """trajectory-averaged error and cost per step for alpha and HS sampling"""
    Wt = np.array([[a[b] @ K[np.ix_(b, b)] @ a[b] for b in blocks] for K in Ks])  # nt x blocks
    al = np.array([a[b].sum() for b in blocks])
    hs = np.array([np.linalg.norm(a[b]) for b in blocks])
    hs /= hs.sum()
    g = gam([len(b) for b in blocks])
    out = {}
    for nm, p in (("alpha", al), ("hs", hs)):
        Et = (Wt / p).sum(1) - VH
        out[nm] = dict(Eb=float(Et.mean()), E0=float(Et[0]), G=float(p @ g))
    return out
