"""Scalable core: Pauli-term Hamiltonians on n qubits (state vectors), correlation graph, planning, MC validation."""

import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def _par(d):
    p = np.zeros(d, np.int8)
    for b in range(d.bit_length() - 1):
        p ^= ((np.arange(d) >> b) & 1).astype(np.int8)
    return p


class Ham:
    def __init__(s, n, xs, zs, coef):
        s.n = n
        s.d = 1 << n
        s.xs = np.asarray(xs)
        s.zs = np.asarray(zs)
        c = np.asarray(coef, float)
        s.lam = np.abs(c).sum()
        s.a = np.abs(c) / s.lam
        s.sg = np.sign(c)
        s.L = len(c)
        s.idx = np.arange(s.d)
        s.par = _par(s.d)
        s.ph = np.array([1, 1j, -1, -1j])

    def _phase(
        s, g
    ):  # vector over t: sign * i^{x.z} (-1)^{z.(t^x)} ; (P psi)[t]=phase[t]*psi[t^x]
        x, z = int(s.xs[g]), int(s.zs[g])
        t = s.idx ^ x
        return (
            s.sg[g]
            * s.ph[bin(x & z).count("1") % 4]
            * (1 - 2 * s.par[t & z].astype(float)),
            t,
        )

    def apply(s, g, psi):
        f, t = s._phase(g)
        return f * psi[t]

    def apply_all(s, psi):
        X = np.empty((s.L, s.d), complex)
        for g in range(s.L):
            X[g] = s.apply(g, psi)
        return X

    def term_sparse(s, g):
        f, t = s._phase(g)
        return sp.csr_matrix((f.astype(complex), (s.idx, t)), shape=(s.d, s.d))

    def block_sparse(s, b, w):
        M = None
        for g, wg in zip(b, w):
            m = wg * s.term_sparse(g)
            M = m if M is None else M + m
        return M

    def Hbar(s):
        """sum_g a_g P_g as a sparse matrix. Terms with the same flip pattern x share the sparsity pattern, so their
        phase vectors are summed first: memory is (#distinct x) * 2^n instead of L * 2^n."""
        acc = {}
        for g in range(s.L):
            f, _ = s._phase(g)
            x = int(s.xs[g])
            if x in acc:
                acc[x] += s.a[g] * f
            else:
                acc[x] = s.a[g] * f
        M = None
        for x, f in acc.items():
            m = sp.csr_matrix((f.astype(complex), (s.idx, s.idx ^ x)), shape=(s.d, s.d))
            M = m if M is None else M + m
        return M.tocsr()


def model_A(n, seed=0, alpha=1.0, Delta=0.7, hx=0.6, hz=0.5):
    """long-range Heisenberg-XXZ + random fields: L=3n(n-1)/2+2n"""
    rng = np.random.default_rng(seed)
    xs = []
    zs = []
    c = []
    for i in range(n):
        for j in range(i + 1, n):
            w = 1.0 / abs(i - j) ** alpha
            for (x, z), J in [((1, 0), 1.0), ((1, 1), 1.0), ((0, 1), Delta)]:
                xs.append((x << i) | (x << j))
                zs.append((z << i) | (z << j))
                c.append(w * J * (1 + 0.3 * rng.standard_normal()))
    for i in range(n):
        xs.append(1 << i)
        zs.append(0)
        c.append(hx * (1 + 0.3 * rng.standard_normal()))
    for i in range(n):
        xs.append(0)
        zs.append(1 << i)
        c.append(hz * rng.uniform(-1, 1))
    return Ham(n, xs, zs, c)


def model_B(n, L, seed=0, alpha=1.0, maxw=4):
    """L distinct random Pauli strings of weight 2..maxw, power-law magnitudes, random signs (molecular-like)"""
    rng = np.random.default_rng(seed)
    seen = set()
    xs = []
    zs = []
    while len(xs) < L:
        w = rng.integers(2, maxw + 1)
        q = rng.choice(n, w, replace=False)
        x = z = 0
        for i in q:
            k = rng.integers(1, 4)
            x |= (k & 1) << i
            z |= (k >> 1) << i
        if (x, z) in seen or (x | z) == 0:
            continue
        seen.add((x, z))
        xs.append(x)
        zs.append(z)
    m = np.arange(1, L + 1) ** (-alpha)
    rng.shuffle(m)
    return Ham(n, xs, zs, m * rng.choice([-1, 1], L))


def init_state(n, kind="neel", seed=0):
    d = 1 << n
    if kind in ("neel", "hf"):
        s = 0
        for i in range(n):
            if (kind == "neel" and i % 2 == 1) or (kind == "hf" and i < n // 2):
                s |= 1 << i
        psi = np.zeros(d, complex)
        psi[s] = 1
        return psi
    rng = np.random.default_rng(seed)
    psi = np.array([1.0 + 0j])
    for i in range(n):
        t = rng.uniform(0, np.pi)
        p = rng.uniform(0, 2 * np.pi)
        psi = np.kron([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)], psi)
    return psi


def trajectory(H, psi0, tau, nt):
    A = -1j * H.Hbar()
    ts = np.linspace(0, tau, nt)
    return ts, expm_multiply(A, psi0, start=0, stop=tau, num=nt, endpoint=True)


def graph(H, psi):
    X = H.apply_all(psi)
    mu = np.einsum("gd,d->g", X, psi.conj())  # <U_g>
    X -= mu[:, None] * psi[None, :]
    Xr = X.view(np.float64)  # Re<x_g,x_h> = sum of real-part and imaginary-part products: one real matmul
    return Xr @ Xr.T


def graphs(H, Psi, keep_all=True):
    """(K at the first state, mean K over the states, list of all K or None). keep_all=False streams the states:
    only three L x L matrices are alive at a time (needed for L ~ 3e4)."""
    if keep_all:
        Ks = [graph(H, p) for p in Psi]
        return Ks[0], np.mean(Ks, 0), Ks
    K0 = graph(H, Psi[0])
    Kb = K0.copy()
    for p in Psi[1:]:
        Kb += graph(H, p)
    Kb /= len(Psi)
    return K0, Kb, None


def price_blocks(blocks, a, K):
    E = -a @ K @ a
    for b in blocks:
        b = np.asarray(b)
        E += (
            a[b[0]] * K[b[0], b[0]]
            if len(b) == 1
            else a[b] @ K[np.ix_(b, b)] @ a[b] / a[b].sum()
        )
    return float(E)


def price(lab, a, K, k=None):
    k = k or lab.max() + 1
    o = np.argsort(lab, kind="stable")
    cuts = np.searchsorted(lab[o], np.arange(k + 1))
    return price_blocks(
        [o[cuts[c] : cuts[c + 1]] for c in range(k) if cuts[c + 1] > cuts[c]], a, K
    )


def anticluster(a, K, k, sweeps=6, seed=0, init=None):
    L = len(a)
    rng = np.random.default_rng(seed)
    if init is None:
        lab = np.empty(L, int)
        lab[np.argsort(-a)] = np.arange(L) % k
    else:
        lab = init.copy()
    W = np.zeros((k, L))
    W[lab, np.arange(L)] = a
    u = W @ K
    al = W.sum(1)
    S = (u * W).sum(1)
    ar = np.arange(L)
    dK = np.diag(K)
    for _ in range(sweeps):
        moved = 0
        for g in rng.permutation(L):
            i = lab[g]
            ag = a[g]
            Si = (
                S[i]
                - 2 * ag * u[i, g]
                + ag**2 * dK[g]
                + 2 * a * (u[i] - ag * K[g])
                + a**2 * dK
            )
            ai = al[i] - ag + a
            j = lab
            uj_h = u[j, ar]
            uj_g = u[j, g]
            Sj = (
                S[j]
                - 2 * a * uj_h
                + a**2 * dK
                + 2 * ag * (uj_g - a * K[g])
                + ag**2 * dK[g]
            )
            aj = al[j] - a + ag
            dl = Si / ai + Sj / aj - S[i] / al[i] - S[j] / al[j]
            dl[lab == i] = np.inf
            h = int(np.argmin(dl))
            if dl[h] < -1e-13:
                j = lab[h]
                ah = a[h]
                du = ah * K[h] - ag * K[g]
                u[i] += du
                u[j] -= du
                S[i] = Si[h]
                S[j] = Sj[h]
                al[i] = ai[h]
                al[j] = aj[h]
                lab[g], lab[h] = j, i
                moved += 1
        if moved == 0:
            break
    return lab


def labels_to_blocks(lab, k):
    return [list(np.where(lab == c)[0]) for c in range(k)]


def random_labels(L, k, rng):
    lab = np.empty(L, int)
    lab[rng.permutation(L)] = np.arange(L) % k
    return lab


def magnitude_blocks(a, m):
    o = np.argsort(-a)
    return (
        [list(o[:m])] + [[int(g)] for g in o[m:]]
        if m > 1
        else [[g] for g in range(len(a))]
    )


COST = dict(model="T", R=40.0, c=5.0, kq=2.0)


def gamma(m, model=None):
    mo = model or COST["model"]
    if mo == "T":
        return COST["R"] if m == 1 else COST["kq"] * (COST["R"] + COST["c"] * m)
    return 1.0 if m == 1 else 1.0 + 2 * np.log2(m) + 2.0 * m


def Gam(blocks, a, model=None):
    return sum(a[b].sum() * gamma(len(b), model) for b in blocks)


# ---- trajectory Monte Carlo of the sampled channel (validation) ----
def mc_infid(H, blocks, tau, r, psi0, phiT, ntraj, seed=0):
    rng = np.random.default_rng(seed)
    th = tau / r
    a = H.a
    al = np.array([a[b].sum() for b in blocks])
    Ms = [H.block_sparse(b, a[b] / al_) for b, al_ in zip(blocks, al)]
    Psi = np.tile(psi0[:, None], (1, ntraj))
    for s in range(r):
        c = rng.choice(len(blocks), size=ntraj, p=al)
        for cc in np.unique(c):
            cols = np.where(c == cc)[0]
            v = Psi[:, cols]
            term = v
            out = v.copy()
            for o in range(1, 8):
                term = (-1j * th / o) * (Ms[cc] @ term)
                out += term
            Psi[:, cols] = out
    f = np.abs(phiT.conj() @ Psi) ** 2
    return 1 - f.mean(), f.std() / np.sqrt(ntraj)
