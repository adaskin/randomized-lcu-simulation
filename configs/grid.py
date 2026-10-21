"""The full grid: every simulation behind the paper's numerical section, on one fixed set of settings.

    python run.py configs/grid.py --dry                       the plan: cases, estimated memory and hours, what is saved already
    python run.py configs/grid.py --jobs 8 --mem-gb 80        everything, 8 cases at a time (resumable: saved cases are skipped)
    python run.py configs/grid.py --max-case-gb 8             the part that fits a small machine

Grid.   n in {5, 10, 15, 20} qubits.  Model B: every L = 2^j from 16 up to the largest that fits (below), at every n.
        Model A: its L is fixed by n (L = 3n(n-1)/2 + 2n = 40, 155, 345, 610), so it is a series in n only.
Largest L per n (model B):
        n = 5, 10   L <= half the number of distinct Pauli strings of weight 2-4 (382 at n=5, 10327 at n=10), i.e. 256 and 8192:
                    beyond that the instance would contain most of all possible strings
        n = 15      32768: the memory of the L x L graphs (three at once) and of the L vectors of dimension 2^n (about 45 GB)
        n = 20      1024: the sparse matrix of the mean Hamiltonian has one block of 2^20 entries per distinct flip pattern
Settings (identical everywhere): tau = 100, 11 time points for the trajectory average, Neel initial state, cost model R=40,
        kappa=2, c=5, block sizes m = 2, 4, ..., 256 (k = ceil(L/m)), 3 random partitions per k, 4 swap sweeps.
Instances: model B has 3 random instances (seeds 0, 1, 2) for L <= 4096 and one (seed 0) above.
Side experiments (planners, blocks of two, block probabilities, unequal block sizes) and the Monte Carlo check of the error
formula run on a subset of this grid, below. The check is run at tau = 8 for n = 5, 10, 15 and at tau = 100 for n = 5, 10; at
n = 15, tau = 100 it was started (model A and model B with L = 256) and stopped after about 16 h of one core each (projected 1.5-3 days),
so it is not part of the grid.
"""

from math import comb

NS = [5, 10, 15, 20]
LS = [2**j for j in range(4, 16)]  # 16 ... 32768
FRAC_STRINGS = 0.5  # model B: L at most this fraction of all weight-2..4 Pauli strings
LMAX_MEM = {15: 32768, 20: 1024}  # memory-limited maxima (see above)
MANY_SEEDS_UP_TO = 4096
DRIFT_UP_TO = 4096  # the per-time graphs of the drift study are kept in memory: only for L up to this

STUDY = dict(
    tau=100.0,
    nt=11,
    nrand=3,
    ms=[2, 4, 8, 16, 32, 64, 128, 256],
    sweeps=4,
    plans=["graph", "oracle", "random", "magnitude", "weights", "variances"],
)


def strings(n):  # distinct Pauli strings of weight 2..4
    return sum(comb(n, w) * 3**w for w in range(2, 5))


def lmax(n):
    return min(
        LMAX_MEM.get(n, 10**9), max(L for L in LS if L <= FRAC_STRINGS * strings(n))
    )


def A_natural(n):
    return 3 * n * (n - 1) // 2 + 2 * n


def tag(model, n, L=None, s=0, init="neel"):
    t = f"A n={n}" if model == "A" else f"B n={n} L={L} s={s}"
    return t + ("" if init == "neel" else f" init={init}")


def ham(model, n, L=None, s=0):
    return (
        dict(model="A", n=n, seed=0)
        if model == "A"
        else dict(model="B", n=n, L=L, seed=s)
    )


def study(model, n, L=None, s=0, init="neel"):
    LL = A_natural(n) if model == "A" else L
    c = dict(
        tag=tag(model, n, L, s, init),
        seed=0,
        hamiltonian=ham(model, n, L, s),
        init=dict(kind=init, seed=5 if init != "neel" else 0),
    )
    if LL <= DRIFT_UP_TO:
        c["drift_k_frac"] = 0.1
    return c


def aux(task, model, n, L=None, **kw):
    return dict(
        task=task,
        tag=tag(model, n, L),
        seed=0,
        hamiltonian=ham(model, n, L),
        nt=11,
        **kw,
    )


def study_cases():
    out = []
    for n in NS:  # model A: the series in n
        out.append(study("A", n))
    for L in LS:  # model B, smallest first; one L at all n that reach it
        for n in NS:
            if L <= lmax(n):
                for s in range(3 if L <= MANY_SEEDS_UP_TO else 1):
                    out.append(study("B", n, L, s))
    for n in (10, 15):  # robustness to the initial state
        for init in ("prod", "hf"):
            out.append(study("A", n, init=init))
            out.append(study("B", n, 1024, 0, init=init))
    return out


AUX_HAMS = [
    ("A", 10, None),
    ("A", 15, None),
    ("B", 10, 256),
    ("B", 10, 1024),
    ("B", 15, 256),
    ("B", 15, 1024),
    ("B", 15, 4096),
]


def side_cases():
    out = []
    for model, n, L in AUX_HAMS:
        if L is None or L <= 1024:
            out.append(aux("planners", model, n, L))
        if L is None or L <= 256:
            out.append(aux("pairs", model, n, L))
        out.append(aux("sampling", model, n, L, ms_aux=[2, 4, 8, 16, 32, 64]))
        out.append(aux("freeblocks", model, n, L, ms_aux=[4, 8, 16, 32, 64]))
    return out


def validation_cases():
    out = []
    for tau, nt in ((8.0, 17), (100.0, 41)):
        for n in (5, 10, 15):
            if n == 15 and tau > 10:
                continue  # stopped after ~16 h, see the module docstring
            for model, L in (("A", None), ("B", 256)):
                plans = [
                    dict(method="qDRIFT", ntraj=100 if tau > 10 else 200),
                    dict(method="graph", k_div=4),
                    dict(method="graph", k_div=10),
                    dict(method="graph", k_div=30),
                    dict(method="random", k_div=10),
                ]
                out.append(
                    dict(
                        task="validate",
                        tag=tag(model, n, L) + f" tau={tau:g}",
                        seed=1,
                        tau=tau,
                        nt=nt,
                        hamiltonian=ham(model, n, L),
                        validate=dict(plans=plans, r=[], target=[0.03, 0.1], ntraj=200),
                    )
                )
    return out


config = dict(
    name="grid",
    seed=0,
    base=STUDY,
    cases=study_cases() + side_cases() + validation_cases(),
)
