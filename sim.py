"""Config-driven experiment runner.

    from sim import run
    out_dir = run(config)

or from the shell:  python run.py configs/scale.py

A config is one plain dict (JSON-serialisable):

    name        run name; the output goes to <output_dir>/<name>_<timestamp>/
    output_dir  parent directory of the run directories (default "runs")
    seed        default seed; a case's RNG is derived from (seed, case index) unless the case sets "seed"
    base        settings shared by all cases (see DEFAULTS below)
    cases       list of dicts; each is deep-merged over `base` and run independently

Each case has a "task":
    "study"      build H, evolve, build correlation graphs, plan blocks (graph/oracle/random/magnitude/...), price them
    "validate"   Monte Carlo of the sampled channel against the predicted infidelity
    "planners"   alternative planners (cluster-then-deal, sort-and-deal, greedy) against swaps          (experiments.py)
    "pairs"      blocks of two terms: exact matching, swaps, commuting pairs                             (experiments.py)
    "sampling"   block probabilities: weight, Hilbert-Schmidt, variance-optimal                          (experiments.py)
    "freeblocks" unequal block sizes                                                                     (experiments.py)
Keys of a case that start with "_" are hints for the scheduler (_mem_gb, _cores, _hours); they are not part of the hash.

Everything needed to regenerate the figures is saved in the run directory:
    <run>/config.json   the config (rerunnable: python run.py <run>/config.json)
    <run>/meta.json     versions, timings, per-case seeds and hashes (cached or computed)
    <run>/results.json  {"records": [one record per case]}
    <output_dir>/cases/<hash>/  case.json (resolved settings + seed), result.json, arrays.npz
                        (block labels of every plan, optionally graphs), info.json (seconds, peak memory, machine); reused by later runs
"""

import copy
import datetime
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

import core
import experiments
from core import (
    Gam,
    Ham,
    anticluster,
    gamma,
    graphs,
    init_state,
    labels_to_blocks,
    magnitude_blocks,
    mc_infid,
    model_A,
    model_B,
    price,
    price_blocks,
    random_labels,
    trajectory,
)

DEFAULTS = dict(
    task="study",
    hamiltonian=dict(
        model="A", n=6, seed=0
    ),  # remaining keys go to the model constructor
    init=dict(kind="neel", seed=0),
    tau=100.0,
    nt=11,  # trajectory points used for the time-averaged graph
    cost=dict(model="T", R=40.0, c=5.0, kq=2.0),
    # --- study ---
    plans=["graph", "oracle", "random", "magnitude"],
    nrand=3,  # random partitions per k
    ms=[
        2,
        3,
        4,
        6,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
    ],  # target mean block sizes -> k = ceil(L/m)
    sweeps=4,  # anticlustering sweeps
    drift_k_frac=None,  # if set, track plan quality along the trajectory with k ~ L*frac
    save_graphs=False,  # store K0 and the averaged graph in arrays.npz (L x L each)
    # --- validate ---
    validate=dict(
        plans=[
            dict(method="qDRIFT"),
            dict(method="graph", k_div=4),
            dict(method="graph", k_div=10),
            dict(method="graph", k_div=30),
            dict(method="random", k_div=10),
        ],
        r=[200, 1000],  # explicit step counts, or ...
        target=None,  # ... a list of target predicted infidelities: r = tau^2 Ebar / target (rounded up to 100)
        ntraj=200,
    ),
    # --- planners / pairs / sampling / freeblocks ---
    ms_aux=[8, 16, 32],  # target block sizes of the side experiments (k = ceil(L/m))
    eps=1e-2,  # target infidelity of the cost-based side experiments (pairs, freeblocks)
)

# ---------------------------------------------------------------- registries
HAMILTONIANS = {"A": model_A, "B": model_B}


def register_hamiltonian(name, fn):
    """fn(**params) -> core.Ham; a case selects it with hamiltonian={'model': name, ...}"""
    HAMILTONIANS[name] = fn


def build_ham(spec):
    spec = dict(spec)
    return HAMILTONIANS[spec.pop("model")](**spec)


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = (
            _merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else copy.deepcopy(v)
        )
    return out


def sweep(template, **axes):
    """Cartesian product of `axes` into a list of cases.

    Values like '{n}' in string fields of `template` are formatted per point, e.g.
    sweep(dict(tag='A n={n}', hamiltonian=dict(model='A', n='{n}')), n=[6, 8])
    (a field that is exactly '{n}' keeps the axis value's type)."""
    import itertools

    def fill(x, pt):
        if isinstance(x, dict):
            return {k: fill(v, pt) for k, v in x.items()}
        if isinstance(x, list):
            return [fill(v, pt) for v in x]
        if isinstance(x, str) and "{" in x:
            for k, v in pt.items():
                if x == "{%s}" % k:
                    return v
            return x.format(**pt)
        return x

    keys = list(axes)
    return [
        fill(template, dict(zip(keys, vals)))
        for vals in itertools.product(*axes.values())
    ]


# ---------------------------------------------------------------- tasks
def _plan_blocks(method, a, K0, Kb, k, rng, sweeps):
    L = len(a)
    if method == "qDRIFT":
        return np.arange(L)
    if method == "graph":
        return anticluster(a, K0, k, sweeps=sweeps)
    if method == "oracle":
        return anticluster(a, Kb, k, sweeps=sweeps)
    if method == "random":
        return random_labels(L, k, rng)
    raise ValueError(method)


def _kgrid(L, ms):
    return sorted({int(np.ceil(L / m)) for m in ms if m <= L / 2} | {2})


def study(c, rng, arrays):
    H = build_ham(c["hamiltonian"])
    core.COST.update(c["cost"])
    psi0 = init_state(H.n, c["init"]["kind"], c["init"].get("seed", 0))
    tau = c["tau"]
    t0 = time.time()
    ts, Ps = trajectory(H, psi0, tau, c["nt"])
    K0, Kb, Ks = graphs(
        H, Ps, keep_all=bool(c["drift_k_frac"])
    )  # streaming (3 L x L matrices) unless drift is tracked
    del Ps
    a, L = H.a, H.L
    q = [[g] for g in range(L)]
    rec = dict(
        tag=c["tag"],
        n=H.n,
        L=L,
        tau=tau,
        lam=H.lam,
        E1_0=price_blocks(q, a, K0),
        E1=price_blocks(q, a, Kb),
        GamL_T=gamma(L, "T"),
        GamL_g=gamma(L, "gate"),
        G1_T=gamma(1, "T"),
        G1_g=gamma(1, "gate"),
        plans={},
    )
    labels = {}

    def add(name, k, blocks, lab=None):
        rec["plans"].setdefault(name, []).append(
            dict(
                k=k,
                E0=price_blocks(blocks, a, K0),
                Eb=price_blocks(blocks, a, Kb),
                G_T=Gam(blocks, a, "T"),
                G_g=Gam(blocks, a, "gate"),
            )
        )
        if lab is not None:
            labels.setdefault(name, []).append(lab)

    plans, sw = c["plans"], c["sweeps"]
    for k in _kgrid(L, c["ms"]):
        for name in ("graph", "oracle"):
            if name in plans:
                lab = _plan_blocks(name, a, K0, Kb, k, rng, sw)
                add(name, k, labels_to_blocks(lab, k), lab)
        # control planners (optional, e.g. plans=[..., "weights", "variances"]): the same swap heuristic on a
        # modified graph, priced on the true graphs. "weights": identity (state-independent), "variances": diagonal of K0
        for name, KK in (
            ("weights", lambda: np.eye(L)),
            ("variances", lambda: np.diag(np.diag(K0))),
        ):
            if name in plans:
                lab = anticluster(a, KK(), k, sweeps=sw)
                add(name, k, labels_to_blocks(lab, k), lab)
        if "random" in plans:
            for _ in range(c["nrand"]):
                lab = _plan_blocks("random", a, K0, Kb, k, rng, sw)
                add("random", k, labels_to_blocks(lab, k), lab)
    if "magnitude" in plans:
        for m in sorted({2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, L // 2}):
            if m < L:
                add("magnitude", L - m + 1, magnitude_blocks(a, m))
    if c["drift_k_frac"]:
        k = min(_kgrid(L, c["ms"]), key=lambda x: abs(x - int(L * c["drift_k_frac"])))
        lab = anticluster(a, K0, k, sweeps=sw)
        rl = random_labels(L, k, rng)
        E1t = np.array([price_blocks(q, a, K) for K in Ks])
        rec["drift"] = dict(
            k=k,
            ts=list(ts),
            E1=list(E1t),
            graph=[price(lab, a, K, k) / e for K, e in zip(Ks, E1t)],
            random=[price(rl, a, K, k) / e for K, e in zip(Ks, E1t)],
        )
    # saved arrays: block labels per plan (so the exact partitions can be reused), optionally graphs
    for name, ls in labels.items():
        arrays[f"labels_{name}"] = np.array(ls)
    arrays["a"], arrays["ts"] = a, ts
    if c["save_graphs"]:
        arrays["K0"], arrays["Kbar"] = K0, Kb
    rec["time"] = time.time() - t0
    return rec


def validate(c, rng, arrays):
    H = build_ham(c["hamiltonian"])
    core.COST.update(c["cost"])
    psi0 = init_state(H.n, c["init"]["kind"], c["init"].get("seed", 0))
    tau = c["tau"]
    ts, Ps = trajectory(H, psi0, tau, c["nt"])
    K0, Kb, _ = graphs(H, Ps)
    a, L = H.a, H.L
    phi = Ps[-1]
    v = c["validate"]
    out = []
    for p in v["plans"]:
        k = L if p["method"] == "qDRIFT" else max(2, L // p["k_div"])
        lab = _plan_blocks(p["method"], a, K0, Kb, k, rng, c["sweeps"])
        bl = labels_to_blocks(lab, k)
        Eb = price_blocks(bl, a, Kb)
        tg = p.get("target", v.get("target"))
        rs = (
            [int(np.ceil(tau**2 * Eb / x / 100.0)) * 100 for x in tg]
            if tg
            else p.get("r", v["r"])
        )
        for r in rs:  # a plan may bring its own step counts and trajectory number
            m, se = mc_infid(
                H, bl, tau, r, psi0, phi, p.get("ntraj", v["ntraj"]), seed=r
            )
            out.append(
                dict(
                    tag=c["tag"],
                    plan=p["method"],
                    k=k,
                    r=r,
                    mc=m,
                    se=se,
                    pred=tau**2 * Eb / r,
                )
            )
            print("  ", out[-1], flush=True)
    return out


def _setup(c):
    H = build_ham(c["hamiltonian"])
    core.COST.update(c["cost"])
    return H, init_state(H.n, c["init"]["kind"], c["init"].get("seed", 0))


def planners(c, rng, arrays):
    H, psi0 = _setup(c)
    return dict(
        tag=c["tag"],
        tau=c["tau"],
        **experiments.planners_exp(H, psi0, c["ms_aux"], rng, c["sweeps"]),
    )


def pairs(c, rng, arrays):
    H, psi0 = _setup(c)
    return dict(
        tag=c["tag"],
        tau=c["tau"],
        **experiments.pairs_exp(H, psi0, c["tau"], c["nt"], c["eps"], rng),
    )


def sampling(c, rng, arrays):
    H, psi0 = _setup(c)
    return dict(
        tag=c["tag"],
        tau=c["tau"],
        **experiments.sampling_exp(H, psi0, c["tau"], c["nt"], c["ms_aux"], rng),
    )


def freeblocks(c, rng, arrays):
    H, psi0 = _setup(c)
    return dict(
        tag=c["tag"],
        tau=c["tau"],
        **experiments.freeblocks_exp(H, psi0, c["tau"], c["nt"], c["ms_aux"], c["eps"]),
    )


TASKS = dict(
    study=study,
    validate=validate,
    planners=planners,
    pairs=pairs,
    sampling=sampling,
    freeblocks=freeblocks,
)


# ---------------------------------------------------------------- estimates (scheduling hints only)
def estimate(c):
    """Rough peak memory (GB), cores and hours of a resolved case, used to schedule parallel runs and to skip cases that do not
    fit. A case may override any of them with _mem_gb, _cores, _hours. Calibrated on the saved runs (info.json has the measured values).
    """
    h = c["hamiltonian"]
    n = h["n"]
    L = (
        h["L"]
        if "L" in h
        else (3 * n * (n - 1) // 2 + 2 * n if h["model"] == "A" else 0)
    )
    d = 2.0**n
    if c["task"] == "validate":
        mem, hours, cores = (
            3 * d * c["validate"]["ntraj"] * 16 / 1e9 + 2 * L * d * 20 / 1e9 + 0.3,
            0.2 * d / 1024 / 4,
            1,
        )
    else:
        nx_ = (
            n * (n + 1) // 2 if h["model"] == "A" else L
        )  # distinct flip patterns: H-bar has one block of 2^n entries each
        hbar = 2 * min(nx_, d) * d * 20 / 1e9  # sparse H-bar, peak while summing
        grams = (
            L * d * 16 / 1e9 + (4 if c["task"] == "study" else 6) * L * L * 8 / 1e9
        )  # state vectors + L x L matrices
        if c["drift_k_frac"]:
            grams += c["nt"] * L * L * 8 / 1e9
        mem = max(hbar, grams) * 1.15 + 0.3
        hours = (
            (L * d * c["nt"] * 3e-9 + (L / 1000.0) ** 2 * 0.002 * len(c["ms"]))
            / 3600
            * (1 if c["task"] == "study" else 0.5)
        )
        cores = int(min(16, max(1, L // 2048)))
    return dict(
        mem_gb=c.get("_mem_gb", mem),
        hours=c.get("_hours", hours),
        cores=c.get("_cores", cores),
    )


# ---------------------------------------------------------------- driver
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _case_hash(c, cseed):
    c = {
        k: v for k, v in c.items() if not k.startswith("_")
    }  # scheduler hints do not change the result
    blob = json.dumps(dict(c=c, seed=cseed), sort_keys=True, default=_json_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def resolve(config):
    """[(index, resolved case, case seed, hash)] for every case of a config (no computation)"""
    seed = config.get("seed", 0)
    base = _merge(DEFAULTS, config.get("base", {}))
    out = []
    for i, case in enumerate(config.get("cases") or [{}]):
        c = _merge(base, case)
        c.setdefault("tag", f"{config.get('name', 'run')}-{i}")
        cseed = c.get("seed", [seed, i])
        out.append((i, c, cseed, _case_hash(c, cseed)))
    return out


def execute(c, cseed, cdir):
    """Run one case and store it in cdir. result.json is written last and marks the case complete."""
    t = time.time()
    arr = {}
    res = TASKS[c["task"]](c, np.random.default_rng(cseed), arr)
    res = json.loads(json.dumps(res, default=_json_default))
    os.makedirs(cdir, exist_ok=True)
    json.dump(
        dict(case=c, seed=cseed),
        open(os.path.join(cdir, "case.json"), "w"),
        indent=1,
        default=_json_default,
    )
    if arr:
        np.savez_compressed(os.path.join(cdir, "arrays.npz"), **arr)
    info = dict(
        seconds=time.time() - t,
        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1e6,  # Linux: kB
        machine=platform.node(),
        python=platform.python_version(),
        numpy=np.__version__,
        finished=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    json.dump(info, open(os.path.join(cdir, "info.json"), "w"))
    json.dump(res, open(os.path.join(cdir, "result.json"), "w"))
    return res


def _start_worker(i, c, cseed, cdir, cores, logdir):
    """one subprocess per case: a case that dies (e.g. out of memory) cannot take the others with it"""
    os.makedirs(cdir, exist_ok=True)
    json.dump(
        dict(case=c, seed=cseed),
        open(os.path.join(cdir, "pending.json"), "w"),
        default=_json_default,
    )
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = str(cores)
    here = os.path.dirname(os.path.abspath(__file__))
    log = open(os.path.join(logdir, f"{i:04d}_{c['task']}.log"), "w")
    return subprocess.Popen(
        [sys.executable, os.path.join(here, "run.py"), "--worker", cdir],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )


def run(
    config,
    force=False,
    jobs=1,
    cores=None,
    mem_gb=None,
    only=None,
    max_case_gb=None,
    min_case_gb=None,
    dry=False,
    quiet=False,
):
    """Run every case in `config`, save everything, return the run directory.

    Each case is stored once, keyed by a hash of its fully resolved settings and seed, in
    <output_dir>/cases/<hash>/ (case.json, result.json, arrays.npz, info.json). A case that is already there is
    loaded instead of recomputed (force=True recomputes), so an interrupted run is simply started again.
    The run directory <output_dir>/<name>_<stamp>/ records config.json, meta.json and results.json.

    jobs=1 runs the cases one after another in this process. jobs>1 runs up to `jobs` cases at once, each in its own
    process, as long as their estimated cores (of `cores`, default all) and memory (of `mem_gb`) fit; the longest first.
    only: regular expression on the tag (and task) selecting the cases to run. max_case_gb / min_case_gb: skip cases estimated
    above / below it (to split a grid between a small and a large machine).
    dry=True only prints the plan."""
    import re

    config = copy.deepcopy(config)
    name = config.get("name", "run")
    root = config.get("output_dir", "runs")
    items = resolve(config)
    rows = []
    for i, c, cseed, h in items:
        rfile = os.path.join(root, "cases", h, "result.json")
        e = estimate(c)
        rows.append(
            dict(
                i=i,
                c=c,
                seed=cseed,
                hash=h,
                est=e,
                cached=os.path.exists(rfile) and not force,
            )
        )
    sel = [
        r
        for r in rows
        if not (only and not re.search(only, f"{r['c']['task']} {r['c']['tag']}"))
    ]
    skipped = [
        r
        for r in sel
        if not r["cached"]
        and (
            (max_case_gb and r["est"]["mem_gb"] > max_case_gb)
            or (min_case_gb and r["est"]["mem_gb"] <= min_case_gb)
        )
    ]
    todo = [r for r in sel if not r["cached"] and r not in skipped]
    ncores = cores or os.cpu_count() or 1
    if mem_gb:
        too_big = [r for r in todo if r["est"]["mem_gb"] > mem_gb]
        for r in too_big:
            print(
                f"WARNING: {r['c']['task']} {r['c']['tag']} needs about {r['est']['mem_gb']:.0f} GB > {mem_gb} GB; skipped"
            )
        skipped += too_big
        todo = [r for r in todo if r not in too_big]
    print(
        f"{len(rows)} cases in config, {len(sel)} selected, {sum(r['cached'] for r in sel)} already saved, {len(skipped)} skipped (memory filters), {len(todo)} to run"
    )
    if dry:
        for r in sel:
            e = r["est"]
            flag = "saved" if r["cached"] else ("SKIP" if r in skipped else "run")
            print(
                f"  {r['i']:4d} {flag:5s} {r['c']['task']:10s} {r['c']['tag']:34s} ~{e['mem_gb']:6.1f} GB  ~{e['hours']:5.2f} h  {e['cores']:2d} cores"
            )
        t = sum(r["est"]["hours"] * r["est"]["cores"] for r in todo)
        print(
            f"  to run: peak ~{max([r['est']['mem_gb'] for r in todo] or [0]):.0f} GB in one case, ~{t:.1f} core-hours (~{t / ncores:.1f} h on {ncores} cores at best)"
        )
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = os.path.join(root, f"{name}_{stamp}")
    logdir = os.path.join(out_dir, "logs")
    os.makedirs(logdir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(out_dir, "config.json"), "w"),
        indent=1,
        default=_json_default,
    )
    t_start = time.time()
    status = {
        r["i"]: ("cached" if r["cached"] else "skipped" if r in skipped else "pending")
        for r in sel
    }
    seconds = {r["i"]: 0.0 for r in sel}
    errors = {}

    def write_results():
        recs = []
        for r in sel:
            rf = os.path.join(root, "cases", r["hash"], "result.json")
            if os.path.exists(rf):
                res = json.load(open(rf))
                recs += res if isinstance(res, list) else [res]
        json.dump(dict(records=recs), open(os.path.join(out_dir, "results.json"), "w"))

    def say(r, text):
        n_done = sum(v in ("cached", "done") for v in status.values())
        if not quiet:
            print(
                f"[{n_done}/{len(sel)}] {r['c']['task']} {r['c']['tag']} {text}",
                flush=True,
            )

    for r in sel:
        if r["cached"]:
            say(r, "cached")
    if jobs <= 1:
        for r in todo:
            t = time.time()
            try:
                execute(r["c"], r["seed"], os.path.join(root, "cases", r["hash"]))
                status[r["i"]] = "done"
            except (
                Exception
            ) as ex:  # keep going: the rest of the grid should not be lost to one case
                status[r["i"]], errors[r["i"]] = "failed", repr(ex)
            seconds[r["i"]] = time.time() - t
            say(r, f"{status[r['i']]} in {seconds[r['i']]:.0f}s")
            write_results()
    else:
        pending = sorted(todo, key=lambda r: -r["est"]["hours"] * r["est"]["cores"])
        running = {}  # index -> (row, Popen, start, cores, mem)
        while pending or running:
            used_c = sum(v[3] for v in running.values())
            used_m = sum(v[4] for v in running.values())
            started = False
            for r in list(pending):
                nc = min(r["est"]["cores"], ncores)
                if (
                    len(running) < jobs
                    and used_c + nc <= ncores
                    and (not mem_gb or used_m + r["est"]["mem_gb"] <= mem_gb)
                ):
                    cdir = os.path.join(root, "cases", r["hash"])
                    running[r["i"]] = (
                        r,
                        _start_worker(r["i"], r["c"], r["seed"], cdir, nc, logdir),
                        time.time(),
                        nc,
                        r["est"]["mem_gb"],
                    )
                    status[r["i"]] = "running"
                    pending.remove(r)
                    used_c, used_m, started = (
                        used_c + nc,
                        used_m + r["est"]["mem_gb"],
                        True,
                    )
                    say(r, f"started (~{r['est']['mem_gb']:.1f} GB, {nc} cores)")
            for i, (r, p, t0, nc, m) in list(running.items()):
                if p.poll() is not None:
                    ok = p.returncode == 0 and os.path.exists(
                        os.path.join(root, "cases", r["hash"], "result.json")
                    )
                    status[i] = "done" if ok else "failed"
                    if not ok:
                        errors[i] = f"worker exit code {p.returncode} (see logs/)"
                    seconds[i] = time.time() - t0
                    del running[i]
                    say(r, f"{status[i]} in {seconds[i]:.0f}s")
                    write_results()
            if not started:
                time.sleep(2.0)
    write_results()
    meta = dict(
        started=stamp,
        seconds=time.time() - t_start,
        python=platform.python_version(),
        numpy=np.__version__,
        machine=platform.node(),
        cases=[
            dict(
                index=r["i"],
                tag=r["c"]["tag"],
                task=r["c"]["task"],
                seed=r["seed"],
                hash=r["hash"],
                status=status[r["i"]],
                seconds=seconds[r["i"]],
                estimate=r["est"],
                error=errors.get(r["i"]),
            )
            for r in sel
        ],
    )
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    bad = [r for r in sel if status[r["i"]] == "failed"]
    if bad:
        print(
            f"{len(bad)} cases FAILED:",
            *[f"{r['c']['task']} {r['c']['tag']}: {errors[r['i']]}" for r in bad],
            sep="\n  ",
        )
    return out_dir


def load(path):
    """Records of a run directory, a results.json, or a legacy list-of-records json."""
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    d = json.load(open(path))
    return d["records"] if isinstance(d, dict) and "records" in d else d


def load_cases(root="runs", task="study", tau=100.0, hashes=None):
    """All saved cases of a given task under <root>/cases/, regardless of which run produced them.
    task='validate' -> list of records; every other task -> {tag: record}. Study records also carry the Hamiltonian and
    initial-state settings and the measured seconds / peak memory. hashes: only these cases (e.g. those of one config).
    The figures and tables are built from this."""
    out = [] if task == "validate" else {}
    cdir = os.path.join(root, "cases")
    for h in sorted(hashes if hashes is not None else os.listdir(cdir)):
        rf, cf = os.path.join(cdir, h, "result.json"), os.path.join(
            cdir, h, "case.json"
        )
        if not (os.path.exists(rf) and os.path.exists(cf)):
            continue
        c = json.load(open(cf))["case"]
        if c["task"] != task:
            continue
        res = json.load(open(rf))
        if task == "validate":
            out += res
            continue
        if tau is not None and abs(res.get("tau", tau) - tau) > 1e-9:
            continue  # e.g. the smoke-test cases (tau=20) share tags with the paper cases
        res["hamiltonian"], res["init"], res["case_hash"] = (
            c["hamiltonian"],
            c["init"],
            h,
        )
        ip = os.path.join(cdir, h, "info.json")
        if os.path.exists(ip):
            res["info"] = json.load(open(ip))
        out[res["tag"]] = res
    return out
