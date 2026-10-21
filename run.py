"""python run.py CONFIG [options]

CONFIG is a .py file defining `config` (a dict) or a .json file (e.g. a saved <run>/config.json).

    python run.py configs/grid.py --dry                       what would run (estimated memory and hours), what is saved already
    python run.py configs/grid.py --jobs 8 --mem-gb 80        the full grid, 8 cases at a time, within 80 GB
    python run.py configs/grid.py --max-case-gb 8             only the cases that fit a small machine
    python run.py configs/grid.py --only "study B n=15"       a subset (regular expression on "<task> <tag>")

Every case is saved once under <output_dir>/cases/<hash>/ and skipped when it is already there, so an interrupted run is
resumed by starting it again, and a case computed on one machine is not recomputed on another once runs/ is shared.

--set KEY=JSON overrides one entry of the config before running, e.g.  --set base.tau=50  --set seed=3
--out DIR       output_dir (parent of the run directory and of cases/)
--force         recompute the selected cases even when saved
"""
import argparse
import importlib.util
import json
import os
import sys

from sim import execute, run


def load_config(path):
    if path.endswith(".json"):
        return json.load(open(path))
    spec = importlib.util.spec_from_file_location("cfg", path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(m)
    return m.config


def set_path(cfg, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, {})
    cfg[keys[-1]] = value


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--out", help="output_dir (parent of the run directory)")
    ap.add_argument("--jobs", type=int, default=1, help="cases to run at once, each in its own process (default 1: in this process)")
    ap.add_argument("--cores", type=int, help="cores the scheduler may use (default: all)")
    ap.add_argument("--mem-gb", type=float, help="memory the scheduler may use in total; also skips any single case estimated above it")
    ap.add_argument("--max-case-gb", type=float, help="skip cases whose estimated memory is above this")
    ap.add_argument("--min-case-gb", type=float, help="skip cases whose estimated memory is at most this (the big-machine half of a split grid)")
    ap.add_argument("--only", help="regular expression on '<task> <tag>': run only the matching cases")
    ap.add_argument("--dry", action="store_true", help="print the plan (estimates, saved or not) and exit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--worker", metavar="CASE_DIR", help=argparse.SUPPRESS)  # internal: run one pending case
    a = ap.parse_args()
    if a.worker:
        d = json.load(open(os.path.join(a.worker, "pending.json")))
        execute(d["case"], d["seed"], a.worker)
        os.remove(os.path.join(a.worker, "pending.json"))
        sys.exit(0)
    cfg = load_config(a.config)
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        set_path(cfg, k, v)
    if a.out:
        cfg["output_dir"] = a.out
    out = run(cfg, force=a.force, jobs=a.jobs, cores=a.cores, mem_gb=a.mem_gb, only=a.only, max_case_gb=a.max_case_gb, min_case_gb=a.min_case_gb, dry=a.dry)
    if out:
        print("saved to", out)
