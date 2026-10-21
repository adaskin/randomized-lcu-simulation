# Randomized Hamiltonian simulation: grouped randomized simulation between qDRIFT and LCU

Code and data for the paper ***Which terms to keep coherent? Grouped randomized Hamiltonian simulation between qDRIFT and LCU* (Ammar Daskin).**
[https://doi.org/10.5281/zenodo.23535793](https://doi.org/10.5281/zenodo.23535793)


State-vector experiments (up to 20 qubits / 32768 Pauli terms): correlation-graph block planning vs qDRIFT / full-H / random / magnitude baselines. The manuscript itself is not part of this repository. `pip install -r requirements.txt`.

## One grid, one script
Every simulation of the paper is a case of one config, `configs/grid.py`, run by `run.py`:

```
python run.py configs/grid.py --dry                         # the plan: cases, estimated memory and hours, what is saved already
python run.py configs/grid.py --jobs 8 --mem-gb 80         # everything (resumable: saved cases are skipped)
python run.py configs/grid.py --max-case-gb 8              # only what fits a small machine
python run.py configs/grid.py --min-case-gb 8 --jobs 3 --mem-gb 75   # the other half
python run.py configs/grid.py --only "^validate" --jobs 5  # a subset: regular expression on "<task> <tag>"
```

| axis | values |
|---|---|
| qubits `n` | 5, 10, 15, 20 |
| model B: terms `L` | `L = 2^j`, from 16 up to the largest that fits: 256 (n=5), 8192 (n=10), 32768 (n=15), 1024 (n=20) |
| model A: terms | fixed by `n`: `L = 3n(n-1)/2 + 2n` = 40, 155, 345, 610 |
| instances | model B: 3 random instances (`seed` 0, 1, 2) for `L <= 4096`, one above |
| settings | identical everywhere: `tau = 100`, 11 time points, Neel state, cost model R=40 / kappa=2 / c=5, block sizes m = 2, 4, ..., 256 (k = ceil(L/m)), 3 random partitions per k, 4 swap sweeps, planners graph / oracle / random / magnitude / weights / variances |

The largest `L` per `n` is set by the number of distinct Pauli strings of weight 2-4 (n=5, 10: at most half of them), and by memory
(n=15: three `L x L` graphs and `L` vectors of dimension `2^n`, about 45 GB; n=20: the sparse mean Hamiltonian has one block of `2^20`
entries per flip pattern). Everything beyond these sizes is extrapolation and is labelled as such in the paper.

Besides the main `study` of every (n, L), the same config holds the side experiments (planners, blocks of two, block probabilities,
unequal block sizes) on a subset of the grid, the robustness runs (random-product and Hartree-Fock-like initial states) and the
Monte Carlo check of the error formula (n = 5, 10, 15 at tau = 8; n = 5, 10 at tau = 100). The check at n = 15, tau = 100 was started and
stopped after about 16 h (projected 1.5-3 days); it is not part of the grid and is stated as such in the limitations of the paper.

## Layout
- `core.py`        Pauli-term Hamiltonians, trajectories, term correlation graph, anticlustering planner, Monte Carlo of the sampled channel
- `sim.py`         the runner: `run(config)`; tasks `study`, `validate`, `planners`, `pairs`, `sampling`, `freeblocks`; cache, scheduler
- `experiments.py` the side experiments (called by the tasks of `sim.py`)
- `run.py`         command line (flags above)
- `configs/grid.py` the grid; `configs/smoke.py` a seconds-long template for new experiments
- `analysis.py`    cost-to-accuracy, noise and break-even analysis from saved cases (shared by the scripts below)
- `plot.py`, `toy_fig.py`   every figure of the paper (`figstyle.py` is the shared style)
- `paper_tables.py`, `paper_numbers.py`   every data table of the paper (written to `tables/table_*.tex`), and every number quoted in the numerical section
- `control_fig.py` the control figure (graph vs oracle vs weights-only planning)

## Saved data
`runs/cases/<hash>/` holds each case once: `case.json` (resolved settings and seed), `result.json`, `arrays.npz` (block labels of every
plan), `info.json` (seconds, peak memory, machine). The hash covers all settings and the seed (keys starting with `_` are scheduler
hints and are not hashed), so a case computed anywhere is loaded, not recomputed, wherever `runs/` is present; commit `runs/` to share.
`runs/<name>_<timestamp>/` has `config.json`, `meta.json` (per-case status, hashes, timings) and `results.json`. Interrupted runs are
resumed by starting the same command again; a case that dies (for example out of memory) is reported at the end and retried then.

## Reproducing the paper
```
python run.py configs/grid.py --jobs 8 --mem-gb 80    # hours on a 16-core, 96 GB machine; or use the saved runs/
python plot.py --out figs                              # every figure from the saved cases (also: python toy_fig.py)
python control_fig.py --out figs                       # the control figure
python paper_tables.py                                 # tables/table_*.tex
python paper_numbers.py                                # every number quoted in the numerical section
```
No figure, table or number needs a simulation: all are computed from `runs/cases/`.

## Where each item of the paper comes from
This table maps the figures and tables of the paper to the code.

| Paper item | Produced by | Cases |
|---|---|---|
| Fig. 1 (schematic), Fig. 2 (circuit) | TikZ drawings in the manuscript | none |
| Fig. 3 (six-term toy example) | `toy_fig.py` | analytic, 3 qubits |
| Figs. 4-6 (cost vs accuracy, scaling, noise) | `plot.py` | `study` |
| Fig. 7 (cost vs number of blocks), Fig. 8 (block size vs L) | `plot.py` | `study` |
| Fig. 9 (graph vs oracle vs weights-only planning) | `control_fig.py` | `study` (control planners) |
| Fig. 10 (appendix: error formula vs Monte Carlo) | `plot.py` | `validate` |
| Fig. 11 (appendix: grouping fixed at t=0) | `plot.py` | `study` with drift (L <= 4096) |
| Table C1 (alternative planners) | `paper_tables.py` | `planners` |
| Tables C2 (costs), C3 (n- and L-dependence), C4 (break-even) | `paper_tables.py` | `study` |
| Table C5 (equal vs unequal block sizes) | `paper_tables.py` | `freeblocks` |
| Table C6 (blocks of two) | `paper_tables.py` | `pairs` |
| Tables C7, C8 (planning on graph / variances / weights; cost-error product with weights only) | `paper_tables.py` | `study` (control planners) |
| Table C9 (block probabilities) | `paper_tables.py` | `sampling` |

All tables are in Appendix C of the paper.


`sim.load_cases` keeps only cases with `tau = 100` for the study tasks, so seconds-long smoke-test cases never enter a figure or table;
`analysis.grid_cases` further restricts to the cases of `configs/grid.py`.
