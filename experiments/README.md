# Experiments

Grouped by where the rates come from. Studies the manuscript no longer uses live in
`archive/experiments/`.

| path | data | what it answers |
|---|---|---|
| `synthetic/verification/` | generated from (2) | correctness gate and convergence rates; the model is right by construction |
| `synthetic/shape-scarcity/` | generated from (2) | when the exact criterion separates from a two-moment surrogate |
| `synthetic/nmc-cost/` | generated from (2) | what nested Monte Carlo costs against the quadratures |
| `bulk-sampling/` | alluvial diamond simulator | allocating a processing budget over blocks |
| `gamma-ray/` | Fermi-LAT 4FGL | scheduling telescope time over source classes |
| `hard-xray/` | Swift-BAT | also schedules telescope time over patches of sky | 
| `common/` | — | `viz.py`, the figure style every study shares |

## Conventions

Each study directory is self-contained: `run.py` (or `compare.py`) writes `results/`,
`visualize.py` writes `figures/`, and a `README.md` says what the study found. Everything is
reproducible from seed 0. A study never writes into another study's directory.

`common/viz.py` used to live inside the verification study, which meant every other
visualizer reached into it by relative path. It is now a peer of the studies that use it.