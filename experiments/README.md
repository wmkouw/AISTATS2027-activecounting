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
| `rate-fields/` | Fermi-LAT + Safecast | which mixing law the measured physical fields need |
| `newsgroups/` | 20 Newsgroups | the linguistic control on that claim |
| `exceedance/` | generated from (2) | whether the third parameter pays when the decision is a threshold |
| `bo-baseline/` | shape-scarcity + Fermi-LAT | Bayesian optimisation on a discrete-output GP, as a baseline |
| `common/` | — | `viz.py`, the figure style every study shares |
| `tables/` | — | LaTeX fragments the manuscript `\input`s |

`make_tables.py` regenerates every table from the recorded CSVs; `check_manuscript.py`
asserts the manuscript's labels, graphics and inputs still resolve.

## Conventions

Each study directory is self-contained: `run.py` (or `compare.py`) writes `results/`,
`visualize.py` writes `figures/`, and a `README.md` says what the study found. Everything is
reproducible from seed 0. A study never writes into another study's directory.

`common/viz.py` used to live inside the verification study, which meant every other
visualizer reached into it by relative path. It is now a peer of the studies that use it.

## Cross-study imports

Three, all one-directional and all read-only:

- `rate-fields/fields.py` imports the catalogue loader from `gamma-ray/fetch.py`
- `newsgroups/` imports the laws and fitting loop from `rate-fields/compare.py`
- `bo-baseline/` imports the environments from `synthetic/shape-scarcity/` and `gamma-ray/`

## What is archived, and why

`archive/experiments/tail-mass/` — produced Table 1. The manuscript no longer `\input`s it
and the paragraph that referenced it is commented out at `main.tex:218`. The table builder in
`make_tables.py` is kept but runs only if the directory is moved back.

`archive/experiments/safecast/` — a standalone radiation study. Nothing in the manuscript or
in any other study depends on it; `rate-fields` loads Safecast through its own cache.

Both are plain directory moves and reverse by moving them back.

## Manuscript paths

The regrouping changed six `\includegraphics` paths. `main.tex` has **not** been edited:

```
experiments/experiment-gigpoisson/figures/prediction
  -> experiments/synthetic/verification/figures/prediction
experiments/experiment-shape-scarcity/figures/shape_scarcity
  -> experiments/synthetic/shape-scarcity/figures/shape_scarcity
experiments/experiment-bulk-sampling/figures/scarcity
  -> experiments/bulk-sampling/figures/scarcity
experiments/experiment-bulk-sampling/figures/bulk
  -> experiments/bulk-sampling/figures/bulk
experiments/experiment-photon-counting/figures/photon
  -> experiments/gamma-ray/figures/photon
experiments/experiment-rate-fields/figures/rate_fields
  -> experiments/rate-fields/figures/rate_fields
```

`\input{experiments/tables/...}` is unchanged. `check_manuscript.py` already expects the new
paths, so it will report these six until the manuscript is updated.
