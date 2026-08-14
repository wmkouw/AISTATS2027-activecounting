# Alluvial diamond bulk sampling

Written 2026-08-08. Reproducible from seed 0: `python run.py` then `python visualize.py`,
about four minutes end to end.

## Why this problem

The setting Sichel invented the GIGP for. Diamonds occur in clusters, so stone counts from
a sample of gravel are far more variable than a Poisson process allows, and no standard
discrete law reproduces observed stone-count frequencies. That model is used to **value** a
deposit once samples are in hand. It is not used to decide **which** samples to take, which
is the gap this study occupies.

Three things line up here that do not line up in word frequencies or photon counting:

1. The model class belongs to the domain, not to us, so "why this distribution" has an
   answer that predates the paper.
2. The exposure action is physical and unambiguous: cubic metres of gravel through the
   plant, times a known per-block plant recovery factor.
3. **The budget is naturally in cubic metres, not in number of samples.** That forces every
   criterion to be scored per unit of gravel. A budget counted in rounds would be won by
   whichever policy asks for the biggest sample, which is a measurement artefact.

**This is a simulator, not a field trial.** Its structure is taken from the domain; no
public dataset is used. Do not let the paper imply otherwise.

## The decision problem

```
lam_k          ~  GIG(alpha, a_k, b_k)     stones per m^3, drawn once per block
u = (k, v)                                 block, and volume of gravel to process
f(u) = v * r_k                             effective exposure, r_k the plant recovery
y | lam_k, u   ~  Poisson( f(u) lam_k )    stones recovered
cost(u) = v                                budget spent, in m^3
```

12 blocks, volumes `{1, 2.5, 5, 10, 20, 40}` m^3, budget 240 m^3, 48 replicate properties.
Blocks differ in expected grade (0.05 to 4 stones/m^3), in how clustered the stones are at
that grade (`omega` in 0.3 to 2.0, prior variance-to-mean 2 to 13), and in plant recovery
(0.55 to 0.98).

Common random numbers are exact rather than approximate: each block's stone field is
realised once as a Poisson process in effective processed volume, and a sample consumes the
next `v r_k` metres of it. Two policies that process the same gravel recover the same
stones, so any difference between them is a difference in decisions.

## What is scored

Three things, because they are not the same thing and a policy can lead on one and trail on
another:

- **NLPD** of held-out stone counts. Needs no true grade.
- **grade RMSE** over blocks. What a resource statement needs.
- **top-4 regret**: grade forgone by mining the four blocks the belief ranks highest rather
  than the four best. What the mine plan needs.

## Agents: two axes, crossed

An agent is a **model** plus an **acquisition criterion**. `methods/countmodels.py` holds
the models, `agents/` holds the criteria, and `run.py` crosses them, so the two can be varied
independently rather than one at a time.

| mixing law | params | conjugate | predictive | ms/decision |
|---|---|---|---|---|
| GIG-Poisson (ours) | 3 | yes | Sichel | 8.9 |
| gamma-Poisson | 2 | yes | negative binomial | 0.8 |
| lognormal-Poisson | 2 | **no** | grid quadrature | 21 |

| criterion | what it scores |
|---|---|
| `eig` | expected information gain about the grade (ours) |
| `epig` | expected predictive information gain; no closed form here |
| `maxent` | maximum entropy sampling, `H[y|u]` |
| `d-optimality` | log Fano factor, the variance-stabilised Fisher form |
| `neyman` | Neyman optimal allocation, share proportional to `sqrt(lam_k / r_k)` |
| `predictive-variance`, `epistemic` | uncertainty sampling, two forms |
| `thompson` | posterior sampling |
| `uniform`, `random` | belief-free designs, run once |

Every model is initialised from the **same prior mean and variance per block** and none is
told the true parameters. Every model returns a pmf on the non-negative integers, so NLPD is
comparable across models. Each model's acquisition is gated against nested Monte Carlo *in
its own family*.

EPIG is not crossed with the lognormal: its outer expectation needs a posterior predictive
entropy per node, and under a grid posterior that is a quadrature inside a quadrature. The
exclusion is a result about non-conjugacy and is reported, not hidden.

### Results, 48 properties

**The criterion effect is invariant to the model.** NLPD relative to EIG under the same
model, in nats:

| criterion | GIG | gamma | lognormal |
|---|---|---|---|
| EPIG | +0.003 | -0.002 | not run |
| D-optimality | +0.001 | +0.001 | -0.003 |
| Neyman | +0.015 | +0.018 | +0.009 |
| epistemic var. | +0.077 | +0.066 | +0.083 |
| total var. | +0.089 | +0.079 | +0.084 |
| Thompson | +0.214 | +0.219 | +0.191 |
| max-entropy | +0.302 | +0.308 | +0.296 |

**The model effect is invariant to the criterion**, and much smaller: the gamma is behind the
GIG by +0.012 to +0.018 nats (t = +3.3 to +3.8) under the criteria that spread; the lognormal
by +0.006 (t = +2.0) under EIG and less elsewhere.

**Choosing the wrong acquisition costs ~25x what choosing the wrong mixing law costs.**

Caveat: the generating process *is* the GIG-Poisson law, so our model is correct by
construction and the model-axis number measures what misspecification costs rather than what
the family is worth in general. The complementary sweep over generating dispersion is not
run.

## Gate

`run.py` checks the closed-form EIG against bias-corrected nested Monte Carlo before
running anything, and aborts on failure.

## Files

| file | what it is |
|---|---|
| `domain.py` | the environment: ground truth, actions, budget, gravel. Holds no model |
| `../../methods/countmodels.py` | the three mixing laws behind one interface |
| `../../agents/` | one module per acquisition criterion |
| `run.py` | gate, sequential study over the criterion-by-model grid |
| `visualize.py` | three panels into `figures/bulk.pdf`, included by the paper |
| `results/sequential.csv` | per policy, property and checkpoint: NLPD, grade RMSE, regret |
| `results/allocation.csv` | volume each policy gave each block, against its true grade |
| `results/gate_eig.csv` | each model's EIG against nested Monte Carlo |

## Notes on the numerics

Two changes were forced by this study and both are in `methods/gigpoisson.py`.

**The Bessel routing.** The order recurrence is cancellation-free but its rounding compounds
along the run: against 30-digit references it is exact for the first few dozen orders, out
by 3e-12 at 10^3, 6e-9 at 5e4 and 3e-7 at 2e5, where the uniform asymptotic expansion is
still exact. It is also a Python loop and an order of magnitude slower there. `sichel_logpmf`
now takes the recurrence below 512 orders and the direct route above.

**The support tolerance.** A heavy-tailed predictive needs a support of order 10^5 to reach
a tail mass of 1e-13, which is far more than an entropy needs. The applied study passes
`tail_tol=1e-9`; the verification study keeps the strict default.
