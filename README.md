# Actively sensing overdispersed count data with Sichel distributed predictions

Paper and experiment code for active sensing under a Poisson likelihood whose rate is
unknown and mixed over a generalised inverse Gaussian law.

A sensor chooses an action `u`, the action sets a known exposure `f(u)`, and the count it
gets back is `y ~ Poisson(f(u) λ)` with `λ ~ GIG(α, a, b)` unknown. That pair is conjugate:
an observation sends

```
GIG(α, a, b)  ->  GIG(α + y, a, b + 2 f(u))
```

so the exposure enters the second scale parameter additively, the family is closed under the
action, and the posterior predictive is the Sichel law. Every information quantity an
acquisition function needs is therefore available in closed form, up to a truncated sum over
the support.

Why this family rather than the gamma-Poisson pair: a gamma mixing law already gives a
negative binomial predictive with a free mean *and* a free variance. The GIG adds a third
parameter, so with the predictive mean and variance both held fixed there is still freedom to
move mass between the centre and the tail. The gamma is the `ω -> 0` boundary of the
parameterisation used here, so the negative binomial sits inside the family as a limiting
case and the distance to it is measured rather than assumed.

## Layout

| path | what it is |
|---|---|
| `main.tex`, `supplement.tex`, `references.bib` | the paper (AISTATS 2026 format, `template/` holds the unmodified style files) |
| `methods/gigpoisson.py` | the mathematical core: GIG mixing law, Sichel predictive, conjugate update, and the Bessel numerics |
| `methods/countmodels.py` | GIG-, gamma- and lognormal-Poisson behind one interface, all initialised from matched prior moments |
| `agents/` | one module per acquisition criterion, written against a small environment protocol |
| `experiments/` | three studies, each self-contained with its own `data/`, `results/` and `figures/` |
| `notes/` | working notes on the Sichel/GIG-Poisson member and its literature |
| `literature/` | reference PDFs |

An **agent** is a model plus a criterion. `agents/__init__.py` crosses the two axes on
purpose: hold the model fixed and vary the criterion to ask whether the acquisition choice
matters; hold the criterion fixed and vary the mixing law to ask whether the modelling choice
matters. Criteria available: expected information gain (ours), EPIG, maximum entropy
sampling, D-optimality, Neyman allocation, predictive- and epistemic-variance sampling,
Thompson sampling, and the belief-free systematic and random designs.

## Experiments

Each has its own README with the numbers; the summaries below are the one-line versions.

**`experiments/experiment-gigpoisson/`** — verification. A photon-counting survey simulated
from the GIG-Poisson law itself, where the model is correctly specified by construction. A
correctness gate checks the Bessel evaluation against 40-digit references, the conjugate
update against numerical Bayes, the predictive against the simulator, and the gamma
boundary, and aborts the study on any failure. `run_prediction.py` measures posterior
predictive error against sample size across four dispersion regimes and recovers the
theoretical `-1`, `-1/2` and `-1` log-log slopes.

**`experiments/experiment-bulk-sampling/`** — application. Alluvial diamond bulk sampling,
the setting Sichel introduced the GIGP for. Twelve blocks, a budget in cubic metres of gravel
rather than in number of samples, so every criterion is scored per unit cost. Scored on
held-out NLPD, grade RMSE and top-4 regret over 48 replicate properties. Headline: choosing
the wrong acquisition costs roughly 25x what choosing the wrong mixing law costs. It is a
simulator whose structure is taken from the domain, not a field trial.

**`experiments/experiment-photon-counting/`** — *in progress.* The same agents on a field of
rates nobody simulated: measured integral photon fluxes from the Fermi-LAT 4FGL catalogue,
so every model in the comparison, ours included, is misspecified. `fetch.py` caches the
catalogue from VizieR, `fit.py` scores each mixing law by AIC, `run.py` runs the sensing
study. Only the mixing-law fit and the EIG gate have been run so far.

## Running

Python 3 with `numpy`, `scipy` and `matplotlib`; `mpmath` additionally for the verification
gate in `experiments/experiment-gigpoisson/dgp.py`. Each study is run from its own directory
and is reproducible from seed 0:

```bash
cd experiments/experiment-gigpoisson && python run.py && python visualize.py
cd experiments/experiment-bulk-sampling && python run.py && python visualize.py
```

Roughly two and four minutes respectively; the correctness gates dominate. Results land in
`results/` as CSV and figures in `figures/`, which the paper includes directly.

## A note on the numerics

`K_v(z)` overflows double precision at moderate order, and the posterior order is
`α + Σy`, which grows without bound as data arrive — following the brightest site at full
dwell, a direct transcription of the closed form overflows at round 4. Everything is
therefore done in log space: `log_besselk` uses the exponentially scaled `kve` below order 45
and Olver's uniform asymptotic expansion above it, and support sums use a cancellation-free
upward recurrence on the order below 512 terms and the direct route above, where the
recurrence's rounding has begun to compound. Analytic tractability and numerical reliability
are different properties.

## License

MIT. See [LICENSE](LICENSE).
