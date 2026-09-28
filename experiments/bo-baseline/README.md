# Bayesian optimisation on a discrete-output Gaussian process

The baseline the comparison was missing on the model axis. Every mixing law in
`methods/countmodels.py` treats the contexts as independent. This one does not.

## The surrogate

`methods/gppoisson.py`. A Gaussian process over log-rates with a Poisson observation model:

```
g ~ GP(m, K)        latent field over the K contexts
lam_k = exp(g_k)    rate of a context
y | k, e ~ Poisson(e * exp(g_k))
```

The output is discrete, so the posterior is not Gaussian and is approximated by Laplace —
Newton to the mode, curvature `W = diag(F_k exp(g_k))` there, and the stable
`B = I + W^{1/2} K W^{1/2}` factorisation of Rasmussen and Williams Algorithm 3.1 rather than
an inverse of `K`. The log posterior is concave, so the mode is unique.

Summing Poissons of a common rate, a context's likelihood depends on its history only through
the accumulated count `T_k` and accumulated exposure `F_k` — the same pair Corollary 1 gives
the conjugate model. The two carry identical state and differ only in what they do with it.

## Two things were needed to make the comparison fair

**The prior marginals are pinned.** A first version put a single amplitude on the kernel, so
each context's prior variance became the average across contexts rather than its own, and
contexts sharing a class collapsed onto one rate. The kernel is now built as a *correlation*
scaled by the per-context prior standard deviation, so the marginal prior on every rate is
exactly the one every other family is handed and the kernel contributes only the correlation.

**The correlation is learned, not assumed.** Lengthscale and nugget are refit by the Laplace
marginal likelihood on a doubling schedule. Fixing them would have let the choice of
hyperparameter decide the result.

**The population fit is shared.** Where an environment fits its own priors — the gamma-ray
study fits each family to the population of the source's class — this model takes that fit
through the lognormal member and converts to moments, rather than falling back to the two
moments and starting behind for no reason.

## The kernel

Over the context features the study already provides. The default feature map is the log
prior mean and log prior index of dispersion, standardised, which is available in every study
and encodes exactly the structure that matters: in the gamma-ray campaign every source of a
class is given its class prior, so sources of a class sit at the same feature point and the
kernel pools them. An environment may override with `context_features`.

## The acquisitions

Bayesian optimisation ranks points, and a point here is a pair — which context, and how much
of it. Expected improvement scores the context and is silent about the exposure, so the
exposure enters as a fidelity:

```
acq(k, v) = EI(k) * rho(k, v),   rho = s_k^2 e lam_k / (1 + s_k^2 e lam_k)
```

`rho` is the fraction of the latent variance a Poisson observation of exposure `e` is
expected to retire, since its Fisher information about the latent is `e lam_k`. The study
divides by cost afterwards, as it does for every criterion. `bo-ucb` replaces EI with an
upper confidence bound at `beta` posterior standard deviations.

**Read these against the regret column first.** Expected improvement is built to find the
largest rate, not to predict counts well everywhere, so losing on held-out predictive score
to criteria that spread is the expected behaviour and not a defect.

## What this does not touch

`agents.build_all()` returns exactly what it returned before: the two agents are opt-in via
`build_all(include_bo=True)`, so every existing study writes what it wrote before. This
directory keeps its own `results/` and imports the two environments read-only; nothing here
regenerates `experiments/tables/`, which the manuscript inputs.

## Running

```bash
cd experiments/bo-baseline
python run.py both     # shape: 60 episodes; photon: 12 programmes; ~17 min
```

Reproducible from seed 0. Results land in `results/episodes.csv` and `results/summary.csv`.
There is no figure yet: the result is a table of the same shape as Tables 2 and 4, and it has
not been decided where or whether it goes in the paper.

## What it found

Gamma-ray campaign, 12 programmes, against the proposed criterion:

| agent | NLPD | t | dex | t | top-4 regret | t | ms |
|---|---|---|---|---|---|---|---|
| EIG (ours) | 3.3739 | — | 0.1311 | — | 0.0093 | — | 17.5 |
| BO (UCB) | 3.4582 | −5.5 | 0.1718 | −11.8 | **0.0043** | +1.1 | **0.49** |
| BO (EI) | 3.9252 | −11.5 | 0.2753 | −14.6 | 0.0445 | −2.5 | 6.95 |

The upper confidence bound halves the point estimate of top-4 regret at a thirty-sixth of the
cost per decision, while losing decisively on held-out predictive score and on the rate
estimate. That is the trade-off the design predicts: it optimises for the largest rate and
does not spread. The regret difference is within noise at twelve programmes (`t = +1.1`), so
the honest reading is a tie on regret and a clear loss elsewhere.

Expected improvement is worse than the upper confidence bound on every column here. It
concentrates too early: with 48 targets and most of them unobservable, the incumbent is set
by a handful of counts and EI stops exploring.

On the synthetic two-class design, where the budget is scarcer relative to the contexts, both
BO agents lose on all three columns (`t` between −4 and −6).

**The harness is anchored.** Run through this module, `eig` returns NLPD 3.3739 and 0.1311 dex
against the 3.373 and 0.131 the manuscript's own photon study reports, so the numbers above
are comparable to Table 4 and are not an artefact of a different loop.
