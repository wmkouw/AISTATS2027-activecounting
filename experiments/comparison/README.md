# Head-to-head on the two overdispersed settings

Written 2026-09-29. Six agents, both settings, one harness, one clock.

```bash
python experiments/comparison/run.py both --smoke        # 2 episodes -> results/smoke_*
python experiments/comparison/run.py both --reps 50      # the study
```

Reproducible from seed 0. Episodes run in parallel (`--jobs`, default 20), one
single-threaded process per episode.

## Agents

| name | model | posterior | EIG |
|---|---|---|---|
| `eig` | GIG-Poisson (ours) | conjugate | two quadratures |
| `eig@gamma-poisson` | gamma-Poisson | conjugate | exact sum |
| `eig@lognormal-likelihood` | binned lognormal *likelihood* on counts, lognormal rate | 128-node grid | quadrature over grid |
| `eig@logskewnormal-poisson` | Poisson, log-skew-normal prior on the rate (3 params) | MCMC, 2048 chains × 30 MH steps / update | nested MC, N = M = 1024 |
| `bo-ei@gp-poisson` | GP over log-rates, Poisson output, RBF kernel on prior features | Laplace | — (expected improvement × fidelity) |
| `bo-ei@gp-poisson-cat` | same, categorical kernel on source class | Laplace | — |

The lognormal-*Poisson* (grid mixing law) of the earlier studies is not in this comparison.

**Priors.** Bulk sampling hands every agent two moments per block; the log-skew-normal is
then matched with skew zero, since no agent is given a third moment. Gamma-ray hands every
agent the population of the source's class; each family is fitted to it by maximum
likelihood in its own parameters, the log-skew-normal to the log-fluxes with all three free.

**Categorical kernel.** Correlation `rho` within a class and none across, `rho` learned by
Laplace marginal likelihood on the same doubling schedule as the RBF lengthscale. Bulk
blocks have no class, so there it is the exchangeable kernel: a component shared by every
block plus an independent one.

## Metrics (`experiments/common/harness.py`)

- **NLPD**: average surprisal of 24 held-out counts per context at the evaluation exposure.
- **top-m regret**, m = 4 and 5: `1 − Σ_{k∈S} λ_k / Σ_{k∈S*} λ_k`, S the agent's m contexts
  of highest posterior mean, S* the true top m. Simple (not cumulative) regret at the
  checkpoint, scale-free.
- **time**: wall-clock of `act` (design selection) plus `observe` (belief update), both
  recorded separately, reported per round. Nothing else is timed.
- diagnostics: RMSE and log10 error of the posterior mean rate, fraction of contexts touched.

## Gate

Before a setting runs: GIG, gamma and lognormal-likelihood EIGs against bias-corrected
nested MC in their own families (fail only if |z| > 4 and the gap exceeds 0.02 nats: the
1/M extrapolation of the reference is itself unreliable where the bias is not yet asymptotic,
as for the binned lognormal at 0.5 Ms, whose quadrature value the plain estimate approaches
from below as n grows). For the log-skew-normal, the MCMC posterior mean
after one observation against quadrature (|z| < 6). Its nested-MC EIG is compared against
quadrature too and **recorded but not failed**: at 2048 chains it overshoots by up to 0.44
nats on the unassociated class (fitted skew +9.3, a heavy right tail) at 25 Ms, converging to the quadrature value only at
~32k chains. That bias is the baseline's, and it is reported rather than gated out.

## Files

| file | contents |
|---|---|
| `results/{bulk,gamma}.csv` | per agent, episode, checkpoint |
| `results/summary_{bulk,gamma}.csv` | final budget: mean, sem, paired t vs `eig` (positive = better than `eig`), ms per round/act/observe |
| `results/gate_{bulk,gamma}.csv` | the gate |
| `results/smoke_*` | the same from the 2-episode smoke run |

## Sensitivity: gentler catalogue cut (`--band 1 99`)

The default gamma-ray environment keeps the 5th-95th percentile of flux. Those hard edges
favour the log-skew-normal prior (fitted skews of +8 to +9, and -17.9 for pulsars, whose
upper tail the 95th-percentile cut truncates): it fits the held-out half of every class
better than the GIG. At 1-99 the skews fall to +3.8 to +6.1 and -0.8 for pulsars, and the GIG
fits pulsars better. Outputs carry the prefix `band1-99_`.

| at 120 Ms, 50 episodes | NLPD 5-95 | NLPD 1-99 |
|---|---|---|
| GIG-Poisson + EIG | 3.371 | 3.475 |
| log-skew-normal, MCMC + NMC | 3.365 (t = +1.0) | 3.474 (t = +0.3) |

## Resuming and the GP line search

Every (episode, agent) run is written to `results/partial/<prefix><setting>/` as it
finishes; rerunning the same command skips those. The Laplace Newton iteration in
`methods/gppoisson.py` now takes a line search: on the 1-99 band an undamped step overflowed
during hyperparameter tuning. Rerunning the BO agents on the 5-95 results changed one
episode per setting and no summary figure at the precision reported; the pre-fix files are
in `results/pre_linesearch/`.

## Baselines added 2026-10-01 (15 agents in all)

Registered in `agents.build_comparison()`; the resumable runner computes only what is missing.

| agent | what it is | where |
|---|---|---|
| `systematic` | round-robin at the largest action that lets the budget visit every context once | `agents/systematic.py` |
| `random` | uniform over contexts and actions | `agents/uniform_random.py` |
| `d-optimality` | local D-optimality per unit cost, GIG model | `agents/dopt.py` |
| `thompson` | posterior sampling at the largest action, GIG model | `agents/thompson.py` |
| `lucb`, `lucb@gamma-poisson` | Bayes-LUCB for the top-5 set (posterior quantiles `1 - 1/(t+1)`, floor 0.9), shortest action, under GIG and gamma | `agents/lucb.py` |
| `eig@gammamix-poisson` | finite gamma mixture prior, J by BIC (1-4), conjugate, exact EIG | `methods/gammamixture.py` |
| `eig@logskewnormal-quad` | log-skew-normal prior, posterior and EIG by quadrature (exact, non-conjugate) | `methods/logskewnormal.py` |
| `eig@logskewnormal-poisson-m8k` | the MCMC baseline with N = 1024 outer and M = 8192 inner samples | `methods/logskewnormal.py` |
| `dad` | amortised linear-softmax policy trained by REINFORCE on per-context sPCE, checkpoint-selected on simulated validation episodes | `agents/dad.py`, `train_dad.py`, weights in `dad/` |

Checks behind them: the gamma mixture's update matches brute force to 6 digits and its EIG
matches nested MC; the exact log-skew-normal EIG matches the gate's grid reference to 0.003
nats; `GIGPoisson.rate_quantile` (new, by quadrature in log-rate) matches SciPy to 5 digits
where SciPy works and keeps working at large orders, where SciPy's `geninvgauss.ppf` fails.

**DAD is greedy D-optimality in practice.** Training improved the stochastic policy, but its
greedy deployment never beat the starting point (D-optimality per unit cost) on validation
in four of five settings, so checkpoint selection kept the start; on bulk it selected a
trained policy 0.3 nats worse on held-out simulations. Training took 9-31 min per setting.

**A kink fix (2026-10-02).** The posterior grid and the MCMC proposals took their width from
the Laplace curvature, which collapses when the mode sits on the edge of a prior so skewed it
is a half-normal (the Swift-BAT cataclysmic variables, fitted skew 8.9e7). The width is now
never below `min(omega, 1 / sqrt(T + 1))`. That changes posterior means by up to 30 per cent
for that class and by at most 4e-6 relative anywhere else, so only the three log-skew-normal
agents on hard X-ray were rerun; the previous files are in `results/pre_kink_fix/`.

`significance.py` now tests all 105 pairs per setting (Holm within the setting) and prints
each method against GIG-Poisson + EIG; figures are drawn twice per setting, as a `models`
and an `acquisitions` group.
