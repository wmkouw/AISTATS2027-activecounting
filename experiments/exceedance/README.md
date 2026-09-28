# Does the third parameter pay when the decision is a threshold?

Table 1 of the manuscript locates what the third parameter moves: the far tail and the mass
at zero. Every score the paper reports — held-out predictive, rate error, top-*m* regret —
weighs the centre. So the model's advantage has been measured with instruments built not to
see it, which is the most likely reason the mixing law is worth 0.012 nats in the bulk study
and ties the lognormal in the gamma-ray study.

A threshold decision weighs the tail, and the domains supply the thresholds: is a Safecast
cell above the evacuation limit, is a source above the detection threshold, is a block above
cutoff grade?

## A. What the scores can see

Analytic, no sampling, no allocation. The truth is a GIG-Poisson predictive computed exactly;
the alternatives are fitted to its first two moments, as the allocation studies fit them.

| true order | model | KL from truth | err at q90 | q95 | q99 | q99.9 |
|---|---|---|---|---|---|---|
| −0.5 | gamma | 0.109 | 24% | 28% | 2% | 63% |
| | lognormal | **0.010** | 5% | 18% | 27% | **15%** |
| −1.0 | gamma | 0.205 | 35% | 53% | 16% | 80% |
| | lognormal | **0.006** | 8% | 1% | 17% | **12%** |
| −1.5 | gamma | 0.300 | 47% | 76% | 62% | 85% |
| | lognormal | **0.028** | 17% | 22% | 9% | **22%** |
| −2.0 | gamma | 0.394 | 39% | 95% | 159% | 61% |
| | lognormal | **0.068** | 17% | 39% | 63% | **18%** |

The lognormal rows are the result. It sits 0.006–0.068 nats from the truth in cross-entropy
— at order −1.0 that is *half* what the bulk study reports as the entire mixing-law effect
and calls negligible — while being 12–22% wrong about exceedance at the 99.9th percentile.
A centre-weighted score cannot see a difference a threshold decision would act on.

## B. Whether it survives a budget

Same three mixing laws under the proposed acquisition, allocating a scarce budget (8 units
over 8 contexts) and then scored on the threshold. 30 episodes, 4 orders, thresholds at the
75th, 90th and 95th percentile of the pooled true predictive. Scored by the absolute error
against the exceedance probability the true rate implies, and by a log-score on held-out
exceedance events.

At the middle threshold, absolute error in the predicted exceedance probability:

| order | GIG | gamma | gain | *t* | lognormal | gain | *t* |
|---|---|---|---|---|---|---|---|
| −0.5 | 0.0560 | 0.0637 | 12% | −1.4 | 0.0537 | −4% | +0.4 |
| −1.0 | 0.0438 | 0.0580 | **24%** | **−3.1** | 0.0471 | 7% | −1.1 |
| −1.5 | 0.0525 | 0.0689 | **24%** | **−2.3** | 0.0616 | 15% | −1.4 |
| −2.0 | 0.0641 | 0.0745 | **14%** | **−2.0** | 0.0685 | 6% | −0.8 |

Across all twelve (order, threshold) points:

- **against the gamma**: ours is ahead at 11 of 12, median *t* = −1.26, reaching |*t*| ≥ 2 at
  two of them and at every order below −0.5 at the middle threshold.
- **against the lognormal**: ahead at 8 of 12, median *t* = −0.38. Not a reliable advantage.

## What to conclude, and what not to

The tail advantage is real against the gamma and survives a budget: 12–24% less error in the
exceedance probability, consistently signed, significant at three of four orders at the
middle threshold. That matters more than its size suggests, because the gamma-Poisson pair is
the *only other conjugate option* and is what the paper's whole argument is competing
against.

It does **not** survive against the lognormal. Study A says the lognormal's prior predictive
is materially wrong in the tail; study B says that after a budget's worth of counts the error
has shrunk enough to tie. The lognormal is flexible enough to track the tail once it has
data, and the gamma is not.

So this is a remark or a short subsection, not a headline result. It strengthens the
comparison the conjugacy argument actually needs and leaves the lognormal comparison where
the gamma-ray study already left it: a tie on accuracy, decided by cost.

## Three metrics that did not work, and why

Recorded because each failed in an instructive direction.

- **Brier score at the 99th percentile.** Saturates. Every model predicts near zero, every
  model is right, false-positive and false-negative rates come out at 0–0.8% for everyone and
  nothing separates. `brier` is still in the CSV so the saturation stays on the record.
- **Relative error in the exceedance probability.** Diverges. The true probability at a faint
  context is near zero and sits in the denominator; the first run reported relative errors of
  10⁹.
- **Re-running the allocation per threshold.** The allocation does not depend on the
  threshold. Doing it three times over bought nothing and timed the study out after one order.

The surviving pair — absolute error against an exactly computable truth, and a proper
log-score — degenerate in neither direction.

## Running

```bash
cd experiments/exceedance
python run.py     # ~10 min; writes results/discrimination.csv and results/allocation.csv
```

Reproducible from seed 0. `ORDERS` spans the range the measured fields show (Newsgroups
+1.73, Fermi −1.43, Safecast −2.07); note the manuscript's simulators all generate at −0.5,
which is the one order at which the gamma is *not* significantly behind here.
