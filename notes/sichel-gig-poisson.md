# Sichel / GIG-Poisson distribution

Date: 2026-08-07

## What the member is

Mixing law is the generalised inverse Gaussian, density proportional to

    λ^(p-1) exp{ -(a/λ + bλ)/2 }

with a Poisson likelihood `P(y | f(u) λ)`. The posterior is

    p(λ | y, u) = GIG(λ | p + y, a, b + 2 f(u))

so conjugacy is preserved and the exposure enters the second scale parameter
additively. The marginal is the Sichel law, whose normalising constants are
modified Bessel functions of the second kind.

**Corrected 2026-08-08, see `sichel-literature-2026-08.md`.** This section previously said
a gamma-Poisson pair fixes the relation between predictive mean and variance, and that the
GIG supplies free dispersion without leaving the class. That is wrong: a negative binomial
has two free parameters, so its mean and variance are already free, and overdispersion
alone is no reason to leave the conjugate Poisson pair.

What the GIG buys is a **third** parameter, so that with the predictive mean and variance
both held fixed there is still freedom to move mass between the centre of the distribution
and the tail. Measured at matched mean and variance, the tail mass beyond `y = 50` differs
from the moment-matched negative binomial by a factor of 216 at `omega = 3` down to 0.7 at
`omega = 0.03`, and the mass at zero by up to 1.6. This is also how the count-modelling
literature justifies the Sichel: skew, tail and zero-inflation, not dispersion. And it is
bought **without leaving the class**, since the GIG is conjugate to a Poisson likelihood
and an exposure action preserves that conjugacy.

## Numerical hazard

The Bessel functions overflow in double precision at moderate order and must be
evaluated in log space. This is the second instance of `rem:conditioning`
(main.tex:629, 1436), the first being the gamma-Poisson EIG cancellation at large
predictive mean.

Its information quantities need the same one-dimensional support sum over y as the
negative binomial member of §7.2 — an unbounded support, so a truncated convergent
series, not a closed form.

## Literature sweep, 2026-08-07

Query run: *"Sichel distribution generalized inverse Gaussian mixed Poisson active
sensing sequential design information gain"*.

**What the distribution is used for in the wild.** Results confirm the Sichel is the
Poisson mixed with the extended GIG (also called the Poisson-generalised inverse
Gaussian, PGIG). Reported application areas: insurance claim counts, protein
abundance, word frequency in text, consumer purchase behaviour. It is characterised
as a long-tailed law for highly skewed, overdispersed, zero-inflated, heavy-tailed
count data.

**What was not found.** The sweep returned nothing connecting the Sichel / GIG-Poisson
to active sensing, sequential design, or information gain. The search summary flagged
this absence explicitly. Results were confined to distributional theory and inference
(e.g. *Generalized Sichel Distribution and Associated Inference*, JSTA 2017; *Limit
Shape of the Generalized Inverse Gaussian-Poisson Distribution*, arXiv:2303.08139;
multivariate mixed Poisson GIG INAR(1) regression).

This matches the pattern in the two companion sweeps run the same day: the gamma
member with unknown shape and the Dirichlet-multinomial with a trial-count action
also returned no closed-form-EIG or active-sensing treatment.

**Caveat on strength of evidence.** This was a single web search, US-only, with a
summarising model between the query and the papers. A hit would be strong evidence;
a miss is weak evidence. Before this is used as a coverage claim in the paper it
needs a proper sweep — at minimum Google Scholar and the mixed-Poisson /
actuarial-design literature, where an exposure-allocation treatment is most likely
to be hiding.

## Open items

1. **Log-space Bessel evaluation.** Required, per `rem:conditioning`. Worth writing
   the same way the NB EIG is written — accumulate over the support term by term
   rather than differencing elementary parts.
2. ~~**Proper literature sweep** - for use in active sensing, active data acquisition.~~
   Done 2026-08-08, eight queries in four vocabularies: `sichel-literature-2026-08.md`.
   No prior treatment found. Remaining: Google Scholar, citation chasing, and reading
   Overstall (2022) in full before any novelty claim is written.
