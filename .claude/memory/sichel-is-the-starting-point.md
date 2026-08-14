---
name: sichel-is-the-starting-point
description: The project's subject is active sensing for overdispersed counts, seeded by the Sichel / GIG-Poisson member that the predecessor paper catalogued but never implemented.
metadata:
  type: project
---

Stated by Wouter on 2026-08-08 at project start: "We will be building and testing an active
sensing framework for overdispersed count data." The seeded note
`notes/sichel-gig-poisson.md` is the entry point.

The member: a Poisson likelihood mixed over a generalised inverse Gaussian rate. Conjugate,
with the exposure entering the second GIG scale parameter additively, hence a $\nu$-action in
the predecessor's vocabulary ([[project-lineage-poissongamma-efe]]). The marginal is the
Sichel law; its normalising constants are modified Bessel functions of the second kind, which
overflow in double precision at moderate order and must be evaluated in log space. Its
information quantities need a truncated convergent series over an unbounded support, as the
negative binomial does, not a closed form.

**The motivation, corrected 2026-08-08.** Not free overdispersion. A negative binomial has
two free parameters, so its mean and variance are already free, and claiming otherwise
invites the first objection a reviewer will raise. What the GIG buys is a **third**
parameter: with the predictive mean and variance both held fixed, it still moves mass
between the centre and the tail. Measured at matched mean and variance, tail mass beyond
`y = 50` differs from the moment-matched negative binomial by a factor 216 at `omega = 3`
down to 0.7 at `omega = 0.03`. This is also how the count literature justifies the Sichel:
skew, tail and zero-inflation. The seeded note `notes/sichel-gig-poisson.md` had this wrong
and now carries the correction.

Both open items carried over from the predecessor are now done:
1. log-space Bessel evaluation, built in `experiments/experiment-gigpoisson/dgp.py` as a
   scaled/asymptotic split plus an order recurrence accumulated through `logaddexp`;
2. the literature sweep, 2026-08-08, eight queries in four vocabularies, recorded in
   `notes/sichel-literature-2026-08.md`. No prior treatment of the Sichel or GIG-Poisson in
   active sensing, sequential design or expected information gain. Still `WebSearch` only,
   so it supports "we found none", not "there is none".

**How to apply:** the absence is now documented with a method attached, but Google Scholar,
citation chasing and a full read of Overstall (2022) remain before a novelty or coverage
claim goes in the paper. The predecessor's research direction died once already from a
novelty collapse.
