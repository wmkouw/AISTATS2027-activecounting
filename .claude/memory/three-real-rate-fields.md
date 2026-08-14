---
name: three-real-rate-fields
description: On three measured rate fields the fitted GIG order predicts whether the third parameter is needed; it is negative on physical counting fields and positive on a linguistic one.
metadata:
  type: project
---

Established 2026-08-11 in `experiments/experiment-rate-fields`. The strongest result in the
project, because it depends on no acquisition, budget, policy or simulator.

| field | n | dAIC gamma | dAIC lognormal | fitted order | AIC/obs |
|---|---|---|---|---|---|
| Fermi-LAT source fluxes | 4557 | +1773 | +555 | **-1.43** | 0.39 |
| Safecast cell rates, Fukushima | 1107 | +568 | +136 | **-2.07** | 0.51 |
| 20 Newsgroups term rates | 22460 | +7 | +1362 | **+1.73** | 0.00 |

**The fitted order is the diagnostic.** The gamma is the boundary of the family at
`omega -> 0` with positive order, so a field fitted at a negative order is one no gamma can
reach; a field fitted positive sits inside the gamma's range and the third parameter has
nothing to do. Both physical counting fields are the first kind, the linguistic field the
second.

This gives a cheap pre-registration check: fit the three laws to whatever rates are already
in hand and read off the order, before designing any sensing at all.

Also: the ordering of the two-parameter laws is unstable (lognormal much better on the
physical fields, much worse on the linguistic one), so committing to one two-parameter
alternative is wrong somewhere.

**How to apply:** lead the paper's model argument with this, not with the simulated
comparisons, which are circular by construction. Do not generalise the linguistic negative
to language as a whole; it is one field, and it excludes rare terms where burstiness is
strongest. Related: [[sichel-is-the-starting-point]],
[[budget-scarcity-decides-whether-acquisition-matters]].
