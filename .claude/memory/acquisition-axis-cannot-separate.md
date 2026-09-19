---
name: acquisition-axis-cannot-separate
description: Four theory-driven probes failed to make the EIG acquisition beat D-optimality; the ceiling is ~0.006 nats against the model axis's 0.049, so the paper should lead with the model.
metadata:
  type: project
---

Established 2026-09-18, after the reviewer pass on `main.tex`. **Do not re-run this search.**

**Why D-optimality ties.** `fisher_dopt` computes `log(1 + f V[lam]/E[lam])`, which is exactly
`log F(u)`, the Fano factor of eq. (6). So D-optimality, Neyman and both variance criteria are
functions of the first two moments only. Two mechanisms then close the gap:

1. *Order-invariance.* By Corollary 1 the belief depends on a block's history only through the
   accumulated count and accumulated exposure, both sums. Per-step argmax disagreement is 33%
   (median EIG shortfall 0, 90th pct 37%), but cumulative allocations converge: cosine 0.987,
   L1 gap 10% of budget. A greedy criterion only has to get long-run proportions right.
2. *The scarcity squeeze.* Abundant budget -> every context is well measured, nothing separates.
   Scarce budget -> spreading is near-optimal, and an *indifferent* criterion spreads perfectly.
   Blindness is punished only when it causes wrong concentration, which is max-ent, Thompson and
   the variance criteria: exactly what the paper already reports.

**The four probes**, each predicted from the mechanism before running:

| setting | gap (nats) | t |
|---|---|---|
| sequential, abundant (paper 5.2) | 0.001 | -0.4 |
| sequential, scarce shape field | 0.0059 | -1.4 |
| open-loop batch, scarce ordinary field | 0.0031 | -0.4 |
| open-loop batch, shape field | 0.0081 | -0.9 |

The **shape field** is the decisive one and worth keeping in the paper. Blocks are matched in
prior mean and variance but differ in GIG order, so mean, variance and log-Fano are identical to
1e-15 across blocks while true EIG spreads 1.43x. D-optimality, Neyman, max-ent and uniform then
produce *bit-identical* allocations (rho(F, G) = 0.000); EIG tracks the true ordering at
rho = 0.905 -- and gains 0.008 nats. That forecloses "you didn't look hard enough".

**The number that should reframe the paper.** The acquisition ceiling is ~0.006 nats. The model
effect on the Fermi field is 0.049 nats at t = -7.4. Against *competent* criteria the model
matters 6-8x more than the acquisition; section 5.2's "acquisition matters 25x more" is measured
against broken ones. Lead with the model. See [[three-real-rate-fields]] and
[[budget-scarcity-decides-whether-acquisition-matters]].

**Salvage, not yet in the paper.** G(F) is concave in exposure (0/12 and 0/48 violations) and by
Corollary 1 the batch objective is separable, max sum_k G_k(F_k) s.t. sum_k F_k <= B, so greedy
marginal allocation solves open-loop design *exactly* with no nesting -- the case that normally
forces nested Monte Carlo. Beats uniform by 0.067 nats at t = -8.3 in the scarce batch run.

**How to apply:** stop trying to win on the acquisition. Report the shape field as a scoping
result, add batch design as a contribution, and move the headline to the model axis.
Probe scripts were scratch-only and were not kept; the recipe above is enough to rebuild them.
