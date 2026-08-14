---
name: overdispersion-has-two-readings
description: The GIG-Poisson DGP makes the rate a fixed unknown per site, so the overdispersion is parameter uncertainty; the random-effect reading is a different process and the choice is unsettled.
metadata:
  type: project
---

Unresolved as of 2026-08-08, raised while building `experiments/experiment-gigpoisson`.
"Overdispersed count data" admits two processes and they are not interchangeable.

1. **Fixed unknown rate** (what is implemented). `lam_k ~ GIG(p,a,b)` drawn once per site,
   `y | lam_k, u ~ Poisson(f(u) lam_k)` for every read. Conditional on the realised rate the
   counts are Poisson, measured Fano 0.85 to 1.14. The overdispersion is in the *marginal*
   and is parameter uncertainty, so it shrinks as the sensor learns. Marginal Fano runs 1.01
   to 3202 across the action grid. This is the reading the conjugate update
   `GIG(p,a,b) -> GIG(p+y, a, b+2f(u))` describes, and it is what an acquisition function
   scores.
2. **Random effect per observation.** A fresh rate for every read. The counts stay
   overdispersed however much is known, and there is no fixed rate to learn: the sensor
   would be learning hyperparameters instead, which is not conjugate in the same way.

Reading 1 is what was asked for and what is built. The concern is that a reader who hears
"overdispersed count data" may assume reading 2, and that the main claim (a GIG-Poisson
sensor beats general-purpose active sensors on overdispersed counts) may need reading 2 to
be interesting, since under reading 1 the overdispersion is transient by construction.

**How to apply:** settle this before the baselines are chosen, because it decides what they
are. Do not let the paper use the phrase without saying which process it means. Detail and
the measured numbers are in `experiments/experiment-gigpoisson/README.md`. Related:
[[sichel-is-the-starting-point]].
