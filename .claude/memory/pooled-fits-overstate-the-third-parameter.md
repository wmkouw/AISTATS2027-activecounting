---
name: pooled-fits-overstate-the-third-parameter
description: Section 5.4 fits the rate fields pooled, but agents carry priors conditioned on class or stratum; on Safecast the third parameter's advantage is largely a pooling artifact, on Fermi it survives.
metadata:
  type: project
---

Found 2026-09-18 while building the Safecast allocation study.

**The gap.** Section 5.4 fits each field POOLED. But the agents of sections 5.2 and 5.3 never
carry a pooled prior: the photon study conditions on source class, and any survey study
conditions on stratum. If pooling groups of different means is itself what creates the heavy
tail, the pooled fit measures the mixture, not the field, and overstates the third parameter
exactly where it is used.

**Fermi-LAT: it survives.** Within every source class the fitted order stays negative and the
GIG wins outright.

| class | n | dAIC gamma | dAIC lognormal | fitted order |
|---|---|---|---|---|
| psr | 239 | 224 | 8 | -0.69 |
| fsrq | 694 | 496 | 102 | -0.99 |
| bll | 1131 | 722 | 142 | -1.21 |
| bcu | 1312 | 421 | 34 | -2.30 |
| unassoc | 1689 | 1181 | 178 | -0.96 |
| **summed** | | **3044** | **464** | |

So section 5.3's model win (t = -7.4) rests on solid ground and the Fermi half of 5.4 is not a
mixture artifact.

**Safecast: it largely does not.** Pooled, the GIG wins by dAIC 568 over the gamma at order
-2.07. Stratified by distance ring x plume sector -- which is what a planner knows before
going out -- and refitted properly:

| stratum | n | AIC gamma | AIC lognormal | AIC GIG | fitted order |
|---|---|---|---|---|---|
| inner_nw | 128 | 1701.9 | **1673.8** | 1674.9 | -1.34 |
| mid_nw | 353 | 4798.7 | 4707.5 | **4692.5** | -0.85 |
| outer_nw | 443 | 4853.6 | **4840.5** | 4843.1 | +0.73 |
| outer_other | 183 | 1768.3 | 1758.9 | **1757.7** | -5.17 |
| **summed** | 1107 | 13122.5 | 12980.7 | **12968.2** | |

The gamma gap falls from 568 pooled to 154 stratified (0.51 -> 0.14 per observation) and the
lognormal gap from 136 to 12.5. Much of the pooled -2.07 is between-stratum mean variation.

**A real bug, found on the way.** `scipy.stats.geninvgauss.fit`, which
`methods/countmodels.py::GIGPoisson.fit_population` and
`experiments/experiment-rate-fields/compare.py` both call, lands in a bad local optimum on
some fields: it returned omega=0.000 with a positive order on two Safecast strata, costing
**119 and 118 AIC points** against a multi-start Nelder-Mead on (order, log omega, log eta).
A three-parameter family containing the gamma as a boundary cannot truly be 108 points worse
than the gamma, which is how the failure was spotted.

**Checked, and no published number is affected.** Multi-start recovers 0.0 AIC on all five
Fermi source classes (the 5.3 priors) and 0.7, 0.1 and 0.0 on the three pooled fields of
table 4. scipy finds the optimum everywhere the paper relies on it; the failure is confined
to the two Safecast strata constructed here. Still use multi-start for any NEW fit on a
small or strongly skewed subset, and check that the GIG never scores worse than the gamma by
more than the two AIC points its extra parameter costs.

**How to apply:** report 5.4 both pooled and conditioned on whatever the corresponding agent
carries; the conditioned number is the honest one for a sensing claim. Safecast is not the
second confirmation the model claim needs -- lead with Fermi, where it holds within class.
Related: [[three-real-rate-fields]], [[acquisition-axis-cannot-separate]].
