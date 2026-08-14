---
name: budget-scarcity-decides-whether-acquisition-matters
description: Adaptive allocation only beats round-robin when the budget is scarce relative to the number of contexts; both experiments were first built in the abundant regime, where nothing separates.
metadata:
  type: project
---

Found 2026-08-09, after three failed attempts to make the photon-counting study separate.

**The rule.** When the budget is large enough that every context ends up well measured
whatever the schedule does, uniform allocation is near-optimal and *no* acquisition function
can distinguish itself. Prior heterogeneity does not rescue this: the differences it creates
are washed out by data. Only scarcity, meaning most contexts cannot be observed at all,
makes the allocation decision real.

**Measured.** Fermi photon counting, same environment, same agents, only the ratio changed:

| regime | contexts | budget | round-robin vs our EIG (NLPD) | gamma vs GIG (NLPD) |
|---|---|---|---|---|
| abundant | 12 | 120 Ms (10 Ms each) | **+0.0015, round-robin nominally ahead** | -0.012, t=-1.2 |
| scarce | 48 | 120 Ms (2.5 Ms each) | -0.4104, t=-18.7 | -0.049, t=-7.4 |

The model axis grows fourfold under scarcity too, and only there does it separate on the
decision metric (t=-2.3). Scarcity is what makes both axes measurable.

**What this invalidates.** The bulk-sampling study of `main.tex` section 5.2 was built in
the abundant regime as well: 12 blocks, 240 m^3, 240 rounds. Its headline numbers, including
"the acquisition matters roughly twenty-five times more than the model" and the top-4 regret
ties, are statements about an abundant budget and not general ones. They must either be
re-measured under scarcity or explicitly scoped. **Do not quote them as regime-free.**

**Why it was missed three times.** Both environments were designed by choosing a plausible
number of contexts and a plausible budget independently, and never checking the ratio. The
diagnostic to run first, before any policy comparison: divide the budget by the number of
contexts and ask whether a single context could be well measured with that share. If it
could, the comparison will measure nothing.

**How to apply:** treat budget scarcity as an axis of the experiments rather than a fixed
setting. "Here is the regime in which choosing where to look matters, and here is where it
does not" answers the question a reviewer actually has, and it is defensible in a way that a
single-budget win is not. Related: [[bulk-sampling-is-the-application]].
