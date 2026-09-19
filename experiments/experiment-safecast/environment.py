"""Allocating detector time across cells of a radiation survey.

A ground team walks a contaminated area with a Geiger counter and must decide which cells to
stand in and for how long. The tube counts decays, so the number recorded in an integration
is Poisson about the cell's rate times the dwell::

    lam_k                             true count rate of cell k, in counts per minute
    u = (k, t)                        which cell, and how many seconds to integrate
    f(u) = t / 60                     exposure, in minutes
    y | lam_k, u  ~  Poisson(f(u) lam_k)
    cost(u) = t                       budget spent, in seconds of detector time

Assumption 1 holds exactly and without idealisation: a dwell scales the rate of the cell
being stood in by a known constant and cannot make one cell observable through another.
There is no recovery factor to estimate and no effective area to assume, which is why this
is the cleanest instance of the exposure action in the paper.

**What is real here.** The rates are the measured count rates of 1107 five-hundred-metre
cells within 60 km of Fukushima Daiichi, recorded in counts per minute by the Geiger tubes of
the Safecast project and released CC0. No model in the comparison produced the field it is
asked to learn. This is the field whose mixing law is fitted at order -2.07 in the rate-field
study, the furthest of the three from anything a gamma can reach.

**The action grid is the instrument's own.** A bGeigie logger integrates in five-second
blocks, so the dwells offered are multiples of five seconds. Nothing here is a round number
chosen for convenience.

**Priors are conditioned on where the cell is, and that is where the problem lives.** A
planner knows a cell's distance from the plant and whether it lies in the north-west corridor
the 2011 plume ran along, before measuring anything. Those strata are not alike: mean rate
varies fourfold across them and the variance-to-mean ratio forty-sixfold. Giving every cell
the same prior would make cells exchangeable and a survey that visits each in turn would be
near-optimal by construction, leaving nothing for an acquisition to find.

**No leakage.** Each stratum is split in two. One half supplies the population a prior is
fitted to, the other supplies the cells a survey actually visits, so a model knows what cells
of a stratum look like in general and nothing about the particular cell in front of it.

**The budget is scarce relative to the cell list, and that is deliberate.** Most cells cannot
be visited at all and the survey must decide which to leave alone. Under an abundant budget
every cell ends up well measured whatever the schedule does and no allocation rule can
distinguish itself; that regime measures nothing, and it is swept rather than assumed.

**Two limitations, stated rather than hidden.** A cell's rate is the mean of drive-by passes
made over years during which the contamination decayed, so it is weaker ground truth than a
catalogue, and cells exist only where roads run. And travel between cells is not charged: the
budget is detector time, not field time, so a survey that hops between distant cells pays the
same as one that works a transect. Charging travel would make this a routing problem, which
is a different paper.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cells import STRATA, load_cells, stratum_of                      # noqa: E402

__all__ = ["SafecastSurvey"]


class SafecastSurvey(object):
    """Ground truth, action set and budget for one survey campaign."""

    label = "Fukushima ground survey"
    action_label = "dwell"
    action_units = "s"

    def __init__(self, n_cells=96, dwells=None, budget=480.0, eval_dwell=20.0,
                 band=(2.0, 98.0), split_seed=0):
        self.n_cells = int(n_cells)
        #: Dwells the team may choose, in seconds; multiples of the logger's 5 s block.
        self.dwells = (np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0]) if dwells is None
                       else np.asarray(dwells, dtype=float))
        self.budget = float(budget)
        self.eval_dwell = float(eval_dwell)

        rate, dist, bearing = load_cells()
        lo, hi = np.percentile(rate, band[0]), np.percentile(rate, band[1])
        keep = (rate >= lo) & (rate <= hi)
        rate, dist, bearing = rate[keep], dist[keep], bearing[keep]
        strat = np.array([stratum_of(d, b) for d, b in zip(dist, bearing)], dtype=object)

        #: Which stratum each slot in the cell list holds. Fixed at construction, so the
        #: priors an agent is given do not depend on whether a survey has been drawn.
        self._strata = [STRATA[i % len(STRATA)] for i in range(self.n_cells)]

        rng = np.random.default_rng(split_seed)
        self.population, self.targets = {}, {}
        for s in STRATA:
            sub = rate[strat == s]
            order = rng.permutation(sub.size)
            half = sub.size // 2
            self.population[s] = sub[order[:half]]
            self.targets[s] = sub[order[half:]]

    # -- the generic environment protocol ----------------------------------

    n_contexts = property(lambda self: self.n_cells)
    action_values = property(lambda self: self.dwells)
    target_exposure = property(lambda self: self.eval_dwell / 60.0)

    # -- the decision problem ----------------------------------------------

    def actions(self):
        return [(k, float(t)) for k in range(self.n_cells) for t in self.dwells]

    def exposure(self, u, recovery=None):
        """``f(u) = t / 60``, converting a dwell in seconds to the rate's own minutes."""
        return float(u[1]) / 60.0

    def cost(self, u):
        """Budget spent by a reading, in seconds of detector time."""
        return float(u[1])

    # -- what every agent is told ------------------------------------------

    def initial_beliefs(self, model):
        """One prior per cell: this family, fitted to the population half of its stratum."""
        cache, out = {}, []
        for s in self._strata:
            key = (model.name, s)
            if key not in cache:
                cache[key] = model.fit_population(self.population[s])
            out.append(cache[key])
        return out

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw a survey: a cell list spanning the strata, and the rates it will face."""
        truths = np.array([self.targets[s][rng.integers(self.targets[s].size)]
                           for s in self._strata])
        moments = [(float(self.population[s].mean()), float(self.population[s].var()))
                   for s in self._strata]
        return moments, truths, np.ones(self.n_cells)

    def stone_fields(self, truths, recovery, rng):
        """One realised decay stream per cell, as a Poisson process in exposure.

        A dwell of ``f`` minutes consumes the next ``f`` of the cell's stream, so two agents
        that stand in the same cell for the same total time record the same decays and any
        difference between them is a difference in decisions rather than in luck.
        """
        extent = self.budget / 60.0 * 1.05 + 1.0
        return [np.sort(rng.uniform(0.0, extent, size=int(rng.poisson(t * extent))))
                for t in truths]

    def held_out(self, truths, recovery, rng, n=24):
        """Fresh readings at the evaluation dwell, shared by every agent."""
        f = self.eval_dwell / 60.0
        return [rng.poisson(f * truths[k], size=n) for k in range(self.n_cells)]
