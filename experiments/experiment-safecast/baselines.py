"""A systematic programme that sizes its dwell to the budget it was given.

The shared :class:`agents.Systematic` visits contexts in turn at a fixed action size, the
median of the grid. That is the right incumbent when the budget covers at least one full
pass, which is the regime the bulk-sampling study runs in. Under a scarce budget it is not:
a fixed 40 s dwell on a 120 s budget reaches three cells of ninety-six, and beating that
would be beating a programme no survey planner would write.

A planner sizes the grid to the area and the time available. This baseline does the same: it
takes the largest dwell on the schedule that still lets the budget cover every cell, falling
back to the shortest when even that does not fit, and then visits cells in turn. It is the
strongest honest version of "spread the effort evenly", and it is what an adaptive scheme
has to beat.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from agents.base import Agent                                          # noqa: E402

__all__ = ["BudgetedSystematic"]


class BudgetedSystematic(Agent):
    criterion = "uniform-budgeted"
    criterion_label = "systematic (budget-matched)"
    closed_form = True

    def __init__(self, model=None, budget=None):
        super(BudgetedSystematic, self).__init__(model)
        self.budget = budget

    def reset(self, env, prior_moments, recovery):
        super(BudgetedSystematic, self).reset(env, prior_moments, recovery)
        self._next = 0
        budget = getattr(self, "budget", None) or env.budget
        grid = np.sort(np.asarray(env.action_values, dtype=float))
        fits = grid[grid * env.n_contexts <= budget + 1e-9]
        self._dwell = float(fits[-1]) if fits.size else float(grid[0])

    def act(self, env, rng):
        k = self._next % env.n_contexts
        self._next = k + 1
        return k, self._dwell
