"""The systematic programme: every block in turn, at one fixed sample volume.

The incumbent. A grid or round-robin survey at a constant sample size is what a sampling
campaign actually runs when nothing adaptive is in place, and any adaptive scheme has to
beat it to be worth the trouble. It is a stronger baseline than it looks, because spreading
effort evenly is close to optimal whenever the quantity of interest is a coarse ranking.
"""

from .base import Agent

__all__ = ["Systematic"]


class Systematic(Agent):
    criterion = "uniform"
    criterion_label = "systematic"
    closed_form = True

    def reset(self, env, prior_moments, recovery):
        super(Systematic, self).reset(env, prior_moments, recovery)
        self._next = 0

    def act(self, env, rng):
        k = self._next % env.n_contexts
        self._next = k + 1
        return k, float(env.action_values[len(env.action_values) // 2])
