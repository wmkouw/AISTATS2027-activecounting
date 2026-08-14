"""The bulk-sampling environment: the ground truth, the actions, and the budget.

The setting Sichel invented the GIGP for. Diamonds occur in clusters, so stone counts from a
sample of gravel are far more variable than a Poisson process allows, and no standard
discrete law reproduces observed stone-count frequencies. That model is used to *value* a
deposit once samples are in hand. It is not used to decide which samples to take.

A property is divided into ``K`` blocks. Block ``k`` carries an unknown stone density
``lam_k`` in stones per cubic metre::

    lam_k          ~  GIG(alpha, a_k, b_k)          drawn once per block
    u = (k, v)                                      block, and volume of gravel to process
    f(u) = v * r_k                                  effective exposure
    y | lam_k, u   ~  Poisson( f(u) lam_k )         stones recovered
    cost(u) = v                                     budget spent, in m^3

``r_k`` is the plant's known recovery factor for that material. It multiplies the exposure,
so it is exactly the attenuation of Assumption 1.

**This object holds no model.** It knows what is true and what an action costs; it does not
know what any agent believes. Agents carry their own beliefs, because a conjugate agent
carries a few numbers and a non-conjugate one carries a grid, and the study compares them.

**What every agent is told.** The environment reports, per block, the mean and variance of
the law its rate was drawn from. That is the same prior information for everyone. An agent
whose family can reproduce those two moments and the shape besides is better placed than one
that can only match the moments, and measuring that difference is the point.

**The budget is in cubic metres, not in samples.** A policy that always requests the largest
sample learns more per round while spending the budget many times faster, so counting the
budget in rounds would measure appetite rather than judgement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from methods.gigpoisson import gig_from_mean, gig_moments, gig_sample   # noqa: E402

__all__ = ["BulkSampling"]


class BulkSampling(object):
    """Ground truth, action set and budget for one sampling campaign."""

    label = "alluvial diamond bulk sampling"
    action_label = "sample volume"
    action_units = "m$^3$"

    def __init__(self, n_blocks=12, volumes=None, order=-0.5,
                 grade_range=(0.05, 4.0), omega_range=(0.3, 2.0),
                 recovery_range=(0.55, 0.98), budget=240.0, eval_volume=5.0):
        self.n_blocks = int(n_blocks)
        #: Pit schedule: the sample volumes the plant is set up to process, in m^3.
        self.volumes = (np.array([1.0, 2.5, 5.0, 10.0, 20.0, 40.0]) if volumes is None
                        else np.asarray(volumes, dtype=float))
        self.order = float(order)
        self.grade_range = grade_range
        self.omega_range = omega_range
        self.recovery_range = recovery_range
        self.budget = float(budget)
        #: Volume of a held-out sample. Prediction is scored here.
        self.eval_volume = float(eval_volume)

    # -- the generic environment protocol ----------------------------------

    #: Agents address contexts and choose an action value; the domain names sit beside.
    n_contexts = property(lambda self: self.n_blocks)
    action_values = property(lambda self: self.volumes)
    #: Exposure of the reference observation a prediction-oriented criterion aims at.
    target_exposure = property(lambda self: self.eval_volume)

    # -- the decision problem ----------------------------------------------

    def actions(self):
        return [(k, float(v)) for k in range(self.n_blocks) for v in self.volumes]

    def exposure(self, u, recovery):
        """``f(u) = v r_k``, the effective volume of gravel the plant sees."""
        k, v = u
        return float(v) * float(recovery[k])

    def cost(self, u):
        """Budget spent by an action, in cubic metres of gravel processed."""
        return float(u[1])

    # -- ground truth ------------------------------------------------------

    def blocks(self, rng):
        """Draw a property.

        Returns ``(prior_moments, truths, recovery)``, where ``prior_moments`` is the mean
        and variance of the law each rate was drawn from, which is what every agent is told,
        and ``truths`` are the rates themselves, which no agent is told.

        Blocks differ in expected grade, in how clustered the stones are at that grade, and
        in plant recovery, so an agent that tracks only one of the three is separable from
        one that tracks all of them.
        """
        prior_moments, truths, recovery = [], [], []
        for _ in range(self.n_blocks):
            grade = float(np.exp(rng.uniform(*np.log(self.grade_range))))
            omega = float(np.exp(rng.uniform(*np.log(self.omega_range))))
            law = gig_from_mean(self.order, grade, omega)
            prior_moments.append(gig_moments(law))
            truths.append(float(gig_sample(law, 1, rng)[0]))
            recovery.append(float(rng.uniform(*self.recovery_range)))
        return prior_moments, np.asarray(truths), np.asarray(recovery)

    def stone_fields(self, truths, recovery, rng):
        """One realised stone field per block, as a Poisson process in processed volume.

        Sampling ``v`` cubic metres of block ``k`` consumes ``v r_k`` metres of its field, so
        two agents that process the same gravel recover the same stones and any difference
        between them is a difference in decisions rather than in luck.
        """
        fields = []
        for k in range(self.n_blocks):
            extent = self.budget * float(recovery[k]) * 1.05 + 10.0
            n = int(rng.poisson(truths[k] * extent))
            fields.append(np.sort(rng.uniform(0.0, extent, size=n)))
        return fields

    def held_out(self, truths, recovery, rng, n=24):
        """Fresh counts at the evaluation volume, shared by every agent."""
        return [rng.poisson(self.eval_volume * recovery[k] * truths[k], size=n)
                for k in range(self.n_blocks)]
