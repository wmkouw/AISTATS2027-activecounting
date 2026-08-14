"""Sensing agents: pairs of a count model and an acquisition criterion.

The registry below spans two axes deliberately, because they answer different questions.

**Does the acquisition matter?** Hold the model at GIG-Poisson and vary the criterion. That
is the first group, and it is where maximum entropy sampling, D-optimality, Thompson
sampling and the rest appear.

**Does the model matter?** Hold the criterion at expected information gain and vary the
mixing law. That is the second group. Every model is initialised from the same prior mean
and variance per block, so a difference between them is a difference in what the family can
express and not in what it was told.

Considered and not included: adaptive cluster sampling (Thompson, 1990), the standard design
for rare and clustered populations, which expands a neighbourhood around any unit exceeding
a threshold. It needs a spatial adjacency between sampling units to expand into, and the
blocks here are independent under the prior, so the rule has nothing to act on. It becomes
relevant once the context set is given spatial structure, which is the extension named in
the discussion.
"""

from methods.countmodels import GammaPoisson, GIGPoisson, LognormalPoisson

from .base import Agent, ScoredAgent          # noqa: F401
from .dopt import DOptimality
from .eig import EIG
from .epig import EPIG
from .maxent import MaxEntropy
from .neyman import Neyman
from .systematic import Systematic
from .thompson import Thompson
from .uniform_random import RandomAgent
from .variance import EpistemicVariance, TotalVariance

#: Criteria compared under the proposed model.
CRITERIA = (EIG, EPIG, MaxEntropy, DOptimality, Neyman, TotalVariance,
            EpistemicVariance, Thompson, Systematic, RandomAgent)

#: Mixing laws compared under the proposed criterion.
ALTERNATIVE_MODELS = (GammaPoisson, LognormalPoisson)


#: Criteria carried across every model. EPIG is excluded from the non-conjugate model: its
#: outer expectation needs a posterior predictive entropy at every node, and under a grid
#: posterior that is a quadrature inside a quadrature, which costs more than the whole rest
#: of the study put together. That exclusion is itself a result and is reported, not hidden.
CROSSED = (EIG, MaxEntropy, DOptimality, Neyman, TotalVariance, EpistemicVariance,
           Thompson)


def build_all(cross="models"):
    """Every agent in the study.

    ``cross`` selects how much of the criterion-by-model grid to run:

        ``False``     the ten criteria under the proposed model only.
        ``"models"``  those, plus the proposed criterion under each alternative mixing law.
                      This is the headline set: one axis at a time, twelve agents.
        ``True``      the full grid, every criterion in :data:`CROSSED` under every model.

    The belief-free designs (systematic, random) are run once whatever the setting, since
    they never consult a model.
    """
    out = [cls() for cls in CRITERIA]
    if cross == "models":
        out += [EIG(model=m()) for m in ALTERNATIVE_MODELS]
    elif cross is True:
        for m in ALTERNATIVE_MODELS:
            out += [cls(model=m()) for cls in CROSSED]
            out += [EPIG(model=m())] if m is not LognormalPoisson else []
    return out


NAMES = tuple(a.name for a in build_all())
LABELS = {a.name: a.label for a in build_all()}


def build(name):
    for a in build_all():
        if a.name == name:
            return a
    raise ValueError("unknown agent {!r}; known: {}".format(name, ", ".join(NAMES)))
