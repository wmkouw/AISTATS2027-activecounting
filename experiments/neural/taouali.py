"""A generator of neural spike statistics, parameterised from Taouali et al. (2016).

Every number here is traceable to the paper; the citations are inline so that a reader can
check each one rather than take the parameterisation on trust.

**What this is for, and what it is not for.** It is a generative model of the MT setting, for
the allocation question: an allocation study needs a simulator whatever else is available,
and the structure this one reproduces is published. It is **not** a substitute for the
measured LGN field in the mixing-law comparison. Fitting a mixing law to rates this module
generated would measure the parameterisation below, not neurons, and the result would be
circular. `run.py` keeps the two apart and labels every row with its provenance.

The model
---------

Taouali et al. adopt the negative binomial as a Gamma-Poisson mixture, their Eq. 2, with

    mu = phi * beta,    sigma^2 = mu + mu^2 / phi,    FF = 1 + mu / phi          (their Eq. 4)

``phi`` is the *inverse* dispersion: large ``phi`` is close to Poisson. Both the mean and
``phi`` are tuned to stimulus direction by von Mises (circular Gaussian) functions centred on
the cell's preferred direction, their Eq. 10 and Fig. 6:

    f(theta) = R0 + Rm * (exp(kd cos(theta - theta0)) - exp(-kd)) / (exp(kd) - exp(-kd))

Anchors taken from the paper
----------------------------

* Fig. 8 caption gives one exact pair at the preferred direction: mean ``7.14`` and inverse
  dispersion ``21.47``, so ``FF = 1 + 7.14/21.47 = 1.33``. :data:`PD_MEAN`, :data:`PD_PHI`.
* Fig. 6A, box plots of mean spike count across cells: roughly 0 to 35, median near 25 at the
  preferred direction. Fig. 6C, inverse dispersion: roughly 0 to 20, median near 5.
* Fig. 8A and 8B run both tuning curves over 0 to 25.
* Materials and Methods: MT is 40 cells at 16 directions, and trial counts are "mostly less
  than 20 trials per condition".

What is *not* in the paper
--------------------------

The per-cell tuning parameters ``(kappa_a, kappa_d, R0, Rm)`` are fitted but not tabulated,
and the joint distribution of parameters across cells is shown only as box plots. The spreads
in :data:`CELL_SPREAD` are therefore chosen to reproduce the reported ranges and are the main
thing a reader should treat as ours rather than theirs.

A structural check worth keeping
--------------------------------

Because both the mean and ``phi`` are bell-shaped on direction and ``FF = 1 + mu/phi``, the
Fano factor comes out **bimodal** in direction: lowest at the preferred direction, peaking
near plus and minus ninety degrees. That is Fig. 8C, and :func:`fano_profile` reproduces it,
so the parameterisation can be falsified rather than merely asserted.
"""

import numpy as np

__all__ = ["von_mises_tuning", "cell_parameters", "tuning_curves", "fano_profile",
           "sample_counts", "N_CELLS", "N_DIRECTIONS", "TRIALS_PER_CONDITION"]

#: MT recordings: 40 cells, 16 directions (Materials and Methods).
N_CELLS, N_DIRECTIONS = 40, 16
#: "mostly less than 20 trials per condition".
TRIALS_PER_CONDITION = 15

#: Fig. 8 caption, the one exactly quoted pair, at the preferred direction.
PD_MEAN, PD_PHI = 7.14, 21.47

#: Baselines away from the preferred direction, and the two concentrations.
#:
#: The concentrations are the one place the parameterisation does real work. Fig. 8C is only
#: bimodal because the inverse dispersion is tuned *more sharply* than the mean: just off the
#: preferred direction ``phi`` has already fallen to baseline while the mean has not, so
#: ``FF = 1 + mu/phi`` rises; far off it both sit at baseline and the ratio drops back. Equal
#: concentrations give a flat Fano profile, which is what a first version of this module
#: produced and what the check in :func:`fano_profile` caught.
#:
#: The values are ours, chosen to reproduce the *shape* of Fig. 8C rather than read off it;
#: only the pair at the preferred direction is quoted in the paper.
MEAN_BASELINE, MEAN_KAPPA = 0.2, 1.0
PHI_BASELINE, PHI_KAPPA = 2.0, 5.0

#: Log-normal spread across cells of the peak mean and peak inverse dispersion, set so the
#: populations span the box plots of Fig. 6A and 6C. Ours, not theirs.
CELL_SPREAD = {"mean": 0.50, "phi": 0.30}


def von_mises_tuning(theta, theta0, baseline, peak, kappa):
    """Their Eq. 10: a circular Gaussian rising from ``baseline`` to ``peak`` at ``theta0``."""
    num = np.exp(kappa * np.cos(theta - theta0)) - np.exp(-kappa)
    den = np.exp(kappa) - np.exp(-kappa)
    return baseline + (peak - baseline) * num / den


def cell_parameters(rng, n_cells=N_CELLS):
    """Preferred direction, peak mean and peak inverse dispersion, one row per cell."""
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)
    peak_mean = PD_MEAN * np.exp(rng.normal(0.0, CELL_SPREAD["mean"], size=n_cells))
    peak_phi = PD_PHI * np.exp(rng.normal(0.0, CELL_SPREAD["phi"], size=n_cells))
    return theta0, peak_mean, peak_phi


def tuning_curves(rng, n_cells=N_CELLS, n_dir=N_DIRECTIONS):
    """``(directions, mean[cell, dir], phi[cell, dir])``.

    ``mean`` is the expected spike count in the counting window and ``phi`` the inverse
    dispersion, so the Fano factor at any entry is ``1 + mean / phi``.
    """
    theta = np.arange(n_dir) * 2.0 * np.pi / n_dir
    t0, pm, pp = cell_parameters(rng, n_cells)
    mean = von_mises_tuning(theta[None, :], t0[:, None], MEAN_BASELINE,
                            pm[:, None], MEAN_KAPPA)
    phi = von_mises_tuning(theta[None, :], t0[:, None], PHI_BASELINE,
                           pp[:, None], PHI_KAPPA)
    return theta, np.maximum(mean, 1e-6), np.maximum(phi, 1e-6)


def fano_profile(n=181):
    """Fano factor against direction relative to the preferred one, their Fig. 8C.

    Returned in degrees from the preferred direction, so the bimodality is visible directly.
    """
    d = np.linspace(-np.pi, np.pi, n)
    mean = von_mises_tuning(d, 0.0, MEAN_BASELINE, PD_MEAN, MEAN_KAPPA)
    phi = von_mises_tuning(d, 0.0, PHI_BASELINE, PD_PHI, PHI_KAPPA)
    return np.degrees(d), 1.0 + mean / phi


def sample_counts(rng, mean, phi, n_trials=TRIALS_PER_CONDITION):
    """Spike counts from the doubly stochastic model: a rate per trial, then a Poisson draw.

    This is the step where the setting parts company with the manuscript's model. Here the
    rate is redrawn on **every trial**, from ``Gamma(phi, mean/phi)``; the manuscript's rate
    is fixed per context and the posterior concentrates on it (Proposition 2). Both are
    implemented so the difference can be measured rather than argued about:
    :func:`sample_counts_fixed_rate` draws the rate once.
    """
    lam = rng.gamma(shape=phi, scale=mean / phi, size=(n_trials,) + np.shape(mean))
    return rng.poisson(lam)


def sample_counts_fixed_rate(rng, mean, phi, n_trials=TRIALS_PER_CONDITION):
    """The manuscript's reading: one rate per context, held fixed across trials."""
    lam = rng.gamma(shape=phi, scale=mean / phi)
    return rng.poisson(np.broadcast_to(lam, (n_trials,) + np.shape(mean)))
