"""Count models a sensing agent might hold, behind one interface.

Every model here is a Poisson likelihood mixed over an unknown rate. They differ only in
the mixing law, which is the whole of the modelling question this paper asks:

    GIGPoisson        generalised inverse Gaussian mixing, three parameters. Conjugate, so
                      the posterior stays in the family and the predictive is Sichel.
    GammaPoisson      gamma mixing, two parameters. Conjugate, predictive negative binomial.
                      This is the ``omega -> 0`` boundary of the family above, so it is the
                      nearest rival rather than a different tradition.
    LognormalPoisson  lognormal mixing, two parameters. Not conjugate: the posterior is
                      carried on a grid and every quantity is a quadrature. Included because
                      it is the mixing law the species-abundance literature most often
                      prefers, and because it prices what leaving the conjugate class costs.

**Fair priors.** A comparison between models is only about the models if each starts from
the same prior information. Every model is therefore initialised by matching the first two
moments of the mixing law, through :meth:`CountModel.prior_from_moments`. No model is given
the true parameters, and no model is given a prior another could not express.

**Fair scoring.** Every model supplies a probability mass function on the non-negative
integers, so held-out predictive scores are comparable across models rather than across
supports.
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import nbinom

from .gigpoisson import (gig_logpdf, gig_moments, gig_sample, posterior as gig_posterior,
                         sichel_logpmf, sichel_support)

__all__ = ["CountModel", "GIGPoisson", "GammaPoisson", "LognormalPoisson",
           "poisson_entropy", "MODELS", "build_model"]

POISSON_SERIES_FROM = 400.0
QUAD_NODES = 64
LOG_DENSITY_DROP = 75.0
_GAUSS = {}


def poisson_entropy(mu):
    """``H[Poisson(mu)]`` in nats, vectorised: direct sum below the crossover, series above."""
    mu = np.asarray(mu, dtype=float)
    scalar = mu.ndim == 0
    mu = np.atleast_1d(mu)
    out = np.zeros_like(mu)
    small = (mu < POISSON_SERIES_FROM) & (mu > 0.0)
    if np.any(small):
        ms = mu[small]
        ymax = int(ms.max() + 12.0 * np.sqrt(ms.max()) + 40.0)
        y = np.arange(ymax + 1, dtype=float)
        lp = -ms[:, None] + y[None, :] * np.log(ms)[:, None] - gammaln(y + 1.0)[None, :]
        out[small] = -np.sum(np.exp(lp) * lp, axis=1)
    big = mu >= POISSON_SERIES_FROM
    if np.any(big):
        m = mu[big]
        out[big] = (0.5 * np.log(2.0 * np.pi * np.e * m) - 1.0 / (12.0 * m)
                    - 1.0 / (24.0 * m ** 2) - 19.0 / (360.0 * m ** 3))
    return float(out[0]) if scalar else out


def _gauss_nodes(n):
    if n not in _GAUSS:
        _GAUSS[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS[n]


class CountModel(object):
    """A belief about one rate, and everything an acquisition might ask of it."""

    name = "model"
    label = "model"
    conjugate = True

    # -- belief -------------------------------------------------------------

    def prior_from_moments(self, mean, var):
        """Initialise a belief with the given mixing mean and variance."""
        raise NotImplementedError

    def fit_population(self, rates):
        """Initialise a belief by maximum likelihood on a sample of rates.

        Where a population of comparable rates has been observed, this is a better prior
        than two matched moments and a fairer comparison between families: each model is
        allowed the best description of the population its own parameters can express,
        rather than being forced through a summary chosen to suit somebody else's family.
        The default falls back to the moments, which is what a family with no closed-form
        fit would do.
        """
        rates = np.asarray(rates, dtype=float)
        return self.prior_from_moments(float(rates.mean()), float(rates.var()))

    def update(self, belief, y, f):
        raise NotImplementedError

    def rate_mean(self, belief):
        raise NotImplementedError

    def rate_moments(self, belief):
        """``(mean, variance)`` of the rate under the belief."""
        raise NotImplementedError

    def sample_rate(self, belief, n, rng):
        raise NotImplementedError

    # -- predictive ---------------------------------------------------------

    def support(self, belief, f, tail_tol=1e-9):
        raise NotImplementedError

    def logpmf(self, belief, f, y):
        raise NotImplementedError

    # -- acquisition ingredients -------------------------------------------

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        y = self.support(belief, f)
        lp = self.logpmf(belief, f, y)
        return -float(np.sum(np.exp(lp) * lp))

    def aleatoric_entropy(self, belief, f):
        raise NotImplementedError

    def eig(self, belief, f):
        return self.predictive_entropy(belief, f) - self.aleatoric_entropy(belief, f)

    def variance_decomposition(self, belief, f):
        m, v = self.rate_moments(belief)
        return float(f * m), float(f * f * v)

    def fisher_dopt(self, belief, f):
        """Local D-optimality in the variance-stabilising coordinate of a Poisson rate."""
        m, v = self.rate_moments(belief)
        return float(np.log1p(f * v / max(m, 1e-12)))


# --------------------------------------------------------------------------

class GIGPoisson(CountModel):
    """Generalised inverse Gaussian mixing. Conjugate; the predictive is the Sichel law."""

    name = "gig-poisson"
    label = "GIG-Poisson"
    conjugate = True

    def __init__(self, order=-0.5):
        self.order = float(order)

    def prior_from_moments(self, mean, var):
        """Solve for the concentration that gives the requested variance at that mean.

        The order is held at its default and the scale follows from the mean, so one
        parameter is left to carry the variance. The equation is monotone in ``omega`` over
        the range used here, so a bisection is enough and no starting value is needed.
        """
        from .gigpoisson import gig_from_mean
        lo, hi = 1e-3, 1e3

        def vom(omega):
            m, v = gig_moments(gig_from_mean(self.order, mean, omega))
            return v / m

        target = var / mean
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            if vom(mid) > target:
                lo = mid
            else:
                hi = mid
        return gig_from_mean(self.order, mean, np.sqrt(lo * hi))

    def fit_population(self, rates):
        from scipy.stats import geninvgauss
        p, omega, _, eta = geninvgauss.fit(np.asarray(rates, dtype=float), floc=0.0)
        # SciPy's (p, b, scale) is our (order, omega, eta); invert eta = sqrt(a/b).
        return {"alpha": float(p), "a": float(eta * omega), "b": float(omega / eta)}

    def update(self, belief, y, f):
        return gig_posterior(belief, y, f)

    def rate_mean(self, belief):
        return gig_moments(belief)[0]

    def rate_moments(self, belief):
        return gig_moments(belief)

    def sample_rate(self, belief, n, rng):
        return gig_sample(belief, n, rng)

    def support(self, belief, f, tail_tol=1e-9):
        return sichel_support(belief, f, tail_tol=tail_tol)

    def logpmf(self, belief, f, y):
        return sichel_logpmf(np.atleast_1d(np.asarray(y, dtype=float)), belief, f)

    def aleatoric_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        alpha, a, b = belief["alpha"], belief["a"], belief["b"]

        def h(s):
            return alpha * s - 0.5 * (a * np.exp(-s) + b * np.exp(s))

        mode = ((alpha - 1.0) + np.sqrt((alpha - 1.0) ** 2 + a * b)) / b
        s_mode = float(np.log(mode)) if mode > 0 else float(np.log(np.sqrt(a / b)))
        target = h(s_mode) - LOG_DENSITY_DROP
        bounds = []
        for direction in (-1.0, 1.0):
            step, far = 1.0, s_mode
            while h(far) > target and step < 1e6:
                far = s_mode + direction * step
                step *= 2.0
            near = s_mode
            for _ in range(60):
                mid = 0.5 * (near + far)
                near, far = (mid, far) if h(mid) > target else (near, mid)
            bounds.append(far)
        s0, s1 = bounds
        x, w = _gauss_nodes(QUAD_NODES)
        s = 0.5 * (s1 - s0) * x + 0.5 * (s1 + s0)
        lam = np.exp(s)
        dens = np.exp(gig_logpdf(lam, belief)) * lam
        return float(0.5 * (s1 - s0) * np.sum(w * dens * poisson_entropy(f * lam)))


# --------------------------------------------------------------------------

class GammaPoisson(CountModel):
    """Gamma mixing. Conjugate; the predictive is negative binomial.

    Two parameters, so the predictive mean and variance are both free but the tail is then
    determined. This is the boundary of the GIG family, and the model whose extra parameter
    the paper is arguing for.
    """

    name = "gamma-poisson"
    label = "gamma-Poisson"
    conjugate = True

    def prior_from_moments(self, mean, var):
        # Gamma(shape, rate): mean = a/b, var = a/b^2.
        return {"a": float(mean * mean / var), "b": float(mean / var)}

    def fit_population(self, rates):
        from scipy.stats import gamma as gamma_dist
        shape, _, scale = gamma_dist.fit(np.asarray(rates, dtype=float), floc=0.0)
        return {"a": float(shape), "b": float(1.0 / scale)}

    def update(self, belief, y, f):
        return {"a": belief["a"] + float(y), "b": belief["b"] + float(f)}

    def rate_mean(self, belief):
        return float(belief["a"] / belief["b"])

    def rate_moments(self, belief):
        a, b = belief["a"], belief["b"]
        return float(a / b), float(a / (b * b))

    def sample_rate(self, belief, n, rng):
        return rng.gamma(belief["a"], 1.0 / belief["b"], size=n)

    def support(self, belief, f, tail_tol=1e-9):
        a, s = belief["a"], f / belief["b"]
        if s <= 0:
            return np.zeros(1, dtype=float)
        ymax = int(nbinom.isf(tail_tol, a, 1.0 / (1.0 + s))) + 16
        return np.arange(min(ymax, 4_000_000) + 1, dtype=float)

    def logpmf(self, belief, f, y):
        a, s = belief["a"], f / belief["b"]
        y = np.atleast_1d(np.asarray(y, dtype=float))
        l1 = np.log1p(s)
        return (gammaln(a + y) - gammaln(a) - gammaln(y + 1.0)
                - a * l1 + y * (np.log(s) - l1))

    def aleatoric_entropy(self, belief, f):
        """Quadrature over the gamma prior, in log-rate to match the other members."""
        if f <= 0.0:
            return 0.0
        a, b = belief["a"], belief["b"]
        # Gamma is log-concave in log-lam: h(s) = a s - b e^s.
        s_mode = float(np.log(a / b))
        target = a * s_mode - b * np.exp(s_mode) - LOG_DENSITY_DROP

        def h(s):
            return a * s - b * np.exp(s)

        bounds = []
        for direction in (-1.0, 1.0):
            step, far = 1.0, s_mode
            while h(far) > target and step < 1e6:
                far = s_mode + direction * step
                step *= 2.0
            near = s_mode
            for _ in range(60):
                mid = 0.5 * (near + far)
                near, far = (mid, far) if h(mid) > target else (near, mid)
            bounds.append(far)
        s0, s1 = bounds
        x, w = _gauss_nodes(QUAD_NODES)
        s = 0.5 * (s1 - s0) * x + 0.5 * (s1 + s0)
        lam = np.exp(s)
        log_dens = a * np.log(b) - gammaln(a) + a * s - b * lam
        return float(0.5 * (s1 - s0) * np.sum(w * np.exp(log_dens)
                                              * poisson_entropy(f * lam)))

    def eig(self, belief, f):
        """Per-term route: average the exact gamma-to-gamma KL over the predictive.

        Every cancellation stays inside one summand, which is the arrangement Remark~1
        recommends and the one the companion project found necessary at large predictive
        mean.
        """
        a, b = belief["a"], belief["b"]
        y = self.support(belief, f)
        a1 = a + y
        kl = ((a1 - a) * digamma(a1) - gammaln(a1) + gammaln(a)
              + a * (np.log(b + f) - np.log(b)) + a1 * (b - (b + f)) / (b + f))
        return float(np.sum(np.exp(self.logpmf(belief, f, y)) * kl))


# --------------------------------------------------------------------------

class LognormalPoisson(CountModel):
    """Lognormal mixing, carried on a grid. Not conjugate.

    The posterior has no closed form, so the belief is a set of weights on a fixed grid in
    log-rate, updated by multiplying by the Poisson likelihood and renormalising. Every
    predictive quantity is then a quadrature over that grid. This is the honest cost of
    leaving the conjugate class: nothing is intractable, but nothing is free either, and the
    grid has to be wide enough to hold the posterior for the whole campaign.
    """

    name = "lognormal-poisson"
    label = "lognormal-Poisson"
    conjugate = False

    def __init__(self, n_grid=128, width=8.0):
        self.n_grid = int(n_grid)
        self.width = float(width)

    def prior_from_moments(self, mean, var):
        sig2 = float(np.log1p(var / (mean * mean)))
        mu = float(np.log(mean) - 0.5 * sig2)
        sig = np.sqrt(sig2)
        s = np.linspace(mu - self.width * sig, mu + self.width * sig, self.n_grid)
        logw = -0.5 * ((s - mu) / sig) ** 2
        logw -= logsumexp(logw)
        return {"s": s, "logw": logw}

    def fit_population(self, rates):
        from scipy.stats import lognorm
        sig, _, scale = lognorm.fit(np.asarray(rates, dtype=float), floc=0.0)
        mu = float(np.log(scale))
        s = np.linspace(mu - self.width * sig, mu + self.width * sig, self.n_grid)
        logw = -0.5 * ((s - mu) / sig) ** 2
        return {"s": s, "logw": logw - logsumexp(logw)}

    def update(self, belief, y, f):
        lam = np.exp(belief["s"])
        ll = -f * lam + y * np.log(f * lam) - gammaln(y + 1.0)
        logw = belief["logw"] + ll
        return {"s": belief["s"], "logw": logw - logsumexp(logw)}

    def _weights(self, belief):
        return np.exp(belief["logw"])

    def rate_mean(self, belief):
        return float(np.sum(self._weights(belief) * np.exp(belief["s"])))

    def rate_moments(self, belief):
        w, lam = self._weights(belief), np.exp(belief["s"])
        m = float(np.sum(w * lam))
        return m, float(np.sum(w * lam * lam) - m * m)

    def sample_rate(self, belief, n, rng):
        idx = rng.choice(belief["s"].size, size=n, p=self._weights(belief))
        return np.exp(belief["s"][idx])

    def support_and_logpmf(self, belief, f, tail_tol=1e-9):
        """Support and its log mass in one pass, grown until the tail is inside tolerance.

        A moment-based guess is not conservative for a heavy-tailed mixture, and a support
        that is too short biases this model's predictive moments low, which would make it
        look worse for a reason that is ours rather than its own. Returning the mass
        alongside the support avoids evaluating the mixture twice, which is most of the cost
        of an acquisition value for this model.
        """
        m, v = self.rate_moments(belief)
        ymax = int(f * m + 12.0 * np.sqrt(max(f * m + f * f * v, 0.0)) + 64.0)
        for _ in range(20):
            ymax = min(ymax, 200_000)
            y = np.arange(ymax + 1, dtype=float)
            lp = self.logpmf(belief, f, y)
            if 1.0 - np.exp(logsumexp(lp)) < tail_tol or ymax == 200_000:
                return y, lp
            ymax *= 2
        return y, lp

    def support(self, belief, f, tail_tol=1e-9):
        return self.support_and_logpmf(belief, f, tail_tol)[0]

    def predictive_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        _, lp = self.support_and_logpmf(belief, f)
        return -float(np.sum(np.exp(lp) * lp))

    def logpmf(self, belief, f, y):
        """Mixture of Poissons over the grid, in log space."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        lam = np.exp(belief["s"])
        mu = f * lam
        ll = (-mu[None, :] + y[:, None] * np.log(mu)[None, :]
              - gammaln(y + 1.0)[:, None])
        return logsumexp(ll + belief["logw"][None, :], axis=1)

    def aleatoric_entropy(self, belief, f):
        if f <= 0.0:
            return 0.0
        lam = np.exp(belief["s"])
        return float(np.sum(self._weights(belief) * poisson_entropy(f * lam)))


MODELS = (GIGPoisson, GammaPoisson, LognormalPoisson)


def build_model(name):
    for cls in MODELS:
        if cls.name == name:
            return cls()
    raise ValueError("unknown model {!r}".format(name))
