"""A Gaussian process over log-rates with a Poisson observation model.

The non-conjugate alternative the paper's argument is aimed at. A latent field
``g = (g_1, ..., g_K)`` carries a Gaussian process prior across contexts, the rate of a
context is ``lam_k = exp(g_k)``, and a measurement of exposure ``e`` at context ``k``
returns ``y ~ Poisson(e * exp(g_k))``. The output is discrete, so the posterior is not
Gaussian and is approximated; the kernel is what makes this different in kind from every
model in ``countmodels``, because it lets an observation at one context move the belief at
another.

**Sufficient statistics.** Summing Poissons of a common rate, the likelihood contributed by
a context depends on the history only through its accumulated count ``T_k`` and accumulated
exposure ``F_k``:

    log p(data_k | g_k) = T_k g_k - F_k exp(g_k) + const.

That is the same pair Corollary 1 gives for the conjugate model, so the two carry the same
state and differ only in what they do with it.

**Laplace approximation.** The log posterior of the latent field is

    -1/2 (g - m)' K^{-1} (g - m) + sum_k [ T_k g_k - F_k exp(g_k) ],

concave in ``g`` because the second term is, so Newton's method converges to the unique mode
and the curvature there is ``W = diag(F_k exp(g_k))``. The iteration and the predictive
variance follow Rasmussen and Williams, Algorithm 3.1, with the Poisson likelihood in place
of the probit: everything is done through the numerically stable ``B = I + W^{1/2} K W^{1/2}``
factorisation rather than by inverting ``K``.

**The kernel.** Over whatever features the study gives each context. The moments an agent is
handed at reset are always available and already encode the structure that matters -- in the
gamma-ray study every source of a class is given its class prior, so contexts of a class sit
at the same point and the kernel pools them -- so the default feature map is the log prior
mean and the log prior index of dispersion, standardised. An environment may override it by
offering ``context_features``.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

__all__ = ["GPPoisson", "rbf_kernel", "default_features"]

#: Ridge added to the kernel diagonal for conditioning.
JITTER = 1e-8
#: Newton iterations and the gradient norm that stops them.
MAX_NEWTON, NEWTON_TOL = 60, 1e-9


def default_features(moments):
    """Standardised ``(log prior mean, log prior index of dispersion)`` per context."""
    m = np.asarray([mv[0] for mv in moments], dtype=float)
    v = np.asarray([mv[1] for mv in moments], dtype=float)
    x = np.column_stack([np.log(np.maximum(m, 1e-12)),
                         np.log(np.maximum(v / np.maximum(m, 1e-12), 1e-12))])
    sd = x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (x - x.mean(axis=0)) / sd


def rbf_correlation(x, lengthscale=1.0, nugget=0.2):
    """Squared-exponential *correlation* with an independent component.

    A correlation rather than a covariance, because the marginal prior variance of each
    context is pinned separately to the one the study supplies; see :meth:`GPPoisson.reset`.
    The nugget is the fraction of each latent that no other context explains, and it is what
    keeps the model honest where contexts coincide: in the gamma-ray study every source of a
    class sits at the same feature point, and without an independent component the model
    would claim that measuring one of them measures the rest exactly.
    """
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    r = (1.0 - nugget) * np.exp(-0.5 * d2 / (lengthscale ** 2))
    return r + (nugget + JITTER) * np.eye(x.shape[0])


class GPPoisson(object):
    """Joint belief over ``K`` log-rates: a GP prior, a Poisson likelihood, Laplace."""

    name = "gp-poisson"
    label = "GP-Poisson"
    conjugate = False

    def __init__(self, amplitude=None, lengthscale=1.0, nugget=1e-3):
        self.amplitude = amplitude
        self.lengthscale = float(lengthscale)
        self.nugget = float(nugget)

    # -- setup -------------------------------------------------------------

    def reset(self, moments, features=None):
        """Prior mean and kernel from the per-context moments the study supplies.

        Every model in the comparison is handed the same two prior moments per context, so
        this one matches them exactly: a lognormal moment match sets the latent mean and the
        latent variance, and the kernel is built as a *correlation* scaled by those
        per-context variances. The marginal prior on each rate is then the same as the other
        families start from, and the only thing the kernel adds is the correlation between
        contexts. Nothing is given away to the Gaussian process at the prior.
        """
        mom = [(float(a), float(b)) for a, b in moments]
        self.K_ctx = len(mom)
        mean = np.asarray([a for a, _ in mom], dtype=float)
        var = np.asarray([b for _, b in mom], dtype=float)
        s2 = np.log1p(np.maximum(var, 1e-18) / np.maximum(mean, 1e-12) ** 2)
        self.prior_sd = np.sqrt(np.maximum(s2, 1e-12))
        self.prior_mean = np.log(np.maximum(mean, 1e-12)) - 0.5 * s2
        self.X = default_features(mom) if features is None else np.asarray(features, float)
        self.T = np.zeros(self.K_ctx)
        self.F = np.zeros(self.K_ctx)
        self.g_hat = None
        self._theta = (self.lengthscale, self.nugget)
        self._set_kernel(*self._theta)
        self._n_obs = 0
        self._next_tune = 4
        self._fit()

    def _set_kernel(self, lengthscale, nugget):
        r = rbf_correlation(self.X, lengthscale, nugget)
        self.Kmat = np.outer(self.prior_sd, self.prior_sd) * r

    def observe(self, k, y, f):
        """Accumulate the pair and refit. Every context's belief may move."""
        self.T[k] += float(y)
        self.F[k] += float(f)
        self._n_obs += 1
        if self._n_obs >= self._next_tune:
            self._next_tune *= 2
            self._tune()
        self._fit()

    # -- hyperparameters ---------------------------------------------------

    def log_marginal(self):
        """Laplace approximation to ``log p(data | theta)``, Rasmussen and Williams 3.32."""
        g, m, K = self.g_hat, self.prior_mean, self.Kmat
        lam = self.F * np.exp(g)
        ll = float(np.sum(self.T * g - lam - gammaln(self.T + 1.0)))
        W = np.maximum(lam, 0.0)
        sw = np.sqrt(W)
        B = np.eye(self.K_ctx) + sw[:, None] * K * sw[None, :]
        c, low = cho_factor(B, lower=True)
        d = g - m
        a = cho_solve(cho_factor(K + JITTER * np.eye(self.K_ctx), lower=True), d)
        logdetB = 2.0 * float(np.sum(np.log(np.diag(c))))
        return ll - 0.5 * float(d @ a) - 0.5 * logdetB

    def _tune(self):
        """Refit lengthscale and nugget by the Laplace marginal likelihood.

        On a doubling schedule rather than every observation: the objective needs a Laplace
        fit per evaluation, and the hyperparameters move slowly once a few counts are in.
        The marginal variances stay pinned, so only the correlation is being learned.
        """
        if not np.any(self.F > 0):
            return
        best, best_theta = -np.inf, self._theta
        for ls in (0.25, 0.5, 1.0, 2.0, 4.0):
            for nug in (0.05, 0.2, 0.5, 0.8, 0.98):
                self._set_kernel(ls, nug)
                self.g_hat = None
                self._fit()
                try:
                    v = self.log_marginal()
                except Exception:
                    continue
                if np.isfinite(v) and v > best:
                    best, best_theta = v, (ls, nug)
        self._theta = best_theta
        self._set_kernel(*best_theta)
        self.g_hat = None

    # -- Laplace -----------------------------------------------------------

    def _fit(self):
        g = getattr(self, "g_hat", None)
        if g is None or g.shape[0] != self.K_ctx:
            g = self.prior_mean.copy()
        m, K = self.prior_mean, self.Kmat
        for _ in range(MAX_NEWTON):
            lam = self.F * np.exp(g)
            grad_ll = self.T - lam            # d/dg of the log likelihood
            W = np.maximum(lam, 0.0)          # -d2/dg2, non-negative
            sw = np.sqrt(W)
            B = np.eye(self.K_ctx) + sw[:, None] * K * sw[None, :]
            c, low = cho_factor(B, lower=True)
            b = W * (g - m) + grad_ll
            a = b - sw * cho_solve((c, low), sw * (K @ b))
            g_new = m + K @ a
            step = np.max(np.abs(g_new - g))
            g = g_new
            if step < NEWTON_TOL:
                break
        self.g_hat = g
        lam = self.F * np.exp(g)
        W = np.maximum(lam, 0.0)
        sw = np.sqrt(W)
        B = np.eye(self.K_ctx) + sw[:, None] * K * sw[None, :]
        c, low = cho_factor(B, lower=True)
        # var(g_k) = K_kk - (sw K)_k' B^{-1} (sw K)_k
        V = cho_solve((c, low), sw[:, None] * K)
        self.g_var = np.maximum(np.diag(K) - np.sum((sw[:, None] * K) * V, axis=0), 1e-12)
        self._chol = (c, low)

    # -- what the agent reads ---------------------------------------------

    def latent(self, k):
        """Posterior mean and variance of ``log lam_k``."""
        return float(self.g_hat[k]), float(self.g_var[k])

    def rate_mean(self, k):
        mu, s2 = self.latent(k)
        return float(np.exp(mu + 0.5 * s2))

    def rate_moments(self, k):
        mu, s2 = self.latent(k)
        m = np.exp(mu + 0.5 * s2)
        return float(m), float(m * m * np.expm1(s2))

    def sample_rate(self, k, n, rng):
        mu, s2 = self.latent(k)
        return np.exp(rng.normal(mu, np.sqrt(s2), size=n))

    def logpmf(self, k, f, y, n_quad=48):
        """Predictive mass of a count, by Gauss-Hermite over the latent."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        mu, s2 = self.latent(k)
        x, w = np.polynomial.hermite_e.hermegauss(n_quad)
        g = mu + np.sqrt(s2) * x
        lam = np.maximum(f * np.exp(g), 1e-300)
        logw = np.log(w / np.sqrt(2.0 * np.pi))
        ll = (-lam[None, :] + y[:, None] * np.log(lam)[None, :]
              - gammaln(y + 1.0)[:, None] + logw[None, :])
        mx = ll.max(axis=1, keepdims=True)
        return np.squeeze(mx[:, 0] + np.log(np.sum(np.exp(ll - mx), axis=1)))
