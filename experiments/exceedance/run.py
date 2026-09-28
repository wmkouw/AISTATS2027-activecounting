"""Does the third parameter pay when the decision is a threshold?

Table 1 of the manuscript locates what the third parameter moves: the far tail and the mass
at zero, not the centre. Every score the paper reports -- held-out predictive, rate error,
top-``m`` regret -- weighs the centre. So the model's advantage has been measured with
instruments built not to see it, which is the most likely reason the mixing law is worth
0.012 nats in the bulk study and ties the lognormal in the gamma-ray study.

A threshold decision weighs the tail. It is also the decision these domains actually make:

    Safecast      is a cell above the evacuation limit?
    Fermi-LAT     is a source above the catalogue detection threshold?
    bulk sampling is a block above the cutoff grade?

Two studies here, answering two different questions.

**A. What the scores can see** (``results/discrimination.csv``). No sampling and no
allocation. The truth is a GIG-Poisson predictive computed exactly; the alternatives are
fitted to its first two moments, as the allocation studies fit them. Each is scored by the
cross-entropy the paper reports and by the relative error in exceedance at a range of
thresholds. The question is how far apart the two scores place the same pair of models.

**B. Whether it survives a budget** (``results/allocation.csv``). The same three mixing laws
under the proposed acquisition, allocating a scarce budget over contexts, scored by a
*threshold* loss: the Brier score of the predicted exceedance probability against what the
held-out counts actually did, plus the false-positive and false-negative rate of the
resulting flag. If the model axis pays anywhere, it pays here.

Run: python experiments/exceedance/run.py
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)

from methods.countmodels import GammaPoisson, GIGPoisson, LognormalPoisson  # noqa: E402
from methods.gigpoisson import gig_from_mean, gig_moments, gig_sample       # noqa: E402

SEED = 0
F_EVAL = 4.0
#: Exceedance quantiles of the true predictive. The last is the one a safety limit sits at.
QUANTILES = (0.90, 0.95, 0.99, 0.999)
#: Orders spanning the range the measured fields show: Newsgroups +1.73, Fermi -1.43,
#: Safecast -2.07. The manuscript's simulators all generate at -0.5.
ORDERS = (-0.5, -1.0, -1.5, -2.0)
GRID = np.arange(0.0, 6000.0)


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.basename(path), len(rows)))


def gig_with(order, mean, phi):
    """The GIG of this order whose mean is ``mean`` and whose ``V/E`` is ``phi``."""
    lo, hi = 1e-4, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        m, v = gig_moments(gig_from_mean(order, mean, mid))
        if v / m > phi:
            lo = mid
        else:
            hi = mid
    law = gig_from_mean(order, mean, np.sqrt(lo * hi))
    m, v = gig_moments(law)
    if abs(v / m - phi) > 1e-3 * phi:
        raise ValueError("V/E=%s unreachable at order %s mean %s" % (phi, order, mean))
    return law


def pmf(model, belief, f=F_EVAL):
    p = np.exp(np.asarray(model.logpmf(belief, f, GRID), dtype=float))
    return p / p.sum()


# --------------------------------------------------------------------------
# A. what the scores can see
# --------------------------------------------------------------------------

def study_discrimination(phi=4.0, mean=1.0):
    rows = []
    gig = GIGPoisson()
    print("  {:>7} {:<20} {:>11}  {}".format(
        "order", "model", "KL(true||m)",
        "  ".join("q%-6g" % q for q in QUANTILES)))
    for order in ORDERS:
        try:
            law = gig_with(order, mean, phi)
        except ValueError:
            print("    order %+.1f: V/E=%.1f unreachable, skipped" % (order, phi))
            continue
        p_true = pmf(gig, dict(law))
        cdf = np.cumsum(p_true)
        thr = [float(GRID[np.searchsorted(cdf, q)]) for q in QUANTILES]
        true_tail = [float(p_true[GRID >= t].sum()) for t in thr]
        m, v = gig_moments(law)

        for name, model in (("gamma-Poisson", GammaPoisson()),
                            ("lognormal-Poisson", LognormalPoisson())):
            b = model.prior_from_moments(float(m), float(v))
            p = pmf(model, b)
            kl = float(np.sum(p_true * (np.log(np.maximum(p_true, 1e-300))
                                        - np.log(np.maximum(p, 1e-300)))))
            tails = [float(p[GRID >= t].sum()) for t in thr]
            rel = [abs(a - b_) / b_ for a, b_ in zip(tails, true_tail)]
            rows.append((order, float(m), float(v / m), name, kl) + tuple(thr)
                        + tuple(true_tail) + tuple(tails) + tuple(rel))
            print("  {:>+7.1f} {:<20} {:>11.5f}  {}".format(
                order, name, kl, "  ".join("%6.0f%%" % (100 * r) for r in rel)))
    return rows


# --------------------------------------------------------------------------
# B. whether it survives a budget
# --------------------------------------------------------------------------

EXPOSURES = np.array([0.5, 1.0, 2.0, 4.0])


def exceedance_scores(model, belief, taus, truth, rng, n_draw=400):
    """How wrong is the predicted exceedance probability, once a budget has been spent?

    Scored at several thresholds from one allocation, because the allocation does not depend
    on the threshold and re-running it per threshold buys nothing.

    Two scores, chosen so that neither degenerates at a rare event. A Brier score saturates:
    everyone predicts near zero, everyone is right. A *relative* error does the opposite and
    diverges, because the true probability at a faint context is near zero and sits in the
    denominator; the first version of this study reported relative errors of 10^9 for that
    reason. What is left is the absolute error against the exceedance probability the true
    rate implies -- exact, since the truth is known -- and a log-score on held-out events,
    which is proper and stays sensitive where the event is rare.
    """
    from scipy.stats import poisson
    p = pmf(model, belief)
    y = rng.poisson(F_EVAL * truth, size=n_draw).astype(float)
    out = []
    for tau in taus:
        p_hat = float(np.clip(p[GRID >= tau].sum(), 1e-9, 1 - 1e-9))
        p_true = float(poisson.sf(tau - 1, F_EVAL * truth))
        hit = (y >= tau).astype(float)
        logs = -float(np.mean(hit * np.log(p_hat) + (1 - hit) * np.log(1 - p_hat)))
        out.append((abs(p_hat - p_true), logs, float(np.mean((p_hat - hit) ** 2))))
    return out


def attainable_phi(order, means, want=4.0):
    """The largest index of dispersion reachable at this order for every context mean."""
    phi = want
    while phi > 0.15:
        try:
            for mu in means:
                gig_with(order, float(mu), phi)
            return phi
        except ValueError:
            phi *= 0.8
    return None


#: Where the thresholds sit in the pooled true predictive.
TAU_QUANTILES = (0.75, 0.90, 0.95)


def run_allocation(order, mean, n_ctx, budget, n_env):
    """Allocate once per episode and model; score the result at every threshold."""
    means = np.geomspace(mean / 3.0, mean * 3.0, n_ctx)
    phi = attainable_phi(order, means)
    if phi is None:
        raise ValueError("no attainable dispersion at order %s" % order)
    laws = [gig_with(order, float(mu), phi) for mu in means]
    gig = GIGPoisson()
    pooled = np.mean([pmf(gig, dict(l)) for l in laws], axis=0)
    cdf = np.cumsum(pooled)
    taus = [float(GRID[np.searchsorted(cdf, q)]) for q in TAU_QUANTILES]

    models = [("GIG-Poisson", GIGPoisson()), ("gamma-Poisson", GammaPoisson()),
              ("lognormal-Poisson", LognormalPoisson())]
    mom = [tuple(float(x) for x in gig_moments(l)) for l in laws]
    out = {nm: [] for nm, _ in models}
    for e in range(n_env):
        r0 = np.random.default_rng(1000 + e)
        truths = np.array([float(gig_sample(l, 1, r0)[0]) for l in laws])
        for nm, model in models:
            rng = np.random.default_rng(5000 + e)
            bel = [model.prior_from_moments(a, b) for a, b in mom]
            spent = 0.0
            while True:
                av = EXPOSURES[EXPOSURES <= budget - spent + 1e-9]
                if av.size == 0:
                    break
                best, arg = -np.inf, None
                for k in range(n_ctx):
                    for f in av:
                        s_ = model.eig(bel[k], float(f)) / float(f)
                        if s_ > best:
                            best, arg = s_, (k, float(f))
                k, f = arg
                bel[k] = model.update(bel[k], rng.poisson(f * truths[k]), f)
                spent += f
            per_ctx = [exceedance_scores(model, bel[k], taus, truths[k], rng)
                       for k in range(n_ctx)]
            # average over contexts, keeping the threshold axis
            out[nm].append([[float(np.mean([c[i][j] for c in per_ctx]))
                             for j in range(3)] for i in range(len(taus))])
    return taus, phi, {nm: np.asarray(v, dtype=float) for nm, v in out.items()}


def study_allocation(n_ctx=8, budget=8.0, n_env=30):
    rows = []
    print("\n  {:>6} {:>5} {:>5} {:<20} {:>10} {:>7} {:>10} {:>7}".format(
        "order", "V/E", "tau", "model", "|dP|", "t", "log-score", "t"))
    for order in ORDERS:
        try:
            taus, phi, res = run_allocation(order, 1.0, n_ctx, budget, n_env)
        except ValueError as exc:
            print("  {:>+6.1f} skipped: {}".format(order, exc))
            continue
        ref = res["GIG-Poisson"]
        for i, tau in enumerate(taus):
            for nm, a in res.items():
                ts = []
                for j in (0, 1):
                    d = ref[:, i, j] - a[:, i, j]
                    ts.append(0.0 if nm == "GIG-Poisson" else
                              float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))
                                                  + 1e-15)))
                rows.append((order, phi, TAU_QUANTILES[i], tau, nm,
                             a[:, i, 0].mean(), ts[0], a[:, i, 1].mean(), ts[1],
                             a[:, i, 2].mean(),
                             a[:, i, 0].std(ddof=1) / np.sqrt(len(a))))
                print("  {:>+6.1f} {:>5.2f} {:>5.0f} {:<20} {:>10.5f} {:>7.1f} "
                      "{:>10.4f} {:>7.1f}".format(order, phi, tau, nm,
                                                  a[:, i, 0].mean(), ts[0],
                                                  a[:, i, 1].mean(), ts[1]))
            print()
    return rows


def main():
    rd = os.path.join(HERE, "results")
    t0 = time.time()
    print("A. what the scores can see (analytic, no sampling)")
    cols = (["order", "mean", "v_over_e", "model", "kl"]
            + ["thr_q%g" % q for q in QUANTILES]
            + ["true_q%g" % q for q in QUANTILES]
            + ["model_q%g" % q for q in QUANTILES]
            + ["relerr_q%g" % q for q in QUANTILES])
    save_csv(os.path.join(rd, "discrimination.csv"), cols, study_discrimination())

    print("\nB. whether it survives a budget  [{:.0f}s]".format(time.time() - t0))
    save_csv(os.path.join(rd, "allocation.csv"),
             ["order", "v_over_e", "tau_q", "tau", "model", "abs_dp", "t_absdp",
              "log_score", "t_logscore", "brier", "sem_absdp"], study_allocation())
    print("\nDone in {:.0f}s.".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
