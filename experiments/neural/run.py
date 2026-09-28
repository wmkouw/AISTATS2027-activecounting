"""Neural spike counts as a rate field, and whether the manuscript's model applies to them.

Three studies, in decreasing order of how much they depend on data we do not yet have.

**1. The measured field** (``results/lgn_fits.csv``). Fit the three mixing laws to the mouse
LGN firing rates, one observation per (cell, direction) pair, exactly as Section 4.5 fits the
photon fluxes and the Safecast cells. This is the one that belongs in the paper. It needs the
CRCNS ``lgn-1`` archive, which requires a free registration and so is not in the repository;
without it this study reports that it cannot run and the other two proceed.

**2. A structural check on the parameterisation** (``results/fano_profile.csv``). Taouali et
al. report that the inverse dispersion is *tuned*, and that the resulting Fano factor is
bimodal in stimulus direction -- lowest at the preferred direction, peaking off it (their
Fig. 8C). ``taouali.py`` is parameterised to reproduce that, and this writes the profile out
so the claim can be checked rather than taken on trust.

**3. Does the doubly stochastic reading break Proposition 2?** This is the obstacle. Taouali
et al.'s overdispersion is doubly stochastic: the rate is redrawn on every trial. The
manuscript's rate is fixed per context, and Proposition 2 says the posterior concentrates on
it with ``F_t Var[lambda] -> lambda*``. Under a per-trial redraw there is no ``lambda*`` to
concentrate on. Rather than argue the point, this generates counts both ways from the same
tuning curves and measures what the manuscript's own estimator does to each
(``results/concentration.csv``).

Nothing here substitutes simulated rates for measured ones: study 1 uses only the archive,
studies 2 and 3 use only the generator, and every row carries its provenance.

Run: python experiments/neural/run.py
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "rate-fields"))

import taouali as T                                                   # noqa: E402
from compare import fit_fields, write_rows                            # noqa: E402
from fields import NEURAL_FIELDS                                      # noqa: E402
from methods.countmodels import GIGPoisson                            # noqa: E402
from methods.gigpoisson import gig_moments                            # noqa: E402

SEED = 0
#: Counting window, in the units the tuning curves are expressed in. Taouali et al. count
#: over the stimulus duration, so one trial is one unit of exposure and Corollary 1 makes
#: trials and window length interchangeable.
DT = 1.0


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.basename(path), len(rows)))


# --------------------------------------------------------------------------

def study_measured_field():
    """Fit the mixing laws to the measured LGN rates, if the archive is present."""
    try:
        return fit_fields(NEURAL_FIELDS), None
    except FileNotFoundError as exc:
        return [], str(exc)


def study_fano_profile():
    """Write out the Fano profile the parameterisation produces, against the paper's anchor."""
    deg, ff = T.fano_profile()
    i0 = int(np.argmin(np.abs(deg)))
    anchor = 1.0 + T.PD_MEAN / T.PD_PHI
    peak = int(np.argmax(ff))
    print("    at the preferred direction FF = {:.3f}; the paper quotes "
          "1 + {:.2f}/{:.2f} = {:.3f}".format(ff[i0], T.PD_MEAN, T.PD_PHI, anchor))
    print("    profile is bimodal: peaks at {:+.0f} deg at FF = {:.2f}, falls to {:.2f} "
          "at 180 deg".format(deg[peak], ff[peak], ff[-1]))
    return [(float(d), float(f)) for d, f in zip(deg, ff)]


def study_concentration(n_ctx=60, max_trials=256, n_rep=40):
    """Is the posterior *calibrated*, under each reading of where the variability sits?

    A first version of this asked whether ``F_t Var[lambda | D_t]`` converges to the target,
    the constant of Proposition 2. It does, under **both** readings, and so it separates
    nothing: the posterior variance is driven by the accumulated count, which grows at the
    same rate whether the rate was redrawn or not. What the redraw changes is not the size of
    the posterior variance but whether it is *honest*.

    So the diagnostic is the standardised error ``z = (E[lambda | D] - target) / sd``. A
    calibrated posterior has ``Var(z) = 1`` and covers a nominal 95 per cent interval 95 per
    cent of the time. A posterior that treats overdispersed counts as Poisson is
    overconfident, and the inflation should track the Fano factor.

    Under the fixed-rate reading the target is the single drawn rate. Under the redraw there
    is no such rate, so the target is the mean the counts are generated around, which is the
    quantity the posterior mean converges to in either case.
    """
    rng = np.random.default_rng(SEED)
    _, mean, phi = T.tuning_curves(rng)
    flat_m, flat_p = mean.ravel(), phi.ravel()
    pick = rng.choice(flat_m.size, size=n_ctx, replace=False)
    model = GIGPoisson()
    pop_m = float(np.mean(flat_m))
    pop_v = float(np.var(flat_m) + np.mean(flat_m ** 2 / flat_p))
    prior = model.prior_from_moments(pop_m, pop_v)

    checkpoints = [4, 16, 64, 256]
    rows = []
    for reading, fixed in (("fixed rate", True), ("redrawn per trial", False)):
        acc = {n: {"z": [], "dex": [], "ff": []} for n in checkpoints}
        for r in range(n_rep):
            rr = np.random.default_rng(1000 + r)
            for k in pick:
                m_k, p_k = float(flat_m[k]), float(flat_p[k])
                ff_k = 1.0 + m_k * DT / p_k
                if fixed:
                    target = float(rr.gamma(shape=p_k, scale=m_k / p_k))
                    y = rr.poisson(target * DT, size=max_trials)
                else:
                    target = m_k
                    lam = rr.gamma(shape=p_k, scale=m_k / p_k, size=max_trials)
                    y = rr.poisson(lam * DT)
                bel = dict(prior)
                seen = 0
                for n in checkpoints:
                    for t in range(seen, n):
                        bel = model.update(bel, float(y[t]), DT)
                    seen = n
                    mu, var = gig_moments(bel)
                    sd = float(np.sqrt(max(var, 1e-300)))
                    acc[n]["z"].append((float(mu) - target) / sd)
                    acc[n]["dex"].append(abs(np.log10(max(float(mu), 1e-12))
                                             - np.log10(max(target, 1e-12))))
                    acc[n]["ff"].append(ff_k)
        line = []
        for n in checkpoints:
            z = np.asarray(acc[n]["z"])
            varz = float(np.var(z))
            cover = float(np.mean(np.abs(z) <= 1.959964))
            rows.append((reading, n, varz, cover, float(np.mean(acc[n]["dex"])),
                         float(np.mean(acc[n]["ff"]))))
            line.append("n={}: Var(z)={:.2f} cover={:.0%}".format(n, varz, cover))
        print("    {:<18} {}".format(reading, "  ".join(line)))
    print("    mean Fano factor across the contexts used: {:.2f}"
          .format(float(np.mean(acc[checkpoints[0]]["ff"]))))
    return rows


def main():
    rd = os.path.join(HERE, "results")
    t0 = time.time()

    print("1. the measured LGN field")
    rows, err = study_measured_field()
    if err:
        print("   NOT RUN. " + err.replace("\n", "\n   "))
    else:
        write_rows(os.path.join(rd, "lgn_fits.csv"), rows)

    print("\n2. structural check on the parameterisation (their Fig. 8C)")
    save_csv(os.path.join(rd, "fano_profile.csv"),
             ["degrees_from_pd", "fano_factor"], study_fano_profile())

    print("\n3. is the posterior calibrated under each reading?  [{:.0f}s]"
          .format(time.time() - t0))
    save_csv(os.path.join(rd, "concentration.csv"),
             ["reading", "trials", "var_z", "coverage_95", "mean_dex", "mean_fano"],
             study_concentration())

    print("\nDone in {:.0f}s.".format(time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
